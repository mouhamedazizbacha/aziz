import os
import re
import bisect
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import matplotlib

# Use non-interactive backend for servers/CI
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings("ignore", category=UserWarning)


# =========================
# 1) GPU/CPU DETECTION HELPERS
# =========================
def get_xgb_device() -> str:
    """Return 'cuda' if a NVIDIA GPU seems available, else 'cpu'."""
    try:
        is_kaggle = "KAGGLE_URL_BASE" in os.environ
        if is_kaggle:
            gpu_available = os.environ.get("CUDA_VISIBLE_DEVICES", "") != "-1"
            if gpu_available:
                print("✅ GPU détecté sur Kaggle - Utilisation du GPU pour XGBoost")
                return "cuda"

        # Fallback: check nvidia-smi
        if os.system("nvidia-smi > /dev/null 2>&1") == 0:
            print("✅ GPU NVIDIA détecté - Utilisation du GPU pour XGBoost")
            return "cuda"

        print("⚠️ Aucun GPU détecté - Utilisation du CPU pour XGBoost")
        return "cpu"
    except Exception:
        print("⚠️ Erreur de détection GPU - Utilisation du CPU par défaut")
        return "cpu"


def get_xgb_params_for_device(base_params: dict, device: str) -> dict:
    """Return XGBoost params adjusted for the installed xgboost version and device.

    - For xgboost >= 2.0, prefers 'device' param
    - For older versions, falls back to 'tree_method'/'predictor'
    """
    import xgboost as xgb

    params = dict(base_params)
    version = tuple(int(p) for p in xgb.__version__.split(".")[:2])
    if device == "cuda":
        if version >= (2, 0):
            params["device"] = "cuda"
        else:
            # Older versions (1.x)
            params["tree_method"] = "gpu_hist"
            params["predictor"] = "gpu_predictor"
    return params


device = get_xgb_device()


# =========================
# 2) IO HELPERS AND PATHS
# =========================
def get_outputs_dir() -> str:
    if os.path.isdir("/kaggle/working"):
        path = "/kaggle/working"
    else:
        path = "/workspace/outputs"
    os.makedirs(path, exist_ok=True)
    return path


def read_excel_safely(path: str) -> pd.DataFrame:
    return pd.read_excel(path)


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Load sales and holidays data.

    Order of attempts:
    1) Kaggle input paths
    2) Local /workspace/data paths
    3) Synthetic fallback (fast smoke test)
    Returns (sales_df, holidays_df, source_label)
    """
    # 1) Kaggle
    kaggle_sales = "/kaggle/input/predection-vente/fact_livraisons_vente.xlsx"
    kaggle_holidays = "/kaggle/input/predection-vente/jours_feries_tunisie_2020_2026.xlsx"
    if os.path.isfile(kaggle_sales) and os.path.isfile(kaggle_holidays):
        print("✅ Données chargées depuis Kaggle")
        return read_excel_safely(kaggle_sales), read_excel_safely(kaggle_holidays), "kaggle"

    # 2) Local workspace data
    local_dir = "/workspace/data"
    local_sales = os.path.join(local_dir, "fact_livraisons_vente.xlsx")
    local_holidays = os.path.join(local_dir, "jours_feries_tunisie_2020_2026.xlsx")
    if os.path.isfile(local_sales) and os.path.isfile(local_holidays):
        print("✅ Données chargées depuis /workspace/data")
        return read_excel_safely(local_sales), read_excel_safely(local_holidays), "local"

    # 3) Synthetic fallback
    print("⚠️ Fichiers non trouvés, génération de données synthétiques pour test rapide…")
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", "2025-05-31", freq="D")
    trend = np.linspace(50, 120, len(dates))
    seasonal = 20 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365)
    noise = rng.normal(0, 8, len(dates))
    base = np.maximum(0, trend + seasonal + noise)

    # Ramadan dip (approximate window in 2024 and 2025)
    ramadan_2024 = pd.date_range("2024-03-11", periods=30, freq="D")
    ramadan_2025 = pd.date_range("2025-03-01", periods=30, freq="D")
    dips_mask = np.asarray(dates.isin(ramadan_2024.union(ramadan_2025)), dtype=bool)
    base = base * np.where(dips_mask, 0.75, 1.0)

    # Dattes uplift (Sep-Nov)
    uplift_mask = np.asarray((dates.month >= 9) & (dates.month <= 11), dtype=bool)
    base = base * np.where(uplift_mask, 1.15, 1.0)

    sales_df = pd.DataFrame({
        "date_livraison": dates,
        "quantite_livree": base.round(0).astype(int),
    })

    holidays = []
    # Minimal synthetic holidays: Ramadan days and Aïd
    for d in ramadan_2024:
        holidays.append({"Date": d, "Nom": f"Ramadan Jour {(d - ramadan_2024[0]).days + 1}"})
    for d in ramadan_2025:
        holidays.append({"Date": d, "Nom": f"Ramadan Jour {(d - ramadan_2025[0]).days + 1}"})
    for d in [pd.Timestamp("2024-04-10"), pd.Timestamp("2025-04-01")]:
        holidays.append({"Date": d, "Nom": "Aïd al-Fitr"})

    holidays_df = pd.DataFrame(holidays)
    print("✅ Données synthétiques générées")
    return sales_df, holidays_df, "synthetic"


# =========================
# 3) DATA PREP HELPERS
# =========================
def parse_date(value):
    """Parse dates given possibly as strings in MM/DD/YYYY[ HH:MM:SS] format.
    If already a Timestamp returns as-is; otherwise returns NaT on failure.
    """
    try:
        if isinstance(value, pd.Timestamp):
            return value
        if isinstance(value, str):
            date_part = value.split()[0]
            parts = date_part.split("/")
            if len(parts) == 3:
                month, day, year = map(int, parts)
                if 1 <= month <= 12:
                    return pd.Timestamp(year, month, day)
        return pd.NaT
    except Exception:
        return pd.NaT


def date_palm_weight(month: int, day: int) -> float:
    """Add higher weight in October; returns multiplicative factor."""
    if 9 <= month <= 11:
        october_progress = min(max(day - 1, 0), 30) / 30
        return 1.0 + 0.5 * october_progress
    return 1.0


def build_features(sales_df: pd.DataFrame, holidays_df: pd.DataFrame, cutoff: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    # Ensure proper date types
    if not np.issubdtype(sales_df["date_livraison"].dtype, np.datetime64):
        print("\n🔧 Correction du format de date (MM/DD/YYYY)…")
        print(f"Format d'exemple avant correction: {sales_df['date_livraison'].iloc[0]}")
        sales_df["date_livraison"] = sales_df["date_livraison"].apply(parse_date)
        valid_dates = sales_df["date_livraison"].notna().sum()
        print(f"✅ {valid_dates}/{len(sales_df)} dates valides après correction")
        sales_df = sales_df[sales_df["date_livraison"].notna()].copy()
        print(f"➡️ Données après nettoyage: {len(sales_df)} entrées")

    print("\n✂️ Filtrage des données pour ne garder que celles avant juin 2025…")
    sales_df = sales_df[sales_df["date_livraison"] < cutoff].copy()
    print(f"➡️ Données après filtrage (avant {cutoff.date()}): {len(sales_df)} entrées")

    daily_sales = (
        sales_df.groupby("date_livraison")["quantite_livree"].sum().reset_index()
    )

    full_date_range = pd.date_range(
        start=daily_sales["date_livraison"].min(),
        end=daily_sales["date_livraison"].max(),
        freq="D",
    )
    daily_sales = (
        daily_sales.set_index("date_livraison").reindex(full_date_range).fillna(0).reset_index()
    )
    daily_sales.columns = ["date_livraison", "quantite_livree"]

    # Holidays cleanup
    holidays_df = holidays_df.copy()
    holidays_df["Date"] = pd.to_datetime(holidays_df["Date"], errors="coerce")
    holidays_df = holidays_df[holidays_df["Date"].notna()].copy()
    holidays_df = holidays_df[holidays_df["Date"] < cutoff].copy()

    holidays_df["is_holiday"] = 1
    holidays_df["is_ramadan"] = holidays_df["Nom"].str.contains("Ramadan", case=False, na=False).astype(int)
    holidays_df["ramadan_day"] = holidays_df["Nom"].apply(
        lambda x: int(re.search(r"Jour (\d+)", str(x)).group(1)) if pd.notna(x) and "Ramadan" in str(x) else 0
    )
    holidays_df["is_aid"] = holidays_df["Nom"].str.contains("Aïd|Eid", case=False, na=False).astype(int)

    merged_df = pd.merge(
        daily_sales,
        holidays_df[["Date", "is_holiday", "is_ramadan", "ramadan_day", "is_aid"]],
        left_on="date_livraison",
        right_on="Date",
        how="left",
    ).fillna(0).drop(columns=["Date"])

    ramadan_dates = holidays_df[holidays_df["is_ramadan"] == 1]["Date"].dt.date.unique()
    ramadan_dates = sorted(ramadan_dates)

    def days_to_ramadan(date: pd.Timestamp) -> int:
        if len(ramadan_dates) == 0:
            return 9999
        idx = bisect.bisect_left(ramadan_dates, date.date())
        if idx == 0:
            return (ramadan_dates[0] - date.date()).days
        if idx == len(ramadan_dates):
            return (date.date() - ramadan_dates[-1]).days
        return min(
            (ramadan_dates[idx] - date.date()).days,
            (date.date() - ramadan_dates[idx - 1]).days,
        )

    merged_df["days_to_ramadan"] = merged_df["date_livraison"].apply(days_to_ramadan)
    merged_df["month"] = merged_df["date_livraison"].dt.month
    merged_df["month_sin"] = np.sin(2 * np.pi * merged_df["month"] / 12)
    merged_df["month_cos"] = np.cos(2 * np.pi * merged_df["month"] / 12)
    merged_df["weekofyear"] = merged_df["date_livraison"].dt.isocalendar().week.astype(int)
    merged_df["week_sin"] = np.sin(2 * np.pi * merged_df["weekofyear"] / 52)
    merged_df["week_cos"] = np.cos(2 * np.pi * merged_df["weekofyear"] / 52)

    merged_df["post_holiday"] = 0
    for i in range(1, 4):
        merged_df.loc[merged_df["is_holiday"].shift(i) == 1, "post_holiday"] = i

    merged_df["trend"] = (merged_df["date_livraison"] - merged_df["date_livraison"].min()).dt.days
    merged_df["trend_exp"] = np.log1p(merged_df["trend"])
    merged_df["days_to_month_end"] = (
        merged_df["date_livraison"].dt.days_in_month - merged_df["date_livraison"].dt.day
    )
    merged_df["date_palm_weight"] = merged_df.apply(
        lambda x: date_palm_weight(x["date_livraison"].month, x["date_livraison"].day), axis=1
    )

    features = [
        "is_holiday",
        "is_ramadan",
        "ramadan_day",
        "is_aid",
        "month_sin",
        "month_cos",
        "week_sin",
        "week_cos",
        "days_to_ramadan",
        "date_palm_weight",
        "trend_exp",
        "days_to_month_end",
        "post_holiday",
    ]

    return merged_df, holidays_df, features


def build_future_df(merged_df: pd.DataFrame, holidays_df: pd.DataFrame, start_prediction: pd.Timestamp, forecast_horizon: int) -> pd.DataFrame:
    def _days_to_ramadan_factory():
        ramadan_dates = holidays_df[holidays_df["is_ramadan"] == 1]["Date"].dt.date.unique()
        ramadan_dates_sorted = sorted(ramadan_dates)

        def days_to_ramadan(date: pd.Timestamp) -> int:
            if len(ramadan_dates_sorted) == 0:
                return 9999
            idx = bisect.bisect_left(ramadan_dates_sorted, date.date())
            if idx == 0:
                return (ramadan_dates_sorted[0] - date.date()).days
            if idx == len(ramadan_dates_sorted):
                return (date.date() - ramadan_dates_sorted[-1]).days
            return min(
                (ramadan_dates_sorted[idx] - date.date()).days,
                (date.date() - ramadan_dates_sorted[idx - 1]).days,
            )

        return days_to_ramadan

    future_dates = pd.date_range(start=start_prediction, periods=forecast_horizon, freq="D")
    future_df = pd.DataFrame({"date_livraison": future_dates})

    future_df = pd.merge(
        future_df,
        holidays_df[["Date", "is_holiday", "is_ramadan", "ramadan_day", "is_aid"]],
        left_on="date_livraison",
        right_on="Date",
        how="left",
    ).fillna(0).drop(columns=["Date"])

    future_df["month"] = future_df["date_livraison"].dt.month
    future_df["month_sin"] = np.sin(2 * np.pi * future_df["month"] / 12)
    future_df["month_cos"] = np.cos(2 * np.pi * future_df["month"] / 12)
    future_df["weekofyear"] = future_df["date_livraison"].dt.isocalendar().week.astype(int)
    future_df["week_sin"] = np.sin(2 * np.pi * future_df["weekofyear"] / 52)
    future_df["week_cos"] = np.cos(2 * np.pi * future_df["weekofyear"] / 52)
    future_df["days_to_ramadan"] = future_df["date_livraison"].apply(_days_to_ramadan_factory())
    future_df["date_palm_weight"] = future_df.apply(
        lambda x: date_palm_weight(x["date_livraison"].month, x["date_livraison"].day), axis=1
    )
    future_df["trend"] = (
        future_df["date_livraison"] - merged_df["date_livraison"].min()
    ).dt.days
    future_df["trend_exp"] = np.log1p(future_df["trend"])
    future_df["days_to_month_end"] = (
        future_df["date_livraison"].dt.days_in_month - future_df["date_livraison"].dt.day
    )
    future_df["post_holiday"] = 0
    return future_df


def main():
    outputs_dir = get_outputs_dir()

    sales_df, holidays_df, source = load_data()

    cutoff_date = pd.Timestamp("2025-06-01")
    merged_df, holidays_df, features = build_features(sales_df, holidays_df, cutoff_date)

    last_date = merged_df["date_livraison"].max()
    print("\n📅 Configuration de la prédiction pour commencer en juin 2025…")
    start_prediction = pd.Timestamp("2025-06-01")
    if start_prediction <= last_date:
        start_prediction = last_date + timedelta(days=1)

    forecast_horizon = 150
    future_df = build_future_df(merged_df, holidays_df, start_prediction, forecast_horizon)
    print(f"➡️ Période de prédiction: {start_prediction.date()} à {future_df['date_livraison'].iloc[-1].date()}")

    # =========================
    # 4) MODELING
    # =========================
    X, y = merged_df[features], merged_df["quantite_livree"]

    # Fast mode for CI/smoke tests
    FAST = os.environ.get("FAST", "0") == "1"

    base_xgb_params = {
        "n_estimators": 500 if not FAST else 100,
        "learning_rate": 0.03,
        "max_depth": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "n_jobs": -1,
        # older xgboost will ignore unsupported params
    }
    final_xgb_base = {
        "n_estimators": 1000 if not FAST else 200,
        "learning_rate": 0.01,
        "max_depth": 12,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "min_child_weight": 3,
        "gamma": 0.1,
        "random_state": 42,
        "n_jobs": -1,
    }

    xgb_cv_params = get_xgb_params_for_device(base_xgb_params, device)
    xgb_final_params = get_xgb_params_for_device(final_xgb_base, device)

    from xgboost import XGBRegressor

    print(f"\n🚀 Démarrage de l'entraînement avec {device.upper()}…")
    print(f"Nombre total de données : {len(merged_df)}")
    print(f"Nombre de features : {len(features)}")

    tscv = TimeSeriesSplit(n_splits=5 if not FAST else 3)
    scores = []
    for i, (train_index, test_index) in enumerate(tscv.split(X)):
        print(f"\nFold {i+1}/{tscv.n_splits} - Entraînement…")
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        estimators = [
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=300 if not FAST else 100,
                    max_depth=20,
                    min_samples_leaf=3,
                    n_jobs=-1,
                    random_state=42,
                    bootstrap=True,
                ),
            ),
            ("xgb", XGBRegressor(**xgb_cv_params)),
        ]

        model = StackingRegressor(
            estimators=estimators, final_estimator=RidgeCV(alphas=[0.1, 1.0, 10.0])
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        scores.append(rmse)
        print(f"Fold {i+1} - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    mean_rmse = float(np.mean(scores)) if scores else float("nan")
    std_rmse = float(np.std(scores)) if scores else float("nan")
    print(f"\n✅ Validation croisée terminée - RMSE moyen: {mean_rmse:.2f} ± {std_rmse:.2f}")

    mean_sales = float(merged_df["quantite_livree"].mean())
    relative_rmse = (mean_rmse / mean_sales) * 100 if mean_sales else float("nan")
    print(f"📊 RMSE relatif: {relative_rmse:.1f}% (moyenne des ventes: {mean_sales:.2f})")
    if relative_rmse < 10:
        print("🟢 Excellent modèle (RMSE relatif < 10%)")
    elif relative_rmse < 15:
        print("🟡 Bon modèle (RMSE relatif < 15%)")
    elif relative_rmse < 20:
        print("🟠 Modèle acceptable (RMSE relatif < 20%)")
    else:
        print("🔴 Modèle à améliorer (RMSE relatif > 20%)")

    print("\n🚀 Entraînement du modèle final…")
    final_model = StackingRegressor(
        estimators=[
            (
                "rf",
                RandomForestRegressor(
                    n_estimators=500 if not FAST else 150,
                    max_depth=25,
                    min_samples_leaf=2,
                    max_samples=0.9,
                    n_jobs=-1,
                    random_state=42,
                    verbose=0,
                ),
            ),
            ("xgb", XGBRegressor(**xgb_final_params)),
        ],
        final_estimator=RidgeCV(alphas=np.logspace(-3, 3, 10)),
    )
    final_model.fit(X, y)
    print("✅ Modèle final entraîné avec succès")

    print("\n🔮 Génération des prévisions à 5 mois…")
    future_predictions = final_model.predict(future_df[features])

    # Post-processing
    future_df = future_df.copy()
    future_df["predicted_sales"] = future_predictions * future_df["date_palm_weight"]

    aid_dates = holidays_df[holidays_df["is_aid"] == 1]["Date"]
    for aid_date in aid_dates:
        if aid_date in future_df["date_livraison"].values:
            idx = future_df[future_df["date_livraison"] == aid_date].index[0]
            future_df.loc[idx : idx + 2, "predicted_sales"] *= 1.3

    # =========================
    # 5) VISUALISATIONS & EXPORTS
    # =========================
    print("\n📊 Génération des visualisations…")
    plt.figure(figsize=(20, 10))
    sns.set_style("whitegrid")
    plt.plot(
        merged_df["date_livraison"],
        merged_df["quantite_livree"],
        label="Ventes réelles (avant juin 2025)",
        color="blue",
        alpha=0.7,
        linewidth=2,
    )
    plt.plot(
        future_df["date_livraison"],
        future_df["predicted_sales"],
        label="Prévision 5 mois (à partir de juin 2025)",
        color="red",
        linewidth=3,
    )
    plt.fill_between(
        future_df["date_livraison"],
        future_df["predicted_sales"] * 0.85,
        future_df["predicted_sales"] * 1.15,
        color="red",
        alpha=0.2,
    )
    start_prediction_line = future_df["date_livraison"].iloc[0]
    plt.axvline(x=start_prediction_line, color="green", linestyle="-", alpha=0.8, linewidth=2)
    ymax = plt.ylim()[1]
    plt.text(
        start_prediction_line,
        ymax * 0.8,
        "Début de la prédiction",
        rotation=90,
        fontsize=12,
        fontweight="bold",
        color="green",
    )
    plt.title(
        f"Prévision des ventes - Modèle Tunisie (5 mois à l'avance à partir de juin 2025)\nRMSE relatif: {relative_rmse:.1f}%",
        fontsize=18,
        fontweight="bold",
    )
    plt.xlabel("Date", fontsize=14)
    plt.ylabel("Quantité vendue", fontsize=14)
    plt.legend(fontsize=12, loc="best")
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    fig_path = os.path.join(outputs_dir, "prevision_ventes.png")
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

    # Feature importance from XGB in final model
    print("\n🔍 Analyse de l'importance des caractéristiques…")
    xgb_model = final_model.named_estimators_["xgb"]
    feature_importance = pd.DataFrame({
        "Feature": features,
        "Importance": getattr(xgb_model, "feature_importances_", np.zeros(len(features))),
    }).sort_values("Importance", ascending=False)

    plt.figure(figsize=(14, 8))
    bars = plt.barh(feature_importance["Feature"], feature_importance["Importance"], color="skyblue")
    plt.xlabel("Importance", fontsize=12)
    plt.title("Importance des caractéristiques - XGBoost", fontsize=16)
    plt.gca().invert_yaxis()
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height() / 2, f"{width:.4f}", ha="left", va="center", fontsize=10)
    plt.tight_layout()
    fi_path = os.path.join(outputs_dir, "importance_features.png")
    plt.savefig(fi_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n💾 Export des résultats…")
    forecast_results = pd.DataFrame(
        {
            "Date": future_df["date_livraison"],
            "Ventes_prévues": future_df["predicted_sales"].round(2),
            "Période_dattes": future_df["date_palm_weight"] > 1,
            "Effet_Ramadan": future_df["days_to_ramadan"].apply(lambda x: "Proche" if x < 15 else "Éloigné"),
            "Poids_dattes": future_df["date_palm_weight"],
        }
    )
    excel_path = os.path.join(outputs_dir, "prevision_5mois.xlsx")
    csv_path = os.path.join(outputs_dir, "prevision_5mois.csv")
    forecast_results.to_excel(excel_path, index=False)
    forecast_results.to_csv(csv_path, index=False)

    print("\n✅ Export terminé avec succès !")
    print(f"Prévisions générées pour {len(future_df)} jours (5 mois)")
    print(f"Fichiers disponibles : {excel_path} | {csv_path}")
    print("\nAperçu des prévisions :")
    print(forecast_results.head(10))

    # =========================
    # 6) VALIDATION (Holdout Mar-Jun 2025 if available)
    # =========================
    print("\n🧪 Validation de la prédiction (simulation)")
    validation_start = pd.Timestamp("2025-03-01")
    validation_mask = merged_df["date_livraison"] >= validation_start
    if validation_mask.sum() > 0:
        X_val = merged_df.loc[validation_mask, features]
        y_val = merged_df.loc[validation_mask, "quantite_livree"]
        y_pred_val = final_model.predict(X_val)
        mae_val = mean_absolute_error(y_val, y_pred_val)
        rmse_val = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
        r2_val = r2_score(y_val, y_pred_val)
        relative_rmse_val = (rmse_val / y_val.mean()) * 100 if y_val.mean() else float("nan")
        print(f"- MAE (mars-juin 2025): {mae_val:.2f}")
        print(f"- RMSE (mars-juin 2025): {rmse_val:.2f}")
        print(f"- RMSE relatif (validation): {relative_rmse_val:.1f}%")
        print(f"- R² (validation): {r2_val:.3f}")

        plt.figure(figsize=(16, 8))
        plt.plot(merged_df.loc[validation_mask, "date_livraison"], y_val, label="Ventes réelles", color="blue", linewidth=2)
        plt.plot(merged_df.loc[validation_mask, "date_livraison"], y_pred_val, label="Prédictions", color="red", linestyle="--", linewidth=2)
        errors = np.abs(y_val.values - y_pred_val)
        plt.fill_between(
            merged_df.loc[validation_mask, "date_livraison"], y_val - errors, y_val + errors, color="gray", alpha=0.2, label="Erreur absolue"
        )
        plt.title(f"Validation du modèle (mars-juin 2025)\nRMSE relatif: {relative_rmse_val:.1f}%", fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Quantité vendue", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        val_fig_path = os.path.join(outputs_dir, "validation_model.png")
        plt.savefig(val_fig_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        print("⚠️ Pas assez de données pour la validation (mars-juin 2025)")

    # =========================
    # 7) SIMPLE BACKTEST BY HORIZON (optional quick variant in FAST)
    # =========================
    print("\n⏳ Backtesting à différents horizons de prévision")
    horizons = [30, 60, 90, 120, 150]
    if FAST:
        horizons = [30, 90]
    rmse_by_horizon = []
    relative_rmse_by_horizon = []

    from xgboost import XGBRegressor as _XGB
    backtest_model = StackingRegressor(
        estimators=[
            ("rf", RandomForestRegressor(n_estimators=300 if not FAST else 100, max_depth=15, random_state=42)),
            ("xgb", _XGB(**get_xgb_params_for_device({"n_estimators": 500 if not FAST else 150, "learning_rate": 0.03}, device))),
        ],
        final_estimator=RidgeCV(),
    )

    tscv_bt = TimeSeriesSplit(n_splits=5 if not FAST else 3)
    for horizon in horizons:
        print(f"\nBacktesting pour un horizon de {horizon} jours…")
        fold_errors = []
        fold_relative_errors = []
        for fold, (train_index, _test_index) in enumerate(tscv_bt.split(merged_df)):
            fold_data = merged_df.iloc[train_index].copy()
            if len(fold_data) < horizon:
                continue
            cutoff_bt = fold_data["date_livraison"].iloc[-horizon]
            train_fold = fold_data[fold_data["date_livraison"] < cutoff_bt]
            test_fold = fold_data[fold_data["date_livraison"] >= cutoff_bt].iloc[:horizon]
            if len(test_fold) == 0:
                continue
            X_train, y_train = train_fold[features], train_fold["quantite_livree"]
            backtest_model.fit(X_train, y_train)
            X_test, y_true = test_fold[features], test_fold["quantite_livree"]
            y_pred = backtest_model.predict(X_test)
            rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
            relative_rmse = (rmse / y_true.mean()) * 100 if y_true.mean() else float("nan")
            fold_errors.append(rmse)
            fold_relative_errors.append(relative_rmse)
            print(f"  Fold {fold+1} - RMSE: {rmse:.2f} (relatif: {relative_rmse:.1f}%)")

        if fold_errors:
            avg_rmse = float(np.mean(fold_errors))
            avg_relative_rmse = float(np.mean(fold_relative_errors))
            rmse_by_horizon.append(avg_rmse)
            relative_rmse_by_horizon.append(avg_relative_rmse)
            print(f"Horizon {horizon} jours - RMSE moyen: {avg_rmse:.2f} (relatif: {avg_relative_rmse:.1f}%)")
        else:
            print(f"Horizon {horizon} jours - Pas assez de données pour le backtesting")

    if rmse_by_horizon:
        plt.figure(figsize=(12, 6))
        plt.plot(horizons[: len(rmse_by_horizon)], rmse_by_horizon, "o-")
        plt.xlabel("Horizon de prévision (jours)")
        plt.ylabel("RMSE")
        plt.title("Dégradation de la précision avec l'horizon de prévision")
        plt.grid(True)
        bt_fig = os.path.join(outputs_dir, "backtesting_horizons.png")
        plt.savefig(bt_fig, dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure(figsize=(12, 6))
        plt.plot(horizons[: len(relative_rmse_by_horizon)], relative_rmse_by_horizon, "o-")
        plt.xlabel("Horizon de prévision (jours)")
        plt.ylabel("RMSE relatif (%)")
        plt.title("Dégradation du RMSE relatif avec l'horizon de prévision")
        plt.grid(True)
        bt_rel_fig = os.path.join(outputs_dir, "backtesting_horizons_relative.png")
        plt.savefig(bt_rel_fig, dpi=300, bbox_inches="tight")
        plt.close()

    # =========================
    # 8) COMPARAISON AVEC UN MODÈLE NAÏF (sur validation si dispo)
    # =========================
    print("\n📊 Comparaison avec un modèle naïf")
    if validation_mask.sum() > 0:
        # Naïf = dernière valeur observée avant validation start
        last_train_value = merged_df.loc[~validation_mask, "quantite_livree"].iloc[-1] if (~validation_mask).any() else merged_df["quantite_livree"].iloc[-1]
        naive_predictions = np.full(shape=len(y_val), fill_value=last_train_value)
        naive_rmse = float(np.sqrt(mean_squared_error(y_val, naive_predictions)))
        model_rmse = float(np.sqrt(mean_squared_error(y_val, y_pred_val)))
        naive_relative_rmse = (naive_rmse / y_val.mean()) * 100 if y_val.mean() else float("nan")
        model_relative_rmse = (model_rmse / y_val.mean()) * 100 if y_val.mean() else float("nan")
        improvement = ((naive_rmse - model_rmse) / naive_rmse) * 100 if naive_rmse else 0.0
        print(f"- RMSE modèle naïf: {naive_rmse:.2f} (relatif: {naive_relative_rmse:.1f}%)")
        print(f"- RMSE votre modèle: {model_rmse:.2f} (relatif: {model_relative_rmse:.1f}%)")
        print(f"- Amélioration: {improvement:.1f}%")

        plt.figure(figsize=(16, 8))
        plt.plot(merged_df.loc[validation_mask, "date_livraison"], y_val, label="Ventes réelles", color="gray", alpha=0.5)
        plt.plot(merged_df.loc[validation_mask, "date_livraison"], y_pred_val, label="Votre modèle", color="blue", linewidth=2)
        plt.plot(merged_df.loc[validation_mask, "date_livraison"], naive_predictions, label="Modèle naïf (dernière valeur)", color="red", linestyle="--", linewidth=2)
        plt.title("Comparaison avec un modèle naïf (validation)", fontsize=16)
        plt.xlabel("Date", fontsize=12)
        plt.ylabel("Quantité vendue", fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        naive_fig = os.path.join(outputs_dir, "comparison_naive_model.png")
        plt.savefig(naive_fig, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        print("⚠️ Comparaison naïve sautée (pas de validation disponible)")

    # =========================
    # 9) ANALYSE & RECO
    # =========================
    print("\n🎯 Insights stratégiques :")
    peak_month = forecast_results.loc[forecast_results["Ventes_prévues"].idxmax(), "Date"].strftime("%B %Y")
    print(f"- Pic de ventes prévu en : {peak_month}")

    avg_sales = forecast_results["Ventes_prévues"].mean()
    ramadan_sales_mean = forecast_results[forecast_results["Effet_Ramadan"] == "Proche"]["Ventes_prévues"].mean()
    if not np.isnan(ramadan_sales_mean) and avg_sales:
        print(f"- Impact Ramadan : +{((ramadan_sales_mean / avg_sales) - 1) * 100:.1f}% de ventes supplémentaires")

    palm_mask = (future_df["date_livraison"] >= pd.Timestamp("2025-10-01")) & (future_df["date_livraison"] <= pd.Timestamp("2025-12-01"))
    if palm_mask.sum() > 0:
        date_sales = future_df.loc[palm_mask, "predicted_sales"].mean()
        if avg_sales:
            print(f"- Impact période des dattes : +{((date_sales / avg_sales) - 1) * 100:.1f}% de ventes supplémentaires")

    # Basic overall score summary
    print("\n==================================================")
    print("ANALYSE FINALE DE LA QUALITÉ DES PRÉDICTIONS")
    print("==================================================")
    rel_val = locals().get("relative_rmse_val", None)
    print(f"- RMSE relatif (CV): {relative_rmse:.1f}%")
    if rel_val is not None:
        print(f"- RMSE relatif (validation): {rel_val:.1f}%")
    print("==================================================")


if __name__ == "__main__":
    main()

