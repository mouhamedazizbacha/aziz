## Prévision des ventes Tunisie (XGBoost + RF Stacking)

Ce projet implémente un pipeline de prévision des ventes avec stacking (RandomForest + XGBoost) et des features spécifiques à la Tunisie (Ramadan, Aïd, période des dattes). Le script détecte automatiquement le GPU disponible (Kaggle/ CUDA) et bascule en CPU si nécessaire.

### Fichiers importants
- `forecast_tunisia.py`: script principal
- `requirements.txt`: dépendances Python
- Dossier de sortie: `/workspace/outputs` (local) ou `/kaggle/working` (Kaggle)

### Données attendues
Placez les fichiers dans l'un des emplacements suivants:
- Kaggle: `/kaggle/input/predection-vente/`
  - `fact_livraisons_vente.xlsx`
  - `jours_feries_tunisie_2020_2026.xlsx`
- Local: `/workspace/data/`
  - `fact_livraisons_vente.xlsx`
  - `jours_feries_tunisie_2020_2026.xlsx`

Si les fichiers sont absents, le script génère des données synthétiques pour un test rapide (smoke test).

### Installation
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r /workspace/requirements.txt
```

### Exécution
```bash
python /workspace/forecast_tunisia.py
```

Mode rapide (réduit le temps d'entraînement pour CI/tests):
```bash
FAST=1 python /workspace/forecast_tunisia.py
```

### Résultats
Les exports et figures sont écrits dans:
- Local: `/workspace/outputs`
- Kaggle: `/kaggle/working`

Fichiers générés (exemples):
- `prevision_5mois.xlsx`, `prevision_5mois.csv`
- `prevision_ventes.png`, `importance_features.png`
- `validation_model.png` (si validation disponible)
- `backtesting_horizons.png`, `backtesting_horizons_relative.png` (si backtest)
- `comparison_naive_model.png` (si comparaison naïve disponible)

### Notes
- Le script gère les versions d'XGBoost: pour >= 2.0, paramètre `device`; pour 1.x, `tree_method=gpu_hist`.
- Les features incluent saisons cycliques (Fourier), proximité du Ramadan, effet post-jours fériés, tendance log, et pondération période des dattes.
