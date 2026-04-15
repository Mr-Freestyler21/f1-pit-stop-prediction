# F1 Pit Stop Prediction — EDA & Machine Learning

An end-to-end data science project analysing tyre strategy across the 2023 
and 2024 F1 seasons and building a machine learning model to predict whether 
a driver will pit on the next lap.

The headline finding is methodological: a naive random split produced 95% 
accuracy; a season-aware split that eliminates data leakage produced 62%. 
The gap between those two numbers is the project.

---

## Dataset

**Source:** Kaggle — F1 Strategy Dataset v2  
**Size:** 52,471 laps · 16 columns · 25 drivers · 25 races · 2023 & 2024 seasons

---

## Key Findings

- `LapTime_Delta` is the strongest predictor of an upcoming pit stop — stronger than tyre age alone
- Hard tyres account for ~50% of all laps across both seasons
- Austria recorded the most pit stops (~800); Monaco the fewest (~110)
- A random 80/20 split inflated accuracy to 95% — laps from the same races 
  appeared in both train and test, giving the model an unfair advantage
- Training on 2023 and testing on 2024 removes leakage entirely and gives 
  an honest accuracy of 62%
- `PitNextLap` is 3.1% positive in 2023 (training) and 33.5% in 2024 (testing) 
  — a label distribution shift in the Kaggle dataset that limits cross-season 
  generalisation regardless of algorithm or features

---

## Model Progression

| Stage | Accuracy | Change |
|---|---|---|
| Random Forest — Random Split | 95% | Baseline, data leakage present |
| Random Forest — Season-Aware | 62% | Leakage removed |
| Random Forest — Tuned | 63% | RandomizedSearchCV, class_weight='balanced' |
| Tuned RF + Engineered Features | 66% | Degradation rate, laps remaining, rolling avg |
| XGBoost + Engineered Features | 65% | Sequential boosting |

---

## Installation

```bash
git clone https://github.com/yourusername/f1-pit-stop-prediction.git
cd f1-pit-stop-prediction

python -m venv .venv
.venv\Scripts\activate        # Windows
source .venv/bin/activate     # Mac/Linux

pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

Open `f1.ipynb` in VS Code or Jupyter and run all cells.  
The dataset `f1_strategy_dataset_v2.csv` must be in the same directory as the notebook.

---

## Tools
Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · XGBoost · Jupyter