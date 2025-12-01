import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

model = joblib.load("random_forest_model.pkl")


feature_names = [
    'tvl_current', 'tvl_mean', 'tvl_volatility', 'max_drawdown',
    'apy_mean', 'apy_std', 'chain_score'
]


importances = model.feature_importances_
indices = np.argsort(importances)[::-1]


plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(len(importances)), importances[indices], align='center')
plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
plt.tight_layout()
plt.savefig("feature_importances.png")
plt.show()