# %% Import

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.utils import shuffle

# %% Load data

radiomics = pd.read_csv("radiomics.csv")
dilated_radiomics = pd.read_csv("dilated_radiomics.csv")

dilated_radiomics = dilated_radiomics.add_suffix("_dilated")

all = radiomics.join(dilated_radiomics)

classes = pd.read_csv("classes.csv", sep=";")

"""
Let's get rid of the duplicates in the classes df. They all have the same value
"""

classes = classes.drop_duplicates(subset="ID")

# %% select dtype

radiomics_venosa = all[all["dtype"] == "Venosa"]
radiomics_arteriosa = all[all["dtype"] == "Arteriosa"]

classes_venosa = classes[classes["ID"].isin(radiomics_venosa["patient_id"])]
classes_arteriosa = classes[classes["ID"].isin(radiomics_arteriosa["patient_id"])]

radiomics_venosa = radiomics_venosa[radiomics_venosa["patient_id"].isin(classes["ID"])]
radiomics_arteriosa = radiomics_arteriosa[
    radiomics_arteriosa["patient_id"].isin(classes["ID"])
]

# %% Sorting

radiomics_venosa = radiomics_venosa.sort_values(by="patient_id")
radiomics_arteriosa = radiomics_arteriosa.sort_values(by="patient_id")
classes_venosa = classes_venosa.sort_values(by="ID")
classes_arteriosa = classes_arteriosa.sort_values(by="ID")

# %% shuffling data

radiomics_venosa = shuffle(radiomics_venosa, random_state=42)
radiomics_arteriosa = shuffle(radiomics_arteriosa, random_state=42)
classes_venosa = shuffle(classes_venosa, random_state=42)
classes_arteriosa = shuffle(classes_arteriosa, random_state=42)

# %% Get ID columns

ID_venosa = radiomics_venosa["patient_id"]
ID_arteriosa = radiomics_arteriosa["patient_id"]

# %% Select numeric values only

radiomics_venosa = radiomics_venosa.select_dtypes(include="float64")
radiomics_arteriosa = radiomics_arteriosa.select_dtypes(include="float64")
classes_arteriosa = classes_arteriosa["Histotype"]
classes_venosa = classes_venosa["Histotype"]

# %% GradientBoosting

"""
-----------------------
GRADIENT BOOSTING
-----------------------
"""

gbc = GradientBoostingClassifier(random_state=42)

params = {
    "max_depth": [2, 3, 4, 5],
    "n_estimators": [100, 200, 300],
    "min_samples_split": [8, 10, 12, 14, 16],
    "max_features": [0.05, 0.1, 0.2, 0.3],
}

# %% Grid search

scv = StratifiedKFold(n_splits=5)

gs = GridSearchCV(estimator=gbc, param_grid=params, scoring="roc_auc", cv=scv)

gs.fit(radiomics_venosa, classes_venosa)

best = gs.best_params_

# %% Scoring

gbc = GradientBoostingClassifier(
    max_depth=4,
    min_samples_split=14,
    n_estimators=100,
    max_features=0.2,
    random_state=42,
)

scv = StratifiedKFold(n_splits=10)
score_10 = cross_val_score(
    gbc, X=radiomics_venosa, y=classes_venosa, cv=scv, scoring="roc_auc"
)

metrics_10 = cross_validate(
    gbc,
    X=radiomics_venosa,
    y=classes_venosa,
    cv=scv,
    scoring=["roc_auc", "precision", "recall", "f1"],
)

metrics_10_df = pd.DataFrame.from_dict(metrics_10, orient="columns")

scv = StratifiedKFold(n_splits=5)
score_5 = cross_val_score(
    gbc, X=radiomics_venosa, y=classes_venosa, cv=scv, scoring="roc_auc"
)

metrics_5 = cross_validate(
    gbc,
    X=radiomics_venosa,
    y=classes_venosa,
    cv=scv,
    scoring=["roc_auc", "precision", "recall", "f1"],
)

metrics_5_df = pd.DataFrame.from_dict(metrics_5, orient="columns")

# %% save

metrics_10_df.to_csv("metrics_gbc_10fold_venosa.csv", index=False)
metrics_5_df.to_csv("metrics_gbc_5fold_venosa.csv", index=False)

# %% Random Forest

"""
-----------------------
RANDOM FOREST
-----------------------
"""

rfc = RandomForestClassifier(random_state=42)

params = {
    "max_depth": [2, 3, 4, 5],
    "n_estimators": [100, 200, 300],
    "min_samples_split": [8, 10, 12, 14, 16],
    "max_features": [0.05, 0.1, 0.2, 0.3],
}

# %% Grid search

scv = StratifiedKFold(n_splits=5)

gs = GridSearchCV(estimator=rfc, param_grid=params, scoring="roc_auc", cv=scv)

gs.fit(radiomics_arteriosa, classes_arteriosa)

best = gs.best_params_

# %% Scoring

rfc = RandomForestClassifier(
    max_depth=3,
    min_samples_split=10,
    n_estimators=100,
    max_features=0.2,
    random_state=42,
)

scv = StratifiedKFold(n_splits=10)
score_10 = cross_val_score(
    rfc, X=radiomics_arteriosa, y=classes_arteriosa, cv=scv, scoring="roc_auc"
)

metrics_10 = cross_validate(
    rfc,
    X=radiomics_arteriosa,
    y=classes_arteriosa,
    cv=scv,
    scoring=["roc_auc", "precision", "recall", "f1"],
)

metrics_10_df = pd.DataFrame.from_dict(metrics_10, orient="columns")

scv = StratifiedKFold(n_splits=5)
score_5 = cross_val_score(
    rfc, X=radiomics_arteriosa, y=classes_arteriosa, cv=scv, scoring="roc_auc"
)

metrics_5 = cross_validate(
    rfc,
    X=radiomics_arteriosa,
    y=classes_arteriosa,
    cv=scv,
    scoring=["roc_auc", "precision", "recall", "f1"],
)

metrics_5_df = pd.DataFrame.from_dict(metrics_5, orient="columns")

# %% save

metrics_10_df.to_csv("metrics_rfc_10fold_arteriosa.csv", index=False)
metrics_5_df.to_csv("metrics_rfc_5fold_arteriosa.csv", index=False)

# %%
