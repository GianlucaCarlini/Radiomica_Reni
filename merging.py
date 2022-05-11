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
from deap import base, creator, tools
from dtreeviz.trees import dtreeviz
from Radiomics.Genetic import eaSimpleWithElitism

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

# %% merging

radiomics_venosa = all[all["dtype"] == "Venosa"]
radiomics_arteriosa = all[all["dtype"] == "Arteriosa"]

radiomics_arteriosa = radiomics_arteriosa[
    radiomics_arteriosa["patient_id"].isin(radiomics_venosa["patient_id"])
]

radiomics_venosa = radiomics_venosa.sort_values(by="patient_id")
radiomics_arteriosa = radiomics_arteriosa.sort_values(by="patient_id")

radiomics_venosa.reset_index(inplace=True, drop=True)
radiomics_arteriosa.reset_index(inplace=True, drop=True)

radiomics_venosa = radiomics_venosa.add_prefix("venosa_")
radiomics_arteriosa = radiomics_arteriosa.add_prefix("arteriosa_")

radiomics_tot = radiomics_venosa.join(radiomics_arteriosa)

# %%

classes = classes[classes["ID"].isin(radiomics_tot["arteriosa_patient_id"])]

radiomics_tot = radiomics_tot[radiomics_tot["arteriosa_patient_id"].isin(classes["ID"])]

radiomics_tot = radiomics_tot.sort_values(by="arteriosa_patient_id")
classes = classes.sort_values(by="ID")

# %% shuffling

radiomics_tot = shuffle(radiomics_tot, random_state=42)
classes = shuffle(classes, random_state=42)

ID = radiomics_tot["venosa_patient_id"]

# %% select numeric data only

radiomics_tot = radiomics_tot.select_dtypes(include="float64")
classes = classes["Histotype"]

cutoff_cols = []

for col in radiomics_tot.columns:
    if ("1-0-mm" in col) or ("2-0-mm" in col) or ("3-0-mm" in col):
        cutoff_cols.append(col)

radiomics_cut_off = radiomics_tot.drop(cutoff_cols, axis=1)

# %% Model

dtc = DecisionTreeClassifier(max_depth=4, min_samples_split=10, random_state=42)

# %% Genetic Constants

POPULATION_SIZE = 200
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATIONS = 250

# %% Fitness

scv = StratifiedKFold(n_splits=5)


def Fitness(X, Y, model, zero_one_list, penalty=0.0):

    if sum(zero_one_list) == 0:
        return 0.5
    else:
        one_indices = [i for i, n in enumerate(zero_one_list) if n == 1]
        X_selected = X.iloc[:, one_indices]

        score = cross_val_score(model, X_selected, Y, cv=scv, scoring="roc_auc")

        return np.mean(score) - (penalty * sum(zero_one_list))


# %% Genetic Tools

toolbox = base.Toolbox()

# we want to maximize the fitness
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
# class individual
creator.create("Individual", list, fitness=creator.FitnessMax)
toolbox.register("ZeroOne", np.random.choice, 2, p=(0.90, 0.10))
toolbox.register(
    "IndividualCreator",
    tools.initRepeat,
    creator.Individual,
    toolbox.ZeroOne,
    radiomics_tot.shape[1],
)
toolbox.register("PopulationCreator", tools.initRepeat, list, toolbox.IndividualCreator)

# %% Genetic Operators

toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / (radiomics_tot.shape[1]))


def Evaluate(individual):

    fitness = Fitness(
        X=radiomics_tot, Y=classes, zero_one_list=individual, model=dtc, penalty=0.0075,
    )
    return (fitness,)


toolbox.register("evaluate", Evaluate)

# %% Genetic Flow

stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("Max", np.max)
stats.register("Average", np.mean)

hof = tools.HallOfFame(maxsize=5)

population = toolbox.PopulationCreator(n=POPULATION_SIZE)

population, logbook = eaSimpleWithElitism(
    population,
    toolbox,
    cxpb=P_CROSSOVER,
    mutpb=P_MUTATION,
    ngen=MAX_GENERATIONS,
    stats=stats,
    halloffame=hof,
    verbose=True,
)

max_fitness, avg_fitness = logbook.select("Max", "Average")

# %% selected features

selected = [i for i, n in enumerate(hof[0]) if n == 1]

selected_rad = radiomics_tot.iloc[:, selected]

# %% score


scv = StratifiedKFold(n_splits=10)
score_10 = cross_val_score(dtc, X=selected_rad, y=classes, cv=scv, scoring="roc_auc")

metrics_10 = cross_validate(
    dtc,
    X=selected_rad,
    y=classes,
    cv=scv,
    scoring=["roc_auc", "precision", "recall", "f1"],
)

metrics_10_df = pd.DataFrame.from_dict(metrics_10, orient="columns")

scv = StratifiedKFold(n_splits=5)
score_5 = cross_val_score(dtc, X=selected_rad, y=classes, cv=scv, scoring="roc_auc")

metrics_5 = cross_validate(
    dtc,
    X=selected_rad,
    y=classes,
    cv=scv,
    scoring=["roc_auc", "precision", "recall", "f1"],
)

metrics_5_df = pd.DataFrame.from_dict(metrics_5, orient="columns")

# %% tree


dtc = DecisionTreeClassifier(max_depth=3, min_samples_split=10, random_state=42)

dtc.fit(selected_rad, classes)

GREEN = "#27AE60"
YELLOW = "#F1C40F"

viz = dtreeviz(
    dtc,
    selected_rad,
    classes,
    feature_names=selected_rad.columns.to_list(),
    target_name="Histotype",
    histtype="barstacked",
    colors={"classes": [None, None, [GREEN, YELLOW]]},
    class_names=["0", "1"],
)
viz.save("decision_tree_venosa-arteriosa.svg")

# %% save

selected_rad["ID"] = ID.to_list()
selected_rad["Class"] = classes.to_list()
selected_rad.to_csv("selected_radiomic_venosa-arteriosa.csv", index=False)
metrics_5_df.to_csv("metrics_dt_5fold_venosa-arteriosa.csv", index=False)
metrics_10_df.to_csv("metrics_dt_10fold_venosa-arteriosa.csv", index=False)

# %% GradientBoosting

"""
-----------------------
GRADIENT BOOSTING
-----------------------
"""

gbc = GradientBoostingClassifier()

params = {
    "max_depth": [2, 3, 4, 5],
    "n_estimators": [100, 200, 300],
    "min_samples_split": [8, 10, 12, 14, 16],
    "max_features": [0.05, 0.1, 0.2, 0.3],
    "random_state": [42],
}

# %% Grid search

scv = StratifiedKFold(n_splits=5)

gs = GridSearchCV(estimator=gbc, param_grid=params, scoring="roc_auc", cv=scv)

gs.fit(radiomics_cut_off, classes)

best = gs.best_params_

# %% Scoring

gbc = GradientBoostingClassifier(
    max_depth=3,
    min_samples_split=8,
    n_estimators=100,
    max_features=0.05,
    random_state=42,
)

scv = StratifiedKFold(n_splits=10)
score_10 = cross_val_score(
    gbc, X=radiomics_cut_off, y=classes, cv=scv, scoring="roc_auc"
)

metrics_10 = cross_validate(
    gbc,
    X=radiomics_cut_off,
    y=classes,
    cv=scv,
    scoring=["roc_auc", "precision", "recall", "f1"],
)

metrics_10_df = pd.DataFrame.from_dict(metrics_10, orient="columns")

scv = StratifiedKFold(n_splits=5)
score_5 = cross_val_score(
    gbc, X=radiomics_cut_off, y=classes, cv=scv, scoring="roc_auc"
)

metrics_5 = cross_validate(
    gbc,
    X=radiomics_cut_off,
    y=classes,
    cv=scv,
    scoring=["roc_auc", "precision", "recall", "f1"],
)

metrics_5_df = pd.DataFrame.from_dict(metrics_5, orient="columns")

# %% save

metrics_10_df.to_csv("metrics_gbc_10fold_venosa-arteriosa_cutoff.csv", index=False)
metrics_5_df.to_csv("metrics_gbc_5fold_venosa-arteriosa_cutoff.csv", index=False)

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
    "random_state": [42],
}

# %% Grid search

scv = StratifiedKFold(n_splits=5)

gs = GridSearchCV(estimator=rfc, param_grid=params, scoring="roc_auc", cv=scv)

gs.fit(radiomics_cut_off, classes)

best = gs.best_params_

# %% Scoring

rfc = RandomForestClassifier(
    max_depth=4,
    min_samples_split=8,
    n_estimators=100,
    max_features=0.2,
    random_state=42,
)

scv = StratifiedKFold(n_splits=10)
score_10 = cross_val_score(
    rfc, X=radiomics_cut_off, y=classes, cv=scv, scoring="roc_auc"
)

metrics_10 = cross_validate(
    rfc,
    X=radiomics_cut_off,
    y=classes,
    cv=scv,
    scoring=["roc_auc", "precision", "recall", "f1"],
)

metrics_10_df = pd.DataFrame.from_dict(metrics_10, orient="columns")

scv = StratifiedKFold(n_splits=5)
score_5 = cross_val_score(
    rfc, X=radiomics_cut_off, y=classes, cv=scv, scoring="roc_auc"
)

metrics_5 = cross_validate(
    rfc,
    X=radiomics_cut_off,
    y=classes,
    cv=scv,
    scoring=["roc_auc", "precision", "recall", "f1"],
)

metrics_5_df = pd.DataFrame.from_dict(metrics_5, orient="columns")

# %% save

metrics_10_df.to_csv("metrics_rfc_10fold_venosa-arteriosa_cutoff.csv", index=False)
metrics_5_df.to_csv("metrics_rfc_5fold_venosa-arteriosa_cutoff.csv", index=False)


# %%
