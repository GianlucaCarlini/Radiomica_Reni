# %% Import

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from deap import base, creator, tools
from Radiomics.Genetic import eaSimpleWithElitism
from sklearn.model_selection import cross_val_score, GridSearchCV, cross_validate
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import shuffle
from dtreeviz.trees import dtreeviz

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

# %% Train test split

X_train, X_test, Y_train, Y_test = train_test_split(
    radiomics_arteriosa,
    classes_arteriosa,
    test_size=0.2,
    random_state=42,
    stratify=classes_arteriosa,
)

# %% Model

dtc = DecisionTreeClassifier(max_depth=4, min_samples_split=10, random_state=42)

# %% Genetic Constants

POPULATION_SIZE = 150
P_CROSSOVER = 0.9
P_MUTATION = 0.1
MAX_GENERATIONS = 150

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
    X_train.shape[1],
)
toolbox.register("PopulationCreator", tools.initRepeat, list, toolbox.IndividualCreator)

# %% Genetic Operators

toolbox.register("select", tools.selTournament, tournsize=2)
toolbox.register("mate", tools.cxUniform, indpb=0.5)
toolbox.register("mutate", tools.mutFlipBit, indpb=1.0 / (X_train.shape[1]))


def Evaluate(individual):

    fitness = Fitness(
        X=X_train, Y=Y_train, zero_one_list=individual, model=dtc, penalty=0.005,
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

selected_train = X_train.iloc[:, selected]
selected_test = X_test.iloc[:, selected]

# %% score
dtc = DecisionTreeClassifier(max_depth=4, min_samples_split=10, random_state=42)
dtc.fit(selected_train, Y_train)
proba = dtc.predict_proba(selected_test)

roc = roc_auc_score(Y_test, proba[:, 1])

# %% score

scv = StratifiedKFold(n_splits=10)
score = cross_val_score(dtc, X=selected_train, y=Y_train, cv=scv, scoring="roc_auc")

metrics_10 = cross_validate(
    dtc,
    X=selected_train,
    y=Y_train,
    cv=scv,
    scoring=["roc_auc", "precision", "recall", "f1"],
)

metrics_10_df = pd.DataFrame.from_dict(metrics_10, orient="columns")

scv = StratifiedKFold(n_splits=5)
score = cross_val_score(dtc, X=selected_train, y=Y_train, cv=scv, scoring="roc_auc")

metrics_5 = cross_validate(
    dtc,
    X=selected_train,
    y=Y_train,
    cv=scv,
    scoring=["roc_auc", "precision", "recall", "f1"],
)

metrics_5_df = pd.DataFrame.from_dict(metrics_5, orient="columns")

# %% Plot tree

dtc = DecisionTreeClassifier(max_depth=3, min_samples_split=10, random_state=42)

dtc.fit(selected_train, Y_train)

GREEN = "#27AE60"
YELLOW = "#F1C40F"

viz = dtreeviz(
    dtc,
    selected_train,
    Y_train,
    feature_names=selected_train.columns.to_list(),
    target_name="Histotype",
    histtype="barstacked",
    colors={"classes": [None, None, [GREEN, YELLOW]]},
    class_names=["0", "1"],
)
viz.save("decision_tree_test.svg")

# %% Save

selected_train["ID"] = ID_venosa.to_list()
selected_train["Class"] = Y_train.to_list()
selected_train.to_csv("selected_radiomic_venosa.csv", index=False)
metrics_5_df.to_csv("metrics_dt_5fold_venosa.csv", index=False)
metrics_10_df.to_csv("metrics_dt_10fold_venosa.csv", index=False)

# %%
