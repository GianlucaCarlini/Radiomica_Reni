# %% import

import pandas as pd
import numpy as np
import os

# %% wrapping up

data = os.listdir("DATA")
metrics = [file for file in data if ("metrics" in file) and (not ("cutoff" in file))]

names = []
roc_aucs = []
precisions = []
recalls = []
f1s = []

for file in metrics:

    temp = pd.read_csv(os.path.join("./Data", file))
    names.append(file[8:-4])
    roc_aucs.append(
        f"{round(temp.iloc[:, 2].mean(), 3)} \u00B1 {round(temp.iloc[:, 2].std(), 3)}"
    )
    precisions.append(
        f"{round(temp.iloc[:, 3].mean(), 3)} \u00B1 {round(temp.iloc[:, 3].std(), 3)}"
    )
    recalls.append(
        f"{round(temp.iloc[:, 4].mean(), 3)} \u00B1 {round(temp.iloc[:, 4].std(), 3)}"
    )
    f1s.append(
        f"{round(temp.iloc[:, 5].mean(), 3)} \u00B1 {round(temp.iloc[:, 5].std(), 3)}"
    )

wrap = {
    "Name": names,
    "roc auc (avg \u00B1 std)": roc_aucs,
    "precision (avg \u00B1 std)": precisions,
    "recall (avg \u00B1 std)": recalls,
    "f1s (avg \u00B1 std)": f1s,
}

# %% saving

dataframe = pd.DataFrame.from_dict(wrap)

dataframe.to_csv("statistics_summary_no_cutoff.csv", index=False)
# %%
