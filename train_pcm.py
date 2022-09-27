import warnings
import sys
import logging
from collections import defaultdict
import yaml
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from common.utils import save_variable, find_pcm


def warn(*args, **kwargs):
    pass


warnings.warn = warn

# load config file from CLI
with open(str(sys.argv[1]), "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

# extract parameters from config file
dataset_name = config["dataset_name"]
iterations = config["iterations"]
dataset_str = config["dataset"]
seed = config["seed"]
target_column = config["target_column"]
target_values = config["target_values"]
categorical_features = config["categorical_features"]

# modeling
model = config["model"]
scorer = config["scorer"]
use_val = config["use_val"]

# logging file
logging.basicConfig(
    filename=f"{dataset_name}_pcm.log",
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
)

# initiliase data and modeling object
print("Initialising data and model objects ...")
logging.info("Initialising data and model objects ...")

# load dataset
df_full = pd.read_csv(dataset_str)
# making sure the label is the last column in the un-split dataframe
df_full_y = df_full.pop(target_column)
df_full.loc[:, target_column] = df_full_y

# encode categorical features
dict_cat_encoder = {}
for col in categorical_features:
    dict_cat_encoder[col] = LabelEncoder().fit(df_full[col])

df_full_encoded = df_full.apply(
    lambda x: dict_cat_encoder[x.name].transform(x)
    if x.name in categorical_features
    else x
)

# calculate PCM for multiple iterations and save the metrics and hyperparam
metrics = ["f1_micro"]
dict_pcm_list = {}
dict_pcm_acc_list = {}

# empty dictionary of dictionaries
for metric in metrics:
    dict_pcm_list[metric] = defaultdict(list)
    dict_pcm_acc_list[metric] = defaultdict(list)

print(f"PCM training for {dataset_name} using {model} and dataset {dataset_str}")
logging.info(f"PCM training for {dataset_name} using {model} and dataset {dataset_str}")

for i in range(0, iterations):
    print(f"Calculating iteration {i} ...")
    logging.info(f"Calculating iteration {i} ...")

    # for PCM use all the dataset without constant features
    dict_pcm, dict_pcm_acc = find_pcm(
        dataframe=df_full_encoded,
        model=model,
        scorer=scorer,
        use_val=use_val,
        folder_str=f"iter_{i}",
        verbose=True,
    )

    # each available scores
    for metric in metrics:
        # append values of each dictionary within the metric
        for user, _ in dict_pcm.items():
            dict_pcm_acc_list[metric][user].append(
                dict_pcm_acc[metric][user]
            )  # CV expected performance
            dict_pcm_list[metric][user].append(dict_pcm[user])  # model

# average of performance for each user
for metric in metrics:
    for user, acc_list in dict_pcm_acc_list[metric].items():
        # finding the model with the best score
        best_model_idx = dict_pcm_acc_list[metric][user].index(
            max(dict_pcm_acc_list[metric][user])
        )
        best_model = dict_pcm_list[metric][user][best_model_idx]
        dict_pcm_list[metric][user] = best_model

        # average metric for each user
        dict_pcm_acc_list[metric][user] = sum(acc_list) / len(acc_list)

    # Save variables
    print(f"Saving variables in data/{dataset_name}")
    logging.info(f"Saving variables in data/{dataset_name}")

    # e.g.`dict_pcm_{model}_{dataset_type}_{dataset_name}_{scorer}.pkl`
    save_variable(
        f"data/{dataset_name}/dict_pcm_{model}_{dataset_name}_{metric}.pkl",
        dict_pcm_list[metric],
    )
    save_variable(
        f"data/{dataset_name}/dict_pcm_acc_{model}_{dataset_name}_{metric}.pkl",
        dict_pcm_acc_list[metric],
    )
