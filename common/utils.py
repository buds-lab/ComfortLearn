import pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.colors import LinearSegmentedColormap
from pythermalcomfort.models import pmv
from pythermalcomfort.utilities import clo_dynamic, v_relative
from natsort import natsort_keygen

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.model_selection import KFold, StratifiedKFold


def save_variable(file_name, variable):
    pickle.dump(variable, open(file_name, "wb"))


def load_variable(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def clf_metrics(test_labels, pred_labels, conf_matrix_print=False, scorer="f1_micro"):
    """Compute the confusion matrix and a particular score based on `scorer`."""
    if scorer == "f1_micro":  # [0, 1]
        metric = f1_score(test_labels, pred_labels, average="micro", zero_division=0)

    # classification report
    class_report = classification_report(
        test_labels, pred_labels, output_dict=True, zero_division=0
    )

    if conf_matrix_print:
        print(f"Confusion Matrix: \n {confusion_matrix(test_labels, pred_labels)} \n")

    return metric, class_report


def choose_tree_depth(
    clf,
    X,
    y,
    k_fold,
    fig_name="",
    scorer="f1_micro",
    save_fig=False,
    verbose=False,
):
    """Choose the optimal depth of a tree model"""
    depths = list(range(1, 11))
    cv_scores = []

    if verbose:
        print("Finding optimal tree depth")

    for d in depths:
        # keep same params but depth
        clf_depth = clf.set_params(max_depth=d)

        if scorer == "f1_micro":
            scorer = "accuracy"  # accuracy = f1-micro

        scores = cross_val_score(clf_depth, X, y, cv=k_fold, scoring=scorer)
        cv_scores.append(scores.mean())

    # changing to misclassification error and determining best depth
    error = [1 - x for x in cv_scores]  # error = 1 - scorer
    optimal_depth = depths[error.index(min(error))]

    if save_fig:
        plt.figure(figsize=(12, 10))
        plt.plot(depths, error)
        plt.xlabel("Tree Depth", fontsize=40)
        plt.ylabel("Misclassification Error", fontsize=40)
        plt.savefig(f"{fig_name}_depth.png")
        plt.close()

    if verbose:
        print(
            f"The optimal depth is: {optimal_depth} with error of {min(error)} and score {max(cv_scores)}"
        )

    return optimal_depth, max(cv_scores)


def cv_model_param(X, y, model, parameters, k_fold, scorer="f1_micro", verbose=False):
    """Choose the best combination of parameters for a given model"""

    grid_search = GridSearchCV(model, parameters, cv=k_fold, scoring=scorer)
    grid_search.fit(X, y)

    if verbose:
        print(
            f"Best parameters set found on CV set: {grid_search.best_params_} with score of {grid_search.best_score_:.2f}"
        )

    return grid_search.best_estimator_, grid_search.best_score_


def train_model(
    dataframe,
    stratified=False,
    model="rdf",
    scorer="f1_micro",
    use_val=False,
    fig_name="",
):
    """
    Finds best set of param with K-fold CV and returns trained model and accuracy
    Assumes the label is the last column.

    Returns
    -------
        clf_cv: object
            Best performing lassification model from CV
        model_acc: dictionary
            Dictionary where the keys are the scorers used and values is the metric itself
        class_report:
            Dictionary where the keys are the scorers used and values is the classifiation report
    """
    model_acc = {}  # TODO: can be extended to more metrics
    model_acc["f1_micro"] = {}
    class_report = {}
    class_report["f1_micro"] = {}

    # create feature matrix X and target vector y
    X = np.array(
        dataframe.iloc[:, 0 : dataframe.shape[1] - 1]
    )  # minus 1 for the target column
    y = np.array(dataframe.iloc[:, -1]).astype(
        int
    )  # casting in case the original variable was a float

    if model == "rdf":
        parameters = {
            "n_estimators": [100, 300, 500],
            "criterion": ["gini"],
            "min_samples_split": [2, 3, 4],
            "min_samples_leaf": [1, 2, 3],
            "class_weight": ["balanced"],
        }
        clf = RandomForestClassifier(
            random_state=100, warm_start=False
        )  # warm_start=true allows for partial_fit

    # cross-validation
    kf = (
        StratifiedKFold(n_splits=5, shuffle=True)
        if stratified
        else KFold(n_splits=5, shuffle=True)
    )

    if use_val:
        dev_size_percentage = 0.2
        X_cv, X_dev, y_cv, y_dev = train_test_split(
            X, y, test_size=dev_size_percentage, random_state=100
        )  # , stratify=y)
        # find params with f1_micro
        clf_cv, cv_score_f1_micro = cv_model_param(
            X_cv, y_cv, clf, parameters, kf, scorer
        )
    else:
        # find params with f1_micro
        clf_cv, cv_score_f1_micro = cv_model_param(X, y, clf, parameters, kf, scorer)

    # plot depth for rdf and update model
    if model == "rdf":
        # find depth
        optimal_depth, cv_score_f1_micro = (
            choose_tree_depth(clf_cv, X_cv, y_cv, kf, fig_name, "f1_micro")
            if use_val
            else choose_tree_depth(clf_cv, X, y, kf, fig_name, "f1_micro")
        )
        clf_cv = clf_cv.set_params(max_depth=optimal_depth)

    # fit the model and get accuracy
    if use_val:
        clf_cv.fit(X_cv, y_cv)
        y_pred = clf_cv.predict(X_dev)
        model_acc["f1_micro"], class_report["f1_micro"] = clf_metrics(
            y_dev, y_pred, conf_matrix_print=False, scorer="f1_micro"
        )
    else:  # no dev_set (use_val=False) average cv_score will be the model_acc
        model_acc["f1_micro"] = cv_score_f1_micro
        class_report["f1_micro"] = ""

    return clf_cv, model_acc, class_report


def find_pcm(dataframe, model, scorer, use_val, folder_str, verbose=False):
    """
    Find the personal comfort model of each user based on CV.
    Assumes a column `user_id` exists.

    Parameters
    ----------
        dataframe: dataframe
            A DataFrame with all data and labels as last column
        model: str
            Name of the classification model to be used
        scorer: str
            Scoring metric for cross-validation performance
        use_val: boolean
            Whether to use a validation set
        folder_str: str
            Name for generated figures

    Returns
    -------
        user_pcm: dictionary
            Dictionary with the model (value) for each user (key)
        user_pcm_acc: dictionary
            Dictionary with the model accuracy (value) for each user (key)
    """

    df = dataframe.copy()

    user_list = df["user_id"].unique()
    if verbose:
        print(
            f"Features used for modeling (`user_id` and the last feature are not used): {df.columns.values}"
        )

    user_pcm = {}
    user_pcm_acc = {}
    user_pcm_acc["f1_micro"] = {}
    # TODO: other metrics can be added

    # for every user, do CV
    for user in user_list:
        df_user = df[df["user_id"] == user]
        df_user = df_user.drop(["user_id"], axis=1)

        fig_name = folder_str + str(user)
        model_user, model_user_acc, _ = train_model(
            dataframe=df_user,
            stratified=True,
            model=model,
            scorer=scorer,
            use_val=use_val,
            fig_name=fig_name,
        )
        user_pcm[user] = model_user
        user_pcm_acc["f1_micro"][user] = model_user_acc["f1_micro"]

    return user_pcm, user_pcm_acc


def simplified_pmv_model(data):
    data = data[["rh-env", "t-env", "clothing", "met", "thermal"]].copy()
    data["met"] = data["met"].map(
        {
            "Sitting": 1.1,
            "Resting": 0.8,
            "Standing": 1.4,
            "Exercising": 3,
        }
    )
    data["clothing"] = data["clothing"].map(
        {
            "Very light": 0.3,
            "Light": 0.5,
            "Medium": 0.7,
            "Heavy": 1,
        }
    )

    arr_pmv_grouped = []
    arr_pmv = []
    for _, row in data.iterrows():
        val = pmv(
            row["t-env"],
            row["t-env"],
            v_relative(0.1, row["met"]),
            row["rh-env"],
            row["met"],
            clo_dynamic(row["clothing"], row["met"]),
        )
        if val < -1.5:
            arr_pmv_grouped.append("Warmer")
        elif -1.5 <= val <= 1.5:
            arr_pmv_grouped.append("No Change")
        else:
            arr_pmv_grouped.append("Cooler")

        arr_pmv.append(val)

    data["PMV"] = arr_pmv
    data["PMV_grouped"] = arr_pmv_grouped

    return data["PMV_grouped"]


def plot_unc_heatmap_single(exp_name, band="", size=(16,6), fontsize=40):
    rc("text.latex", preamble=r"\usepackage{cmbright}")
    rc("text", usetex=True)

    band = f"_{band}" if band != "" else ""
    all_zones=[1,2]
    zone_idx = exp_name.index("zone")
    zone_num = exp_name[zone_idx+4:zone_idx+5]
    zone_num = all_zones if zone_num == "s" else [zone_num]

    zones, dict_zones_unc, dict_zones_unc_avg = {}, {}, {}

    for z in zone_num:
        zones[z] = load_variable(f"{exp_name}/Zone_{z}.pkl")
        dict_zones_unc[z] = load_variable(f"{exp_name}/dict_zones_unc{band}.pkl")[f"Zone_{z}"]
        dict_zones_unc_avg[z] = load_variable(f"{exp_name}/dict_zones_unc_avg{band}.pkl")[f"Zone_{z}"]

    cols = dict_zones_unc[z].keys()
    df = pd.DataFrame(columns=cols)

    # unc per occupant
    for z in zone_num:
        df_aux = pd.DataFrame(dict_zones_unc[z], index=[z])
        df = df.append(df_aux)
    
    fig, ax = plt.subplots(figsize=size)
    sns.heatmap(df, vmin=0, vmax=1, annot=True, ax=ax)

    plt.xlabel("Occupants", fontsize=fontsize)
    plt.ylabel("Zones", fontsize=fontsize)
    plt.savefig(f"img/heatmp_{exp_name}")


def plot_unc_heatmap_loop(tol, exp_name, band="", n_iter=30, size=(16,6), fontsize=40):
    df_iters = pd.DataFrame()
    exp_name_final = f"{exp_name}_iters"
    all_zones = [1,2]
    band = f"_{band}" if band != "" else ""
        
    # extract zone number
    zone_idx = exp_name.index("zone")
    zone_num = exp_name[zone_idx+4:zone_idx+5]
    zone_num = all_zones if zone_num == "s" else [zone_num]

    # average UNC
    for iter in range(1, n_iter+1):
        zones, dict_zones_unc, dict_zones_unc_avg = {}, {}, {}
        
        for z in zone_num:
            zones[z] = load_variable(f"{exp_name}/Zone_{z}_{iter}.pkl")
            dict_zones_unc[z] = load_variable(f"{exp_name}/dict_zones_unc{band}_{iter}.pkl")[f"Zone_{z}"]
            dict_zones_unc_avg[z] = load_variable(f"{exp_name}/dict_zones_unc_avg{band}_{iter}.pkl")[f"Zone_{z}"]

        cols = dict_zones_unc[z].keys()
        df = pd.DataFrame(columns=cols)

        # unc per occupant
        for z in zone_num:
            df_aux = pd.DataFrame(dict_zones_unc[z], index=[z])
            df = df.append(df_aux)
        
        # merge all iterations
        df_iters = pd.concat((df_iters, df))
    
    df_iters_mean = df_iters.groupby(df_iters.index).mean()
    df_iters_mean = df_iters_mean.sort_index(axis=1, key=natsort_keygen())
    df_iters_std = df_iters.groupby(df_iters.index).std()

    # actual plot
    plot_unc_heatmap(df_iters_mean, exp_name_final, size=size, fontsize=fontsize)


def plot_unc_heatmap(df, exp_name, size=(16,6), fontsize=40):
    rc("text.latex", preamble=r"\usepackage{cmbright}")
    rc("text", usetex=True)

    fig, ax = plt.subplots(figsize=size)
    sns.heatmap(df, vmin=0, vmax=1, annot=True, ax=ax, annot_kws={"fontsize":fontsize*0.75})

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fontsize*0.5)
    
    plt.xticks(rotation = 45) 
    plt.tick_params(labelsize=fontsize * 0.75)

    # custom for plotting
    if "zone1" in exp_name or "zone2" in exp_name:
        ax.set_xticklabels([])
        plt.xlabel("", size=fontsize)
    else:
        ax.set_xticklabels([f"occu_{i}" for i in range(1, 11)])
        plt.xlabel("Occupants", fontsize=fontsize)
    
    plt.ylabel("Zones", fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f"img/heatmp_{exp_name}")


def plot_unc_ts_single(exp_name, band="", all_zones=["1", "2"], size=(16, 32), fontsize=40):
    rc("text.latex", preamble=r"\usepackage{cmbright}")
    rc("text", usetex=True)

    band = f"_{band}" if band != "" else ""
    zone_idx = exp_name.index("zone")
    zone_num = exp_name[zone_idx+4:zone_idx+5]
    zone_num = all_zones if zone_num == "s" else [zone_num]

    dict_zones_unc_ts = {}
    for z in zone_num:
        dict_zones_unc_ts[z] = load_variable(f"{exp_name}/dict_zones_unc_ts{band}.pkl")[f"Zone_{z}"]

    num_occupants = len(dict_zones_unc_ts[zone_num[0]])

    fig, ax = plt.subplots(num_occupants, 1, figsize=size, sharex=True)

    for zone_id in zone_num:
        for i in range(0, num_occupants):
            ax[i].plot(dict_zones_unc_ts[zone_id][f"user_{i+1}"], label=f"Zone {zone_id}")
            for j in range(1, 5):
                ax[i].vlines(96*j, 0, 1, linestyles="dashed", colors="k")

            ax[i].set_ylabel(f"occu_{i+1}", fontsize=fontsize * 0.75)

            ax[i].tick_params(labelsize=fontsize * 0.75)
        plt.xlabel("Time steps (15 min)", fontsize=fontsize)

    plt.legend(fontsize=fontsize * 0.75)
    plt.savefig(f"img/unc_ts_{exp_name}")


def plot_unc_ts_loop(tol, exp_name, band="", n_iter=30, ylim=(0,1), size=(16, 32), fontsize=40):
    df_zones_unc_ts_mean, df_zones_unc_ts_std = {}, {}
    exp_name_final = f"{exp_name}_iters"
    all_zones = [1,2]
    band = f"_{band}" if band != "" else ""

    # extract zone number
    zone_idx = exp_name.index("zone")
    zone_num = exp_name[zone_idx+4:zone_idx+5]
    zone_num = all_zones if zone_num == "s" else [zone_num]
    
    # unc over time
    for z in zone_num:
        df_iters = pd.DataFrame()

        for iter in range(1, n_iter+1):
            df_aux = pd.DataFrame(load_variable(f"{exp_name}/dict_zones_unc_ts{band}_{iter}.pkl")[f"Zone_{z}"])
            df_iters = df_iters.append(df_aux)
            
        # calculate mean and std 
        df_zones_unc_ts_mean[z] = df_iters.groupby(df_iters.index).mean()        
        # df_zones_unc_ts_mean[z] = df_zones_unc_ts_mean[z].sort_index(axis=0, key=natsort_keygen())
        df_zones_unc_ts_std[z] = df_iters.groupby(df_iters.index).std()

    num_occupants = len(df_zones_unc_ts_mean[zone_num[0]].columns)
    
    # actual plot
    plot_unc_ts(
        df_zones_unc_ts_mean, 
        df_zones_unc_ts_std, 
        exp_name_final, 
        num_occupants, 
        zone_num,
        legend_loc="upper right" if tol == "100" else "lower right",
        ylim=ylim,
        size=size, 
        fontsize=fontsize
    )

def plot_unc_ts(df_mean, df_std, exp_name, num_occupants, zone_num, legend_loc, ylim=(0,1), size=(16, 32), fontsize=40):
    rc("text.latex", preamble=r"\usepackage{cmbright}")
    rc("text", usetex=True)

    fig, ax = plt.subplots(num_occupants, 1, figsize=size, sharex=True)
    
    for zone_id in zone_num:        
        for i in range(0, num_occupants):
            # extract lists

            x_axis_mean = df_mean[zone_id][[f"user_{i+1}"]].values.flatten().tolist()
            x_axis_std = df_std[zone_id][[f"user_{i+1}"]].values.flatten().tolist()
        
            lower_bound = [mean - 2*std for mean, std in zip(x_axis_mean, x_axis_std)]
            upper_bound = [mean + 2*std for mean, std in zip(x_axis_mean, x_axis_std)]

            # some nice colors for plots
            color = "C1" if zone_num == ['2'] else "C0"
            if len(zone_num) == 1:
                ax[i].plot(x_axis_mean, label=f"Zone {zone_id}", color=color, linewidth=5)
                ax[i].fill_between(np.arange(0, len(x_axis_mean)), lower_bound, upper_bound, alpha=0.3, color=color)
            else:
                ax[i].plot(x_axis_mean, label=f"Zone {zone_id}", linewidth=5)
                ax[i].fill_between(np.arange(0, len(x_axis_mean)), lower_bound, upper_bound, alpha=0.3)
            
            for j in range(1, 5):
                ax[i].vlines(96*j, 0, 1, linestyles="dashed", colors="k")
            
            # ax[i].hlines(0.5, 0, 96*5, linestyles="dotted", colors="k", linewidth=0.75)
            ax[i].set_ylabel(f"occu_{i+1}", fontsize=fontsize * 0.75)
            ax[i].tick_params(labelsize=fontsize * 0.75)
            ax[i].set_ylim(ylim)        
            # ax[i].set_yticks([0, 1])
            
            
        x_ticks = [48*i for i in range(0, 11)]
        labels = ["0", "12", "24", "12", "24", "12", "24", "12", "24", "12", "24"]
        ax[i].set_xticks(x_ticks)
        ax[i].set_xticklabels(labels)
        plt.xlabel("Hour of the day", fontsize=fontsize)

    plt.legend(fontsize=fontsize * 0.75, loc=legend_loc, ncol=2)

    plt.tight_layout()
    plt.savefig(f"img/unc_ts_{exp_name}")
    

def vote_by_user(
    dataframe,
    dataset="dorn",
    show_percentages=False,
    preference_label="thermal_cozie",
    fontsize=40,
):
    """
    Original code by Dr. Federico Tartarini
    https://github.com/FedericoTartarini
    """

    rc("text.latex", preamble=r"\usepackage{cmbright}")
    rc("text", usetex=True)

    df = dataframe.copy()
    df[preference_label] = df[preference_label].map(
        {9.0: "Warmer", 10.0: "No Change", 11.0: "Cooler"}
    )
    _df = (
        df.groupby(["user_id", preference_label])[preference_label]
        .count()
        .unstack(preference_label)
    )
    _df.reset_index(inplace=True)

    df_total = _df.sum(axis=1)
    df_rel = _df[_df.columns[1:]].div(df_total, 0) * 100
    df_rel["user_id"] = _df["user_id"]

    # sort properly
    df_rel["user_id"] = df_rel["user_id"].str.replace(dataset, "").astype(int)
    df_rel = df_rel.sort_values(by=["user_id"], ascending=False)
    df_rel["user_id"] = dataset + df_rel["user_id"].astype(str)
    df_rel = df_rel.reset_index(drop=True)

    # plot a Stacked Bar Chart using matplotlib
    df_rel.plot(
        x="user_id",
        kind="barh",
        stacked=True,
        mark_right=True,
        cmap=LinearSegmentedColormap.from_list(
            preference_label,
            [
                "tab:blue",
                "tab:green",
                "tab:red",
            ],
            N=3,
        ),
        width=0.95,
        figsize=(16, 16),
    )

    plt.legend(
        bbox_to_anchor=(0.5, 1.02),
        loc="center",
        borderaxespad=0,
        ncol=3,
        frameon=False,
        fontsize=fontsize,
    )
    sns.despine(left=True, bottom=True, right=True, top=True)

    plt.tick_params(labelsize=fontsize * 0.75)
    plt.xlabel(r"Percentage [\%]", size=fontsize)
    plt.ylabel("User ID", size=fontsize)

    if show_percentages:
        # add percentages
        for index, row in df_rel.drop(["user_id"], axis=1).iterrows():
            cum_sum = 0
            for ix, el in enumerate(row):
                if ix == 1:
                    plt.text(
                        cum_sum + el / 2 if not np.isnan(cum_sum) else el / 2,
                        index,
                        str(int(np.round(el, 0))) + "\%",
                        va="center",
                        ha="center",
                        size=fontsize * 0.6,
                    )
                cum_sum += el

    plt.tight_layout()
    plt.savefig(f"img/{dataset}_vote_dist.png", pad_inches=0, dpi=300)
    plt.show()


def tp_dist(df, tol, folder_name, size=(16, 6), fontsize=40):
    _df = df[["t-ubi", "thermal_cozie"]]
    _df["Thermal Preference"] = _df["thermal_cozie"].map(
        {9.0: "Warmer", 10.0: "No Change", 11.0: "Cooler"}
    )

    if tol == 0.1:
        palette = ["#1f77b4", "#2c9f2c", "#d62728"]  # blue, green, red
    else:
        palette = ["#2c9f2c", "#1f77b4", "#d62728"]  # green, blue, red

    fig, ax = plt.subplots(figsize=size)
    kde = sns.kdeplot(
        data=_df,
        x="t-ubi",
        hue="Thermal Preference",
        bw_adjust=0.5,
        palette=palette,
        ax=ax
    )

    plt.xlabel("Indoor Temperature [°C]", size=fontsize)
    plt.ylabel("Density", size=fontsize)
    plt.tick_params(labelsize=fontsize * 0.75)
    plt.setp(ax.get_legend().get_texts(), fontsize=fontsize*0.5) # for legend text
    plt.setp(ax.get_legend().get_title(), fontsize=fontsize*0.75) # for legend title
    plt.tight_layout()
    plt.savefig(f"{folder_name}/preferences_dist_tol{tol}.png")

    return kde


def plot_temperatures(df, ylim=(24, 33.5), fontsize=40, size=(16, 16)):
    rc("text.latex", preamble=r"\usepackage{cmbright}")
    rc("text", usetex=True)

    fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=size)

    palette = {"Outside": "C2", "Zone 1": "C0", "Zone 2": "C1"}

    x = sns.violinplot(
        data=df, 
        cut=0,
        scale="count",
        inner="quartile",
        linewidth=1,
        ax=ax,
        palette=palette
    )

    plt.ylabel("Temperature [°C]", size=fontsize)
    plt.ylim(ylim)
    plt.tick_params(labelsize=fontsize * 0.75)

    ax.grid(axis="y", alpha=0.3)
    plt.xlabel("Zones", size=fontsize)
    plt.tight_layout()
    plt.savefig("img/zones_temperatures.png")

    
def plot_bands_ts(
    folder, 
    band1=[], 
    band2=[], 
    all_zones=[1,2],
    ylim=(19.5, 28), 
    fontsize=40, 
    size=(16,8)
):
    rc("text.latex", preamble=r"\usepackage{cmbright}")
    rc("text", usetex=True)

    fig, ax = plt.subplots(figsize=size)

    for zone_id, i in zip(all_zones, range(2)):
        df_data = pd.read_csv(f"data/{folder}/Zone_{zone_id}.csv")
        color = "C1" if zone_id == 2 else "C0"
        ax.plot(df_data.loc[0:480]["t_in"], label=f"Zone {zone_id}", color=color, linewidth=5)

    ax.fill_between(np.arange(0, 480), band1[0], band1[1], alpha=0.3, color="C4", label="Band 1")
    ax.fill_between(np.arange(0, 480), band2[0], band2[1], alpha=0.3, color="C5", label="Band 2")

    for j in range(1, 5):
        ax.vlines(96*j, 19.50, 27.5, linestyles="dashed", colors="k")

    ax.tick_params(labelsize=fontsize * 0.75)
    
    x_ticks = [48*i for i in range(0, 11)]
    labels = ["0", "12", "24", "12", "24", "12", "24", "12", "24", "12", "24"]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(labels)
    
    plt.ylim(ylim)
    plt.xlabel("Hour of the day", fontsize=fontsize)
    plt.ylabel("Temperature [°C]", size=fontsize)
    plt.legend(fontsize=fontsize * 0.75, ncol=2, loc="center right")
    plt.tight_layout()
    plt.savefig(f"img/bands_ts")    
