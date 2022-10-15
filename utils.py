"""
Here, we create some common util functions for regression. 
"""
#%%
import enum
import os 
import sys
import numpy as np
from numpy.core.fromnumeric import var
import scipy 
import pandas as pd
import matplotlib as mpl
#mpl.use('Agg', warn= True)
import matplotlib.pyplot as plt
from seaborn.rcmod import axes_style

if sys.platform == 'win32':
    home = 'D:\\'
else:
    home=os.path.expanduser('~')
from functools import reduce 
import scipy.interpolate as sci
import scipy.optimize as sco
import seaborn as sns

from sklearn import metrics

import warnings

sys.path.append(os.path.join(home, 'repo', 'research_current', 'VisSoft'))
#plt.style.use(os.path.join(home, 'repo', 'mplstyles', 'mypaper.mplstyle'))
plt.style.use(os.path.join(home, 'repo', 'mplstyles', 'mypresentation.mplstyle'))


def make_histograms(df_red, variable_list, ncols=5):
    """
    This is a simple function to make histograms to do EDA on the data we have. 
    """
    ncols = ncols; nrows = int(np.ceil(len(variable_list)/ncols))
    #Plot histograms of each of the variables. 
    fig_hist, axes_list = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*7, nrows*6))
    axes_list = axes_list.ravel()
    fig_hist.subplots_adjust(wspace=0.4, hspace=0.2)
    for var_num, variable in enumerate(variable_list):
        #print(variable)
        ax = axes_list[var_num]
        ax2 = ax.twinx()
        ax.hist(df_red[variable], bins=10)

        #Also added a boxplot.
        # max. whisker length Q3 + 1.5 * IQR or Q1 - 1.5 * IQR.
        sns.boxplot(x=df_red[variable], whis=1.5, width=0.1, zorder=1, ax=ax2, color="white",)


        ax.set_xlabel(variable)
        ax.set_ylabel("Count")

def make_count_tables(df_red, variable_list, ncols=5):
    """
    This is a simple function to make histograms to do EDA on the data we have. 
    """
    ncols = ncols; nrows = int(np.ceil(len(variable_list)/ncols))
    #Plot histograms of each of the variables. 
    fig_hist, axes_list = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*7, nrows*6))
    if nrows * ncols > 1:    
        axes_list = axes_list.ravel()
    fig_hist.subplots_adjust(wspace=0.7, hspace=0.3)

    #initialize a dummy variable.
    df_red = df_red.copy(deep=True)
    df_red["Count"] = 0
    for var_num, variable in enumerate(variable_list):
        if nrows * ncols > 1:
            ax = axes_list[var_num]
        else:
            ax = axes_list

        #ax.hist(df_red[variable], bins=10)
        sns.heatmap(df_red.groupby(variable).\
        count().sort_values("Count", ascending=False).\
            loc[:, ["Count"]], annot=True, ax=ax, fmt=".0f",
            )
        ax.set_xlabel("")


def plot_evaluation_curves(gs, grid_params):
    scoring_parameter = list(gs.scorer_.keys())[0]
    print(f"Scoring parameter: {scoring_parameter}")
    df = pd.DataFrame(gs.cv_results_)
    results = [f'mean_test_{scoring_parameter}',
               f'mean_train_{scoring_parameter}',
               f'std_test_{scoring_parameter}', 
               f'std_train_{scoring_parameter}']

    def pooled_var(stds):
        # https://en.wikipedia.org/wiki/Pooled_variance#Pooled_standard_deviation
        n = gs.cv #3 # size of each group
        return np.sqrt(sum((n-1)*(stds**2))/ len(stds)*(n-1))

    fig, axes = plt.subplots(1, len(grid_params), 
                             figsize = (7*len(grid_params), 6),
                             sharey='row')
    if len(grid_params) == 1:
        axes = [axes]

    axes[0].set_ylabel("Score", fontsize=12)

    for idx, (param_name, param_range) in enumerate(grid_params.items()):
        grouped_df = df.groupby(f'param_{param_name}')[results]\
            .agg({f'mean_train_{scoring_parameter}': 'mean',
                  f'mean_test_{scoring_parameter}': 'mean',
                  f'std_train_{scoring_parameter}': pooled_var,
                  f'std_test_{scoring_parameter}': pooled_var})

        previous_group = df.groupby(f'param_{param_name}')[results]
        axes[idx].set_xlabel(param_name, fontsize=12)
        axes[idx].set_ylim(0.1, 1.1)
        lw = 2
        axes[idx].plot(param_range, grouped_df[f'mean_train_{scoring_parameter}'], label=f"Training {scoring_parameter}",
                    color="darkorange", lw=lw)
        axes[idx].fill_between(param_range,grouped_df[f'mean_train_{scoring_parameter}'] - grouped_df[f'std_train_{scoring_parameter}'],
                        grouped_df[f'mean_train_{scoring_parameter}'] + grouped_df[f'std_train_{scoring_parameter}'], alpha=0.2,
                        color="darkorange", lw=lw)
        axes[idx].plot(param_range, grouped_df[f'mean_test_{scoring_parameter}'], label="Cross-validation score",
                    color="navy", lw=lw)
        axes[idx].fill_between(param_range, grouped_df[f'mean_test_{scoring_parameter}'] - grouped_df[f'std_test_{scoring_parameter}'],
                               grouped_df[f'mean_test_{scoring_parameter}'] + grouped_df[f'std_test_{scoring_parameter}'], alpha=0.2,
                               color="navy", lw=lw)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle('Validation curves', fontsize=12)
    fig.legend(handles, labels, loc=8, ncol=2, fontsize=12)

    fig.subplots_adjust(bottom=0.25, top=0.85)  
    plt.show()
    
def validate_regression_model(y_train, y_train_pred, y_test, y_test_pred, title="", ax_lims_to_show=None, return_figure=False):
    """
    This function will validate the model.
    """

    if ax_lims_to_show is None:
        #ax_lims_to_show = (min(min(y_train), min(y_pred)), max(max(y), max(y_pred)))
        y_list = list(y_train) + list(y_test) + list(y_train_pred) + list(y_test_pred)
        ax_lims_to_show = (min(y_list), max(y_list))

    #fig_reg = plt.figure(figsize=(14, 12))
    fig_reg = plt.figure(figsize=(21, 6))
    #ax_reg = fig_reg.add_subplot(221)
    ax_reg = fig_reg.add_subplot(131)
    ax_reg.scatter(y_test, y_test_pred, label="Test", color=plt.cm.Set1(0))
    ax_reg.scatter(y_train, y_train_pred, label="Train", color=plt.cm.Set1(1))
    ax_reg.set_xlim(ax_lims_to_show)
    ax_reg.set_ylim(ax_lims_to_show)
    ax_reg.plot([0, 1], [0, 1], ls='--', color=plt.cm.Greys(200), transform=ax_reg.transAxes)
    ax_reg.set_xlabel("Actual")
    ax_reg.set_ylabel("Predicted")
    #ax_reg.text(0.01, 0.99, f"R2 = {r2_score(y, y_pred)}", horizontalalignment='left',
    #    verticalalignment='top', transform=ax_reg.transAxes)
    ax_reg.set_title(title)

    #ax_res = fig_reg.add_subplot(222)
    ax_res = fig_reg.add_subplot(132)
    ax_res.scatter(y_test, y_test - y_test_pred, label="Test", color=plt.cm.Set1(0))
    ax_res.scatter(y_train, y_train - y_train_pred, label="Train", color=plt.cm.Set1(1))
    ax_res.set_xlim(ax_lims_to_show)
    ax_res.plot(ax_lims_to_show, [0, 0], ls='--', color=plt.cm.Greys(200))
    ax_res.set_xlabel("Actual")
    ax_res.set_ylabel("Residuals")
    ax_res.legend(loc="upper right")

    #ax_hist = fig_reg.add_subplot(223)
    ax_hist = fig_reg.add_subplot(133)
    ax_hist.hist(y_test - y_test_pred, label="Test", color=plt.cm.Set1(0))
    ax_hist.hist(y_train - y_train_pred, label="Train", color=plt.cm.Set1(1))
    ax_hist.set_xlabel("Residuals")
    ax_hist.set_ylabel("Counts")

    #ax_coef = fig_reg.add_subplot(224)
    #ax_coef.plot(np.arange(coefs_.shape[0]) + 1, coefs_, marker="o", ls="--")
    #ax_coef.set_xlabel("Coef Index")
    #ax_coef.set_ylabel("Coefficient")

    #plot residuals, histogram, coefficients. 
    if return_figure:
        return fig_reg


def make_violinplots(df_red, feature_list, target_var, ncols=5, color=plt.cm.Set1(0),
        return_fig=False):
    """
    This is a simple function to make boxplots when the target variable is categorical.
    """
    ncols = ncols; nrows = int(np.ceil(len(feature_list)/ncols))
    #Plot histograms of each of the variables. 
    fig_box, axes_list = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*7, nrows*6))
    axes_list = axes_list.ravel()
    fig_box.subplots_adjust(wspace=0.4, hspace=0.2)
    for var_num, variable in enumerate(feature_list):
        ax = axes_list[var_num]
        sns.violinplot(x=target_var, y=variable, data=df_red, color=color, scale="count", inner="quartile", ax=ax)
        ax.set_xlabel(target_var)
        ax.set_ylabel(variable)
    if return_fig:
        return fig_box


def make_contingency_tables_for_binary(df_red, feature_list, target_var, ncols=5, color=plt.cm.Set1(0),
        return_fig=False):
    """
    Simple function to make contingency tables when the target variable is categorical.
    Here, target_var must have "0" and "1" as the values. This is hardcoded. 
    ONLY works for binary classification. 
    How to intepret
    """
    ncols = ncols; nrows = int(np.ceil(len(feature_list)/ncols))
    #Plot histograms of each of the variables. 
    fig_box, axes_list = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*7, nrows*6))
    axes_list = axes_list.ravel()
    fig_box.subplots_adjust(wspace=0.4, hspace=0.2)
    for var_num, variable in enumerate(feature_list):
        ax = axes_list[var_num]
        df_cont = pd.crosstab(index=df_red[variable], columns=df_red[target_var])
        df_cont_new = pd.DataFrame.from_dict(df_cont.to_dict())
        df_cont_new["sum_row"] = df_cont_new.sum(axis=1)
        
        df_cont_new["percentage_0"] = df_cont_new[0] / df_cont_new["sum_row"] *100
        df_cont_new["percentage_1"] = df_cont_new[1] / df_cont_new["sum_row"] *100
        df_cont_new["percentage_point_diff"] = df_cont_new["percentage_1"] - df_cont_new["percentage_0"]
        sns.heatmap(df_cont_new.sort_values(by="sum_row", ascending=False)\
            [[0, 1, "sum_row", "percentage_point_diff"]], annot=True, ax=ax, fmt=".0f")
        ax.set_ylabel(variable)

    if return_fig:
        return fig_box


def make_contingency_tables_for_multiclass(df_red, feature_list, target_var, ncols=5, color=plt.cm.Set1(0),
        return_fig=False):
    """
    Function to make contingency tables when the target variable is categorical.
    Works for multiclass classification. 
    How to intepret. 
    """
    ncols = ncols; nrows = int(np.ceil(len(feature_list)/ncols))
    #Plot histograms of each of the variables. 
    fig_box, axes_list = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*7, nrows*6))
    if nrows * ncols > 1:
        axes_list = axes_list.ravel()

    fig_box.subplots_adjust(wspace=0.4, hspace=0.2)
    for var_num, variable in enumerate(feature_list):
        if nrows * ncols > 1:
            ax = axes_list[var_num] 
        else:
            ax =  axes_list
        df_cont = pd.crosstab(index=df_red[variable], columns=df_red[target_var])
        df_cont_new = pd.DataFrame.from_dict(df_cont.to_dict())
        df_cont_new["sum_row"] = df_cont_new.sum(axis=1)
        
        # df_cont_new["percentage_0"] = df_cont_new[0] / df_cont_new["sum_col"] *100
        # df_cont_new["percentage_1"] = df_cont_new[1] / df_cont_new["sum_col"] *100
        # df_cont_new["percentage_point_diff"] = df_cont_new["percentage_0"] - df_cont_new["percentage_1"]
        sns.heatmap(df_cont_new.sort_values(by="sum_row", ascending=False),
            annot=True, ax=ax, fmt=".0f")
            #[["sum_col", "percentage_point_diff"]], 
        ax.set_ylabel(variable)

    if return_fig:
        return fig_box


def make_correlation_table(df, variable_list, figsize=(7,6), fontsize=14):
    """
    This is a wrapper around the seaborn heatmap further optimized to view correlations
    between features. 
    """
    plt.figure(figsize=figsize)
    corr = df[variable_list].corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f", 
        annot_kws={"fontsize":fontsize}, vmin=-1, vmax=1, 
        center=0, cmap=plt.cm.turbo, cbar_kws={"label": "Correlation coefficient"})





def plot_roc_curve(X, y, model, ax=None, pos_label=None, label="Test", return_ax_and_vals=False):
    """
    Plot the ROC curve for the given X and y.
    """
    if ax is None:
        ax = plt.gca()
    if pos_label is None:
        pos_label = model.classes_[-1]

    lr_probs = model.predict_proba(X)
    lr_fpr, lr_tpr, _ = metrics.roc_curve(y.values, lr_probs[:,1], pos_label=pos_label)
    # plot the roc curve for the model
    auc_roc = metrics.auc(lr_fpr, lr_tpr)
    ax.plot(lr_fpr, lr_tpr, label=f"{label} (AUC={auc_roc:.3f})")
    if return_ax_and_vals:
        return [ax, lr_probs, lr_fpr, lr_tpr, auc_roc]

def validate_classification_model(y_train, y_train_pred, y_test, y_test_pred, title="", return_figure=False,
        get_auc_roc=False, model=None, X_train=None, X_test=None, pos_label=None, classes_ = None):
    """
    Here, we build function to analyze how well the model performs.
    We make confusion matrix for training, test, AUC ROC curve, classification report. 
    model: serves two purposes: 1. for getting ROC curve, 2. getting classes. If model not specified, we must specify
    classes as an iterable. 
    """

    if model is not None:
        classes_ = model.classes_
    else:
        classes_ = classes_

    fig_class = plt.figure(figsize=(14, 6))
    ax_tr_cf = fig_class.add_subplot(121)
    tr_con = metrics.confusion_matrix(y_train, y_train_pred, labels=classes_)
    metrics.ConfusionMatrixDisplay(tr_con, display_labels=classes_).plot(ax=ax_tr_cf)
    ax_tr_cf.set_title("Train")

    ax_test_cf = fig_class.add_subplot(122)
    test_con = metrics.confusion_matrix(y_test, y_test_pred, labels=classes_)
    metrics.ConfusionMatrixDisplay(test_con, display_labels=classes_).plot(ax=ax_test_cf)
    ax_test_cf.set_title("Test")

    fig_class.suptitle(title, fontsize=16)

    if get_auc_roc:
        fig_auc = plt.figure()
        ax_auc_roc = fig_auc.add_subplot(111)
        ax_auc_roc.plot([0, 1], [0, 1], linestyle='--', label='No Skill', transform=ax_auc_roc.transAxes)
        plot_roc_curve(X_test, y_test, model, ax=ax_auc_roc, pos_label=pos_label, label="Test")
        plot_roc_curve(X_train, y_train, model, ax=ax_auc_roc, pos_label=pos_label, label="Train")
        # axis labels
        ax_auc_roc.set_xlabel('False Positive Rate')
        ax_auc_roc.set_ylabel('True Positive Rate')
        ax_auc_roc.legend(loc="lower right")

    fig_rep = plt.figure(figsize=(14, 6))
    ax_rep_train = fig_rep.add_subplot(121)
    #We will also plot the classification reports. 
    cr_train = metrics.classification_report(y_train, y_train_pred, output_dict=True)
    #plt.figure()
    #sns.heatmap(pd.DataFrame(cr_train).iloc[:-1, :].T, annot=True, fmt=".2f", annot_kws={"fontsize": 12}, vmin=0, vmax=1.0, cmap=plt.cm.turbo)
    #plt.title("Train")
    sns.heatmap(pd.DataFrame(cr_train).iloc[:-1, :].T, annot=True, fmt=".2f", annot_kws={"fontsize": 12}, vmin=0, vmax=1.0, cmap=plt.cm.turbo, ax=ax_rep_train)
    ax_rep_train.set_title("Train")

    cr_test = metrics.classification_report(y_test, y_test_pred, output_dict=True)
    # plt.figure()
    # sns.heatmap(pd.DataFrame(cr_test).iloc[:-1, :].T, annot=True, fmt=".2f", annot_kws={"fontsize": 12}, vmin=0, vmax=1.0, cmap=plt.cm.turbo)
    # plt.title("Test")
    ax_rep_test = fig_rep.add_subplot(122)
    sns.heatmap(pd.DataFrame(cr_test).iloc[:-1, :].T, annot=True, fmt=".2f", annot_kws={"fontsize": 12}, vmin=0, vmax=1.0, cmap=plt.cm.turbo, ax=ax_rep_test)
    ax_rep_test.set_title("Test")

    fig_rep.subplots_adjust(wspace=0.15)

    #plot residuals, histogram, coefficients. 
    if return_figure:
        return fig_class

def combine_X_and_y(X, y, y_pred, model_name="", set_="Train", pred_desc="pred", 
    extra_variables={}, pos_label="Yes"):
    """
    Combine X, y and y_pred values for the training and test sets. 
    X : Pandas dataframe with columns as features
    y: Pandas series object with "Name" as the variable name.
    y_pred : numpy array or list but not a Pandas df.  
    extra_variables: Dictionary with keys as the variable names and the values as methods that transforms the X into a vector. 
    ASSUMPTIONS: 
    1. The X and y orders are the same; i.e., 1st row of X must correspond to 1st row of y. 
    2. If this is to be combined with the original dataframe (with columns that are not features, the index of X must not be tampered with).  
    """
    X_comb = X.copy(deep=True)
    if X_comb.shape[0] == y.shape[0]:
        X_comb[f"{model_name}_TrainTestLabel"] = set_
        X_comb[f"{model_name}_{y.name}"] = y.values
        X_comb[f"{model_name}_{y.name}_{pred_desc}"] = y_pred

        #Add other columns if necessary. 
        for var in extra_variables.keys():
            #Here, we can take other variables and combine them easily with the original dataframe. 
            X_comb[f"{model_name}_{var}"] = extra_variables[var](X)
        return X_comb
    else: 
        raise Exception("X and y don't have same lengths")

def merge_train_and_test(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred,
    model_name="", pred_desc="pred", extra_variables={}, 
    sort_index=True):
    """
    Here, we will merge training and test sets vertically with the remaining columns 
    from the original dataset and reorganize the dataframe. 
    """

    df_train = combine_X_and_y(X_train, y_train, y_train_pred, model_name=model_name, set_="Train", pred_desc=pred_desc, extra_variables=extra_variables)
    df_test = combine_X_and_y(X_test, y_test, y_test_pred, model_name=model_name, set_="Test", pred_desc=pred_desc, extra_variables=extra_variables)

    df_comb = pd.concat([df_train, df_test])

    if sort_index:
        df_comb.sort_index(inplace=True)
    return df_comb

def combine_df_with_original(df_comb, df_original, og_columns_to_merge=[], index_var="Name", 
    rearrange_columns=True):
    """
    """
    df_comb = df_comb.copy(deep=True)
    og_columns_to_merge_in_comb = [col for col in og_columns_to_merge if col in df_comb.columns.to_list()]
    if len(og_columns_to_merge_in_comb) > 0:
        warnings.warn(f"Some columns in og_columns_to_merge are also present as features / targets. \
            This is not recommended. Please remove them. Those columns are: \n {og_columns_to_merge_in_comb}")
        og_columns_to_merge = [col for col in og_columns_to_merge if col not in df_comb.columns.to_list()]

    df_comb = df_comb.merge(df_original[og_columns_to_merge], left_index=True, right_index=True)

    if rearrange_columns:
        if index_var != "": 
            cols = df_comb.columns.to_list()
            cols.remove(index_var)
            cols = [index_var] + cols
            df_comb = df_comb[cols]

    return df_comb


def make_catplots(df_comb, feature_list, target_var, hue_var, ncols=5, return_fig=False, 
    title=""):
    """
    Make catplots with respect to each feature. Typically, we would like to understand where something has been
    misclassified / rightly classified for the training and/or test sets.  
    """
    ncols = ncols; nrows = int(np.ceil(len(feature_list)/ncols))
    fig, axes_list = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*7, nrows*6))
    axes_list = axes_list.ravel()
    fig.subplots_adjust(wspace=0.4, hspace=0.2)
    for feature_num, feature in enumerate(feature_list):
        ax = axes_list[feature_num]
        #sns.catplot(data=df_comb, x=target_var, y=feature, kind="swarm",
        #    hue=hue_var, legend=False, legend_out=False, ax=ax)
        sns.swarmplot(data=df_comb, x=target_var, y=feature, 
            hue=hue_var, ax=ax)
        ax.set_ylabel(feature)
        ax.set_xlabel(target_var)
        if feature_num == (len(feature_list)-1): #last feature
            ax.legend(loc="upper center")
        else:
            ax.get_legend().set_visible(False)
    #fig.supxlabel(target_var) #will be avaialbel in MPL 3.4
    fig.suptitle(title) #will be avaialbel in MPL 3.4

    if return_fig:
        return fig

def make_pairwise_catplots(df_comb, feature_list_1, feature_list_2, hue_var, return_fig=False, 
    title="", row_feature = "first"):
    """
    Make catplots with respect to each feature. Typically, we would like to understand where something has been
    misclassified / rightly classified for the training and/or test sets.  
    """
    if row_feature == "first":
        nrows = len(feature_list_1); ncols = len(feature_list_2)
    else:
        nrows = len(feature_list_2); ncols = len(feature_list_1)

    fig, axes_list = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*7, nrows*6))
    #axes_list = axes_list.ravel()
    fig.subplots_adjust(wspace=0.4, hspace=0.2)
    for feature_num_1, feature_1 in enumerate(feature_list_1):
        for feature_num_2, feature_2 in enumerate(feature_list_2):
            if row_feature == "first":
                ax = axes_list[feature_num_1, feature_num_2]
            else:
                ax = axes_list[feature_num_2, feature_num_1]

            #sns.catplot(data=df_comb, x=target_var, y=feature, kind="swarm",
            #    hue=hue_var, legend=False, legend_out=False, ax=ax)
            # sns.swarmplot(data=df_comb, x=feature_1, y=feature_2, 
            #     hue=hue_var, ax=ax)
            sns.stripplot(data=df_comb, x=feature_1, y=feature_2, 
                hue=hue_var, ax=ax)
            ax.set_xlabel(feature_1)
            ax.set_ylabel(feature_2)
            if (feature_num_1 == (len(feature_list_1)-1)) & \
                (feature_num_2 == (len(feature_list_2)-1) ): #last feature
                ax.legend(loc="upper center")
            else:
                ax.get_legend().set_visible(False)
    #fig.supxlabel(target_var) #will be avaialbel in MPL 3.4
    fig.suptitle(title) #will be avaialbel in MPL 3.4

    if return_fig:
        return fig




if __name__=="__main__":

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    #We can test the functions.
    tips = sns.load_dataset("tips")
    tips["index_var"] = tips.index

    feature_list = ["total_bill", "tip"]
    target_var = "sex"
    #make_violinplots(tips, feature_list, target_var)

    clf = RandomForestClassifier(random_state=1000)
    X_train, X_test, y_train, y_test = train_test_split(tips[feature_list], tips[target_var], test_size=0.2)
    clf.fit(X_train, y_train)
    y_train_pred = clf.predict(X_train)
    y_test_pred = clf.predict(X_test)
    y_test_pred_proba = clf.predict_proba(X_test)[:, 1]

    #validate_classification_model(y_train, y_train_pred, y_test, y_test_pred, clf, title="RF",
    #get_auc_roc=True, X_train=X_train, X_test=X_test)
    model_name = "RF"
    #df_comb = merge_train_and_test(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred, 
    #    model_name="RF", pred_desc="pred", extra_variables={}, sort_index=True)
    df_comb = merge_train_and_test(X_train, y_train, y_train_pred, X_test, y_test, y_test_pred, 
        model_name="RF", pred_desc="pred", extra_variables={"sex_pred_proba": lambda X: clf.predict_proba(X)[:, -1]}, sort_index=True)
    df_comb_w_og = combine_df_with_original(df_comb, tips, og_columns_to_merge=["size", "day", "index_var"], index_var = "index_var",
         rearrange_columns=True)

    #Assigning the various categories to the combined dataframe and plotting it. 
    class_combs = [("Male", "Male", "TP"), ("Male", "Female", "FN"), ("Female", "Male", "FP"), ("Female", "Female", "TN")]
    df_comb[f"RF_{target_var}_class_stat"] = "" #initializing. 

    for class_comb in class_combs:
        truth=class_comb[0]; prediction=class_comb[1]; label=class_comb[2]
        print(f"{truth}  + {prediction} + {label}")
        #df_comb.loc[(df_comb[f"RF_{target_var}"]==truth) & (df_comb[f"RF_{target_var}_pred"]==prediction)
        #        ,f"RF_{target_var}_class-stat"] = label  
        #df_comb.query(f'({model_name}_{target_var} == "{truth}") & ({model_name}_{target_var}_pred == "{prediction}")')\
        #    [f"RF_{target_var}_class-stat"] = label
        idx = df_comb.query(f'{model_name}_{target_var} == "{truth}" & {model_name}_{target_var}_pred == "{prediction}"').index
        display(df_comb.loc[idx, :].head())
        df_comb.loc[idx, f"RF_{target_var}_class-stat"] = label

    make_catplots(df_comb, feature_list, target_var=f"RF_{target_var}", hue_var=f"RF_{target_var}_class-stat")



# %%
