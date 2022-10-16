"""
Some common util functions for EDA.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use('mypresentation.mplstyle')


def show_filled_value_percentages(data, columns,
                                ylabel="Percentage of filled values (%)"):
    """Shows the percentage of filled values for each column in the dataset.

    Creates a new figure to show the percentage of filled values per variable.

    Parameters
    ----------
    data: DataFrame
        The data frame to analyze.
    columns: list
        The list of columns for which we want to inspect the
        percentage of filled values.
    ylabel: str, optional
        Label to print on the y axis.

    Returns
    -------
        None
    """

    plt.figure()
    filled_series = data[columns].count() / data.shape[0] * 100
    filled_series.sort_values(ascending=False).plot.bar().\
        set_ylabel(ylabel)


def make_histograms(df, columns, ncols=5):
    """Makes histograms to visualize distribution of the numerical variables in the list.

    Creates a new figure to show the distributions of each numerical variable in the list
    along with a box plot. In the whisker plot, the box represents 25th (Q1), 50th, and 75th (Q3)
    quartiles. The interquartile range (IQR) is given by Q3-Q1. The whiskers represent
    the extremes of the distribution. These are max(min(df[variable]), Q1 - 1.5 * IQR) and
    min(max(df[variable], Q3 + 1.5 * IQR)).

    Parameters
    ----------
    df: DataFrame
        The data frame to analyze.
    columns: list
        The list of columns whose distribution we want to see. ONLY include numerical columns.
        The rest will be ignored.
    ncols: int, optional
        Number of figures to show in each row.

    Returns
    -------
        None
    """

    nrows = int(np.ceil(len(columns)/ncols))

    fig_hist, axes_list = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*7, nrows*6))
    axes_list = axes_list.ravel()
    fig_hist.subplots_adjust(wspace=0.4, hspace=0.2)

    for var_num, variable in enumerate(columns):
        ax = axes_list[var_num]
        ax2 = ax.twinx()
        ax.hist(df[variable], bins=10)

        #Also added a boxplot.
        # max. whisker length Q3 + 1.5 * IQR or Q1 - 1.5 * IQR.
        sns.boxplot(x=df[variable], whis=1.5, width=0.1, zorder=1, ax=ax2, color="white")

        ax.set_xlabel(variable)
        ax.set_ylabel("Count")


def make_count_tables(df, categorical_columns, ncols=5):
    """Creates a count table for categorical variables.

    Creates a new figure to show the counts of each value encountered in the
    categorical variables we wish to analyze.

    Parameters
    ----------
    df: DataFrame
        The data frame to analyze.
    categorical_columns: list
        The list of categorical (ONLY) columns whose counts we wish to visualize.
    ncols: int, optional
        Number of figures to show in each row.

    Returns
    -------
        None
    """

    nrows = int(np.ceil(len(categorical_columns)/ncols))

    fig_hist, axes_list = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*7, nrows*6))
    if nrows * ncols > 1:
        axes_list = axes_list.ravel()
    fig_hist.subplots_adjust(wspace=0.7, hspace=0.3)

    df = df.copy(deep=True)
    # Initialize a dummy variable to keep track of counts.
    df["Count"] = 0
    for var_num, variable in enumerate(categorical_columns):
        if nrows * ncols > 1:
            ax = axes_list[var_num]
        else:
            ax = axes_list

        sns.heatmap(df.groupby(variable).\
        count().sort_values("Count", ascending=False).\
            loc[:, ["Count"]], annot=True, ax=ax, fmt=".0f",
            )


def make_correlation_table(df, variable_list, figsize=(7,6), fontsize=14):
    """Creates a table to visualize correlation between numeric variables.

    Creates a new figure to show the correlation matrix between the variables we
    wish to analyze.

    Parameters
    ----------
    df: DataFrame
        The data frame to analyze.
    variable_list: list
        The list of numerical (ONLY) columns whose correlations we wish to visualize.
    figsize: tuple, optional
        Keyword argument to matplotlib.pyplot.figure to specify the figure size in inches.
    fontsize: int, optional

    Returns
    -------
        None
    """

    plt.figure(figsize=figsize)
    corr = df[variable_list].corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    sns.heatmap(corr, mask=mask, annot=True, fmt=".1f",
        annot_kws={"fontsize":fontsize}, vmin=-1, vmax=1,
        center=0, cmap=plt.cm.turbo, cbar_kws={"label": "Correlation coefficient"})


def make_contingency_tables(df, feature_list, target_variable, ncols=5,
                                    target_variable_binary=False,
                                    fig_size_per_plot=(7,6),
                                    return_figure=False
                                    ):
    """Creates contingency tables to visualize relation between categorical variables.

    Makes contingency tables (https://www.statisticshowto.com/what-is-a-contingency-table/)
    when the variables to analyze are categorical.
    Shows the number values for each category of variable in feature_list along with the
    target_variable. Also, sums up these values per row to show how many values exist per row
    of the variable in feature_list.

    Parameters
    ----------
    df: DataFrame
        The data frame to analyze.
    feature_list: list
        The list of categorical (ONLY) columns whose relations with the target variable
        we wish to visualize.
    target_variable: str
        The target variable with which the relation of each variable in the feature_list
        is visualized.
    ncols: int, optional
        Number of figures to show in each row.
    target_variable_binary: bool, optional, default: False
        Specifies if the target variable is binary. If True, the classes MUST be "0" and "1".
        When the target variable is binary (with classes labelled as "0" and "1"),
        we also return another column called "percentage_point_difference".
        This indicates the difference in percentage of values in the positive class ("1")
        and that between the negative class ("0").

        $$Percentage\ point\ difference = \frac{Freq.\ class\ 1 - Freq.\ class\ 0}
            {Freq.\ class\ 1 + Freq.\ class\ 0} \times 100 $$

        A higher difference indicates a larger separaration between the classes of the target
        variable.

    fig_size_per_plot: tuple, optional
        Keyword argument to specify the size of each panel in the figure in inches.
    return_figure: bool, optional, default: False
        Specifies whether to return a figure object when function is called.

    Returns
    -------
        Figure or None
    """

    nrows = int(np.ceil(len(feature_list)/ncols))
    width_per_plot, height_per_plot = fig_size_per_plot

    fig_box, axes_list = plt.subplots(nrows=nrows, ncols=ncols,
        figsize=(ncols*width_per_plot, nrows*height_per_plot))
    if nrows * ncols > 1:
        axes_list = axes_list.ravel()

    fig_box.subplots_adjust(wspace=0.4, hspace=0.2)
    for var_num, variable in enumerate(feature_list):
        if nrows * ncols > 1:
            ax = axes_list[var_num]
        else:
            ax =  axes_list
        df_cont = pd.crosstab(index=df[variable], columns=df[target_variable])
        df_cont_new = pd.DataFrame.from_dict(df_cont.to_dict())
        df_cont_new["sum_row"] = df_cont_new.sum(axis=1)

        if target_variable_binary:
            df_cont_new["percentage_0"] = df_cont_new[0] / df_cont_new["sum_row"] *100
            df_cont_new["percentage_1"] = df_cont_new[1] / df_cont_new["sum_row"] *100
            df_cont_new["percentage_point_diff"] = df_cont_new["percentage_1"] -\
                df_cont_new["percentage_0"]
            sns.heatmap(df_cont_new.sort_values(by="sum_row", ascending=False)\
                [[0, 1, "sum_row", "percentage_point_diff"]], annot=True, ax=ax, fmt=".0f")
        else:
            sns.heatmap(df_cont_new.sort_values(by="sum_row", ascending=False),
            annot=True, ax=ax, fmt=".0f")

        ax.set_ylabel(variable)

    if return_figure:
        return fig_box


def make_violinplots(df, feature_list, target_variable, ncols=5,
                    color=plt.cm.Set1(0),
                    return_fig=False):
    """Creates violin plots when the target variable is numerical and the variables
        in the feature_list are categorical.

    Creates a new figure to show the distributions of the numerical variable with respect
    to the various categories of the categorical variables.

    Parameters
    ----------
    df: DataFrame
        The data frame to analyze.
    feature_list: list
        The list of catgorical (ONLY) columns with respect to which we wish to 
        visualize the target variable.
    target_variable: str
        The target variable (numerical ONLY) whose distribution with respect to each category of
        the categorical variable is visualized.
    ncols: int, optional
        Number of panels to show in each row.
    color: tuple, optional, default: plt.cm.Set1(0)
        Color (RGBA) of the violins plotted.
    return_fig: bool, optional, default: False
        Specifies whether to return a figure object when function is called.

    Returns
    -------
        Figure or None
    """

    nrows = int(np.ceil(len(feature_list)/ncols))
    fig_box, axes_list = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*7, nrows*6))
    if nrows * ncols > 1:
        axes_list = axes_list.ravel()

    fig_box.subplots_adjust(wspace=0.4, hspace=0.2)
    for var_num, variable in enumerate(feature_list):
        if nrows * ncols > 1: 
            ax = axes_list[var_num]
        else:
            ax = axes_list
        sns.violinplot(x=target_variable, y=variable, data=df, color=color, scale="count", 
                        inner="quartile", ax=ax)
        ax.set_xlabel(target_variable)
        ax.set_ylabel(variable)

    if return_fig:
        return fig_box


def make_pairwise_stripplots(df, feature_list_1, feature_list_2, hue_var,
                            title="", 
                            row_feature="first", 
                            return_fig=False):
    """Makes catplots with respect to each feature colored by hue_var. 
    
    Plots the distribution of a categorical variable (hue_var) along a numerical variable and a 
    categorical variable (feature_list_1 and feature_list_2).

    Parameters
    ----------
    df: DataFrame
        The data frame to analyze.
    feature_list_1: list
        The list of catgorical / numerical (but not both) columns.
    feature_list_1: list
        The list of catgorical / numerical (but not both) columns. If feature_list_1 contains
        numerical variables, this should contain categorical variables and vice versa.
    hue_var: str
        The categorical variable which colors the stripplot.
    title: str, optional, default: ""
        Figure title
    row_feature: str, {"first", "second"}, default: "first"
        Which feature list (1 or 2), we want to show along the row or column of the
        figure grid. 
    return_fig: bool, optional, default: False
        Specifies whether to return a figure object when function is called.

    Returns
    -------
        Figure or None
    """

    if row_feature == "first":
        nrows = len(feature_list_1); ncols = len(feature_list_2)
    else:
        nrows = len(feature_list_2); ncols = len(feature_list_1)

    fig, axes_list = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*7, nrows*6))
    fig.subplots_adjust(wspace=0.4, hspace=0.2)
    for feature_num_1, feature_1 in enumerate(feature_list_1):
        for feature_num_2, feature_2 in enumerate(feature_list_2):
            if row_feature == "first":
                ax = axes_list[feature_num_1, feature_num_2]
            else:
                ax = axes_list[feature_num_2, feature_num_1]

            sns.stripplot(data=df, x=feature_1, y=feature_2,
                hue=hue_var, ax=ax)
            ax.set_xlabel(feature_1)
            ax.set_ylabel(feature_2)

            #Legend plotted, by default, on the last panel.
            if (feature_num_1 == (len(feature_list_1)-1)) & \
                (feature_num_2 == (len(feature_list_2)-1) ): #last feature
                ax.legend(loc="upper center")
            else:
                ax.get_legend().set_visible(False)
    fig.suptitle(title)

    if return_fig:
        return fig

