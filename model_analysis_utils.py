"""
Some common util functions for analyzing model fit.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('mypresentation.mplstyle')


def plot_evaluation_curves(gs, grid_params):
    """Plots evaluation curves of model performance with respect to hyperparameters.

    Received from Kedar Dabhadkar. 

    Plots the mean and standard deviations of the the accuracy measures so
    that we understand which ones are the best.

    Parameters
    ----------
    gs: sklearn.model_selection._search.GridSearchCV
        Cross-validation object obtained by performing cross-validation on the model. 
    grid_params: dict
        Specifies the hyperparameters which have been tuned via sklearn GridSearchCV. 

    Returns
    -------
        None
    """

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
