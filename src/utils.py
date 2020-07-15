from typing import List

import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

from sklearn.model_selection import learning_curve

from IPython.core.display import display


def plot_cat(df: pd.DataFrame, col: str, x_order = None):
    vc_abs = df[col].value_counts().to_frame('cnt')
    vc_rel = df[col].value_counts(normalize=True).to_frame('perc')
    vc_agg = pd.concat([vc_abs, vc_rel], axis=1)
    display(vc_agg.style.format("{:,.0f}", subset=['cnt'])
                        .format("{:,.1%}", subset=['perc']))
    if not x_order:
        x_order = df[col].value_counts().index
    display(sns.countplot(df[col],
                          order=x_order))

def choose_random_user(df: pd.DataFrame,
                       k=1) -> pd.DataFrame:
    sample = np.random.choice(df['customer'].unique(),
                              size=k)
    res = df.loc[df['customer'].isin(sample)]
    return res.sort_values(['customer', 'step'])

def describe_segment(df: pd.DataFrame,
                     name: str = '0') -> pd.DataFrame:
    stats = dict()
    stats['cnt_row'] = df.shape[0]
    stats['cnt_cust'] = df['customer'].nunique()
    stats['avg_amount'] = df['amount'].mean()
    stats['med_amount'] = df['amount'].median()
    stats['perc_female'] = (df.query('gender == "F"')
                              .shape[0]
                              /
                              stats['cnt_row'])
    stats['avg_steps'] = (df.groupby('customer')
                            ['step']
                            .nunique()
                            .mean())
    stats['avg_txn'] = stats['cnt_row'] / stats['cnt_cust']
    return pd.DataFrame(stats, index=[name])

def describe_segments(df_list: List[pd.DataFrame],
                      names: List[str]):
    desc_df_list = []
    for i in range(len(df_list)):
        desc_df = describe_segment(df_list[i], names[i])
        desc_df_list.append(desc_df)
    concat_df = pd.concat(desc_df_list, axis=0)
    prefix_cols = lambda col: concat_df.columns.str.startswith(col)
    display(concat_df.style.format("{:,.1%}", subset=prefix_cols('perc'))
                           .format("{:,.0f}", subset=prefix_cols('cnt'))
                           .format("{:,.1f}", subset=prefix_cols('avg'))
                           .format("{:,.1f}", subset=prefix_cols('med'))
                           .background_gradient(axis=0, vmin=0))

def crosstab(df: pd.DataFrame, feature: str, label: str):
    ct_df = pd.crosstab(index=df[feature], columns=df[label],
                        values=df['customer'],
                        aggfunc='nunique', normalize='index')
    display(ct_df.style.format("{:,.1%}")
                        .background_gradient(axis=0, vmin=0))

from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5),
                        scoring=None):
    """
    Source: https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))

    scoring: str or callable, default=None
        A str (see model evaluation documentation) or a scorer callable object 
        / function with signature scorer(estimator, X, y).
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].set_ylim(0.5, 1.0)
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return train_sizes, train_scores, test_scores
