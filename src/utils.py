from typing import List

import pandas as pd
import numpy as np
import seaborn as sns

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
