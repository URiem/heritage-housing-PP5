import plotly.express as px
import numpy as np
from feature_engine.discretisation import ArbitraryDiscretiser
import streamlit as st
from src.data_management import load_house_prices_data

import matplotlib.pyplot as plt
import seaborn as sns
import ppscore as pps
sns.set_style("whitegrid")


def page_sale_price_analysis_body():

    # load data
    df = load_house_prices_data()

    # copied from *** notebook
    vars_to_study = ['1stFlrSF', 'TotalBsmtSF',
                     'OverallCond', 'OverallQual', 'Lot Area']

    st.write("### Property Sale Price Analysis")
    st.info(
        f"* The client is interested in understanding the correlation "
        f" between a properties attributes/features and the sale price."
        f" Therefore, the client expects data visualization of the correlated"
        f" variables against the sale prices for illustration. \n"
    )

    # inspect data
    if st.checkbox("Inspect Sale Price Data"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns, "
            f"find below the first 10 rows.")

        st.write(df.head(10))

    st.write("---")

    # Correlation Study Summary
    st.write(
        f"* A correlation study was conducted in the notebook to better understand how "
        f"the variables are correlated to Sale Price. \n"
        f"The most correlated variable are: **{vars_to_study}**"
    )

    # Text based on "02 - ***" notebook - "Conclusions and Next steps" section
    st.info(
        f"The correlation indications and plots below interpretation converge. "
        f"It is indicated that Sale Price correlates most strongly with "
        f"the following variables in order of the strength of the "
        f"correlation: \n"
        f"* Variable 1 \n"
        f"* Variable 2 \n"
        f"* Variable 3 \n"
        f"* Variable 4 \n"
    )

    # Code copied from "02 - ***" notebook - "EDA on selected variables" section
    # df_eda = df.filter(vars_to_study + ['Churn'])

    # Correlation plots adapted from the Data Cleaning Notebook
    if st.checkbox("Pearson Correlation"):
        calc_display_pearson_corr(df)

    if st.checkbox("Spearman Correlation"):
        calc_display_spearman_corr(df)

    if st.checkbox("Predictive Power Score"):
        calc_display_pps_matrix(df)

    if st.checkbox("Variable Correlation Plots"):
        st.write("Correlation Plots for the variable with the strongest correlations")


def calc_display_spearman_corr(df):
    df_corr_spearman = df.corr(method="spearman")

    st.write("*** Heatmap: Spearman Correlation ***")
    st.write("It evaluates monotonic relationship \n")
    heatmap_corr(df=df_corr_spearman, threshold=0.3,
                 figsize=(12, 10), font_annot=10)


def calc_display_pearson_corr(df):
    df_corr_pearson = df.corr(method="pearson")

    st.write("*** Heatmap: Pearson Correlation ***")
    st.write(
        "It evaluates the linear relationship between two continuous variables \n")
    heatmap_corr(df=df_corr_pearson, threshold=0.3,
                 figsize=(12, 10), font_annot=10)


def calc_display_pps_matrix(df):
    pps_matrix_raw = pps.matrix(df)
    pps_matrix = pps_matrix_raw.filter(['x', 'y', 'ppscore']).pivot(
        columns='x', index='y', values='ppscore')

    # pps_score_stats = pps_matrix_raw.query(
    #     "ppscore < 1").filter(['ppscore']).describe().T
    # st.write(pps_score_stats.round(3))

    st.write("*** Heatmap: Power Predictive Score (PPS) ***")
    st.write(f"PPS detects linear or non-linear relationships between two columns.\n"
             f"The score ranges from 0 (no predictive power) to 1 (perfect predictive power) \n")
    heatmap_pps(df=pps_matrix, threshold=0.2, figsize=(12, 10), font_annot=10)


def heatmap_corr(df, threshold, figsize=(20, 12), font_annot=8):
    if len(df.columns) > 1:
        mask = np.zeros_like(df, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        mask[abs(df) < threshold] = True

        fig, axes = plt.subplots(figsize=figsize)
        sns.heatmap(df, annot=True, xticklabels=True, yticklabels=True,
                    mask=mask, cmap='viridis', annot_kws={"size": font_annot}, ax=axes,
                    linewidth=0.5
                    )
        axes.set_yticklabels(df.columns, rotation=0)
        plt.ylim(len(df.columns), 0)
        st.pyplot(fig)


def heatmap_pps(df, threshold, figsize=(20, 12), font_annot=8):
    if len(df.columns) > 1:
        mask = np.zeros_like(df, dtype=bool)
        mask[abs(df) < threshold] = True
        fig, ax = plt.subplots(figsize=figsize)
        ax = sns.heatmap(df, annot=True, xticklabels=True, yticklabels=True,
                         mask=mask, cmap='rocket_r', annot_kws={"size": font_annot},
                         linewidth=0.05, linecolor='grey')
        plt.ylim(len(df.columns), 0)
        st.pyplot(fig)
