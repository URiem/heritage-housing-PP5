import plotly.express as px
import numpy as np
import streamlit as st
from src.data_management import load_house_prices_data
import matplotlib.pyplot as plt
import seaborn as sns
import ppscore as pps
sns.set_style("whitegrid")


def page_sale_price_analysis_body():

    # load data
    df = load_house_prices_data()
    # The variable most strongly correlated with Sale Price/target
    vars_to_study = ['OverallQual', 'GrLivArea', 'GarageArea', 'TotalBsmtSF']

    st.write("### Property Sale Price Analysis")
    st.success(
        f"* The client is interested in understanding the correlation "
        f" between a property's attributes/features and the sale price."
        f" Therefore, the client expects data visualization of the correlated"
        f" variables against the sale prices for illustration "
        f" (Business Requirement 1), \n"
    )

    # inspect data
    if st.checkbox("Inspect Sale Price Dataset"):
        st.write(
            f"* The dataset has {df.shape[0]} rows and {df.shape[1]} columns, "
            f"find below the first 10 rows.")
        st.write(df.head(10))
        st.write(
            f"**Information on Categorical Features**\n\n"
            f"* Basement Exposure: Gd - Good Exposure, Av - Average Exposure, "
            f" Mn - Minimum Exposure, No: No Exposure, None - No Basement.\n\n"
            f"* Basement Finish Type: GLQ - Good Living Quarters, ALQ - Average"
            f" Living Quarters, BLQ - Below Average Living Quarters, REC - "
            f" Average Rec Room, LwQ - Low  Quality, Unf - Unfinished, None - "
            f" No Basement.\n\n"
            f"* Garage Finish: Fin - Finished, RFn: Rough Finish, Unf - Unfinished"
            f" None - No Garage.\n\n"
            f"* Kitchen Quality: Ex - Excellent, Gd - Good, TA - Typical/Average, "
            f" Fa: Fair, Po: Poor.\n\n"
            f"* Overall Condition: 1 - Very Poor up to 10 - Very Excellent.\n\n"
            f"* Overall Quality: 1 - Very Poor up to 10 - Very Excellent.\n\n"
        )

    st.write("---")

    st.write("### Correlation Study")
    # Correlation Study Summary
    st.write(
        f"A correlation study was conducted to better understand how "
        f"the variables are correlated to Sale Price. \n"
        f" Below, the results from the Pearson and Spearman correlations"
        f" are displayed in a heatmap plot. These figures show that"
        f" the most correlated variable are: **{vars_to_study}**. \n"
        f" Therefore, we also display scatterplots illustrating  the "
        f" correlation of each of these variables with the Sale Price."
    )

    st.info(
        f"*** Correlation Scatterplots *** \n\n"
        f"The correlation indicators below confirm that "
        f" Sale Price correlates most strongly with "
        f"the following variables in order of the strength of the "
        f"correlation: \n"
        f"* Sale Price tends to increase as Overall Quality "
        f" (OverallQual) goes up. \n"
        f"* Sale Price tends to increase as Groundlevel Living Area "
        f" (GrLivArea) increases. \n"
        f"* Sale Price tends to increase with increasing Garage Area "
        f" (GarageArea). \n"
        f"* Sale Price tends to increase with an increase in Total "
        f" Basement Area (TotalBsmtSF). \n"
    )

    # Correlation plots adapted from the Data Cleaning Notebook
    if st.checkbox("Correlation Plots of Variables vs Sale Price"):
        correlation_to_sale_price_hist(df, vars_to_study)
        # correlation_to_sale_price_scat(df, vars_to_study)

    st.info(
        f"*** Heatmap: Pearson Correlation *** \n\n"
        f"The Pearson Correlation evaluates the linear relationship between "
        f" two continuous variables, that is how closely the correlation"
        f" between the variable can be represented by a straight line. \n"
        f" The last line of the heatmap shows the variables on the x-axis"
        f" which have a linear correlation with the Sale Price "
        f" of more than 0.6. ")

    if st.checkbox("Pearson Correlation"):
        calc_display_pearson_corr(df)

    st.info(
        f"*** Heatmap: Spearman Correlation ***  \n\n"
        f"The Spearman correlation evaluates monotonic relationship, "
        f"that is a relationship "
        f"where the variables behave similarly but not necessarily linearly.\n"
        f" As with the Pearson heatmap, the last line shows the variables"
        f" on the x-axis, that have a correlation of 0.6 or more with"
        f" with the Sale Price.")

    if st.checkbox("Spearman Correlation"):
        calc_display_spearman_corr(df)

    st.info(
        f"*** Heatmap: Predictive Power Score (PPS) ***  \n\n"
        f"The PPS detects linear or non-linear relationships "
        f"between two variables.\n"
        f"The score ranges from 0 (no predictive power) to 1 "
        f"(perfect predictive power). \n"
        f" To use the plot, find the row on the y-axis labeled 'SalePrice' "
        f" then follow along the row and see the variables, labeled on the "
        f" x-axis, with a pps of more"
        f" than 0.2 expressed on the plot. Overall Quality (OverallQual)"
        f" has the highest predictive power for the Sale Price target.")

    if st.checkbox("Predictive Power Score"):
        calc_display_pps_matrix(df)


def correlation_to_sale_price_hist(df, vars_to_study):
    """ Display correlation plot between variables and sale price """
    target_var = 'SalePrice'
    for col in vars_to_study:
        fig, axes = plt.subplots(figsize=(8, 5))
        axes = sns.histplot(data=df, x=col, y=target_var)
        plt.title(f"{col}", fontsize=20, y=1.05)
        st.pyplot(fig)
        st.write("\n\n")


def correlation_to_sale_price_scat(df, vars_to_study):
    """  scatterplots of variables vs SalePrice """
    target_var = 'SalePrice'
    for col in vars_to_study:
        fig, axes = plt.subplots(figsize=(12, 5))
        axes = sns.scatterplot(data=df, x=col, y=target_var)
        plt.xticks(rotation=90)
        plt.title(f"{col}", fontsize=20, y=1.05)
        st.pyplot(fig)
        print("\n\n")


def calc_display_pearson_corr(df):
    """ Calcuate and display Pearson Correlation """
    df_corr_pearson = df.corr(method="pearson")
    heatmap_corr(df=df_corr_pearson, threshold=0.6,
                 figsize=(12, 10), font_annot=10)


def calc_display_spearman_corr(df):
    """ Calcuate and display Spearman Correlation """
    df_corr_spearman = df.corr(method="spearman")
    heatmap_corr(df=df_corr_spearman, threshold=0.6,
                 figsize=(12, 10), font_annot=10)


def calc_display_pps_matrix(df):
    """ Calcuate and display Predictive Power Score """
    pps_matrix_raw = pps.matrix(df)
    pps_matrix = pps_matrix_raw.filter(['x', 'y', 'ppscore']).pivot(
        columns='x', index='y', values='ppscore')

    # pps_score_stats = pps_matrix_raw.query(
    #     "ppscore < 1").filter(['ppscore']).describe().T
    # st.write(pps_score_stats.round(3))
    heatmap_pps(df=pps_matrix, threshold=0.15, figsize=(12, 10), font_annot=10)


def heatmap_corr(df, threshold, figsize=(20, 12), font_annot=8):
    """ Heatmap for correlations from CI template"""
    if len(df.columns) > 1:
        mask = np.zeros_like(df, dtype=bool)
        mask[np.triu_indices_from(mask)] = True
        mask[abs(df) < threshold] = True
        fig, axes = plt.subplots(figsize=figsize)
        axes = sns.heatmap(df, annot=True, xticklabels=True, yticklabels=True,
                           mask=mask, cmap='viridis',
                           annot_kws={"size": font_annot},
                           ax=axes, linewidth=0.5
                           )
        axes.set_yticklabels(df.columns, rotation=0)
        plt.ylim(len(df.columns), 0)
        st.pyplot(fig)


def heatmap_pps(df, threshold, figsize=(20, 12), font_annot=8):
    """ Heatmap for predictive power score from CI template"""
    if len(df.columns) > 1:
        mask = np.zeros_like(df, dtype=bool)
        mask[abs(df) < threshold] = True
        fig, axes = plt.subplots(figsize=figsize)
        axes = sns.heatmap(df, annot=True, xticklabels=True, yticklabels=True,
                           mask=mask, cmap='rocket_r',
                           annot_kws={"size": font_annot},
                           linewidth=0.05, linecolor='grey')
        plt.ylim(len(df.columns), 0)
        st.pyplot(fig)
