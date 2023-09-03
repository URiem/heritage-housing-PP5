import plotly.express as px
import numpy as np
import streamlit as st
from src.data_management import load_house_prices_data
import matplotlib.pyplot as plt
import seaborn as sns
import ppscore as pps
sns.set_style("whitegrid")


def page_sale_price_analysis_v2_body():

    # load data
    df = load_house_prices_data()
    # The variable most strongly correlated with Sale Price/target
    vars_to_study = ['OverallQual', 'GrLivArea',
                     'GarageArea', 'TotalBsmtSF', 'YearBuilt', '1stFlrSF']

    st.write("### Property Sale Price Analysis V2")
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
        f" are displayed in bar plots. These figures show that"
        f" the most correlated variable are: **{vars_to_study}**. \n"
        f" Therefore, we also display scatterplots illustrating  the "
        f" correlation of each of these variables with the Sale Price."
    )

    st.info(
        f"*** Bar: Pearson Correlation *** \n\n"
        f"The Pearson Correlation evaluates the linear relationship between "
        f" two continuous variables, that is how closely the correlation"
        f" between the variable can be represented by a straight line. \n"
        f" The barplot shows the variables on the x-axis"
        f" which have a linear correlation with the Sale Price "
        f" of more than 0.6. ")

    if st.checkbox("Pearson Correlation"):
        calc_display_pearson_corr(df)

    st.info(
        f"*** Bar: Spearman Correlation ***  \n\n"
        f"The Spearman correlation evaluates monotonic relationship, "
        f"that is a relationship "
        f"where the variables behave similarly but not necessarily linearly.\n"
        f" The barplot shows the variables"
        f" on the x-axis, that have a correlation of 0.6 or more with"
        f" with the Sale Price.")

    if st.checkbox("Spearman Correlation"):
        calc_display_spearman_corr(df)

    st.info(
        f"*** Correlation Scatterplots *** \n\n"
        f"The correlation indicators above confirm that "
        f" Sale Price correlates most strongly with "
        f"the following variables: \n"
        f"* Sale Price tends to increase as Overall Quality "
        f" (OverallQual) goes up. \n"
        f"* Sale Price tends to increase as Groundlevel Living Area "
        f" (GrLivArea) increases. \n"
        f"* Sale Price tends to increase with increasing Garage Area "
        f" (GarageArea). \n"
        f"* Sale Price tends to increase with an increase in Total "
        f" Basement Area (TotalBsmtSF). \n"
        f"* Sale Price tends to increase with an increase in "
        f" Year Built (YearBuilt). \n"
        f"* Sale Price tends to increase with an increase in "
        f" 1st Floor Squarefootage (1stFlrSF). \n\n"
        f"The scatterplots below illustrate the trends of the"
        f"correlations. Each data point is also colored according"
        f"to the Overall Quality of that data point. The"
        f"trend that with increasing overall quality the Sale Price"
        f"increases can be clearly seen on all plots."
    )

    # Correlation plots adapted from the Data Cleaning Notebook
    if st.checkbox("Correlation Plots of Variables vs Sale Price"):
        correlation_to_sale_price_scat(df, vars_to_study)


def correlation_to_sale_price_scat(df, vars_to_study):
    """  scatterplots of variables vs SalePrice """
    target_var = 'SalePrice'
    for col in vars_to_study:
        fig, axes = plt.subplots(figsize=(8, 5))
        axes = sns.scatterplot(data=df, x=col, y=target_var, hue='OverallQual')
        # plt.xticks(rotation=90)
        plt.title(f"{col}", fontsize=20, y=1.05)
        st.pyplot(fig)
        st.write("\n\n")


def correlation_to_sale_price_joint(df, vars_to_study):
    """  Joint plots of variables vs SalePrice """
    target_var = 'SalePrice'
    for col in vars_to_study:
        x, y, hue = col, target_var, 'OverallQual'
        sns.jointplot(data=df, x=x, y=y, kind='hex')
        # sns.jointplot(data=df, x=x, y=y, hue=hue)
        plt.title(f"{col}", fontsize=20, y=1.3, x=-3)
        plt.show()
        print("\n\n")


def calc_display_pearson_corr(df):
    """ Calcuate and display Pearson Correlation """
    corr_pearson = df.corr(method='pearson')['SalePrice'].sort_values(
        key=abs, ascending=False)[1:]
    fig, axes = plt.subplots(figsize=(6, 3))
    axes = plt.bar(x=corr_pearson[:5].index, height=corr_pearson[:5])
    plt.title("Pearson Correlation", fontsize=20, y=1.05)
    st.pyplot(fig)


def calc_display_spearman_corr(df):
    """ Calcuate and display Spearman Correlation """
    corr_spearman = df.corr(method='spearman')['SalePrice'].sort_values(
        key=abs, ascending=False)[1:]
    fig, axes = plt.subplots(figsize=(6, 3))
    axes = plt.bar(x=corr_spearman[:5].index, height=corr_spearman[:5])
    plt.title("Spearman Correlation", fontsize=20, y=1.05)
    st.pyplot(fig)
