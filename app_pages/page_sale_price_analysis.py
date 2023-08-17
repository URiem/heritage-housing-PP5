import plotly.express as px
import numpy as np
from feature_engine.discretisation import ArbitraryDiscretiser
import streamlit as st
from src.data_management import load_house_prices_data

import matplotlib.pyplot as plt
import seaborn as sns
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

    # Individual plots per variable
    if st.checkbox("Pearson Correlation"):
        st.write("Pearson Correlation Plot")
        #   churn_level_per_variable(df_eda)

    if st.checkbox("Spearman Correlation"):
        st.write("Spearman correlation Plot")
        #   churn_level_per_variable(df_eda)

    if st.checkbox("Predictive Power Score"):
        st.write("PPS Plot")
        #   churn_level_per_variable(df_eda)

        # Parallel plot
    if st.checkbox("Correlation Plots"):
        st.write("Several correlation Plots")
        # parallel_plot_churn(df_eda)

        # function created using "02 - Churned Customer Study" notebook code - "Variables Distribution by Churn" section
        # def churn_level_per_variable(df_eda):
        #     target_var = 'Churn'

        #     for col in df_eda.drop([target_var], axis=1).columns.to_list():
        #         if df_eda[col].dtype == 'object':
        #             plot_categorical(df_eda, col, target_var)
        #         else:
        #             plot_numerical(df_eda, col, target_var)

        # code copied from "02 - Churned Customer Study" notebook - "Variables Distribution by Churn" section
        # def plot_categorical(df, col, target_var):
        #     fig, axes = plt.subplots(figsize=(12, 5))
        #     sns.countplot(data=df, x=col, hue=target_var,
        #                   order=df[col].value_counts().index)
        #     plt.xticks(rotation=90)
        #     plt.title(f"{col}", fontsize=20, y=1.05)
        #     st.pyplot(fig)  # st.pyplot() renders image, in notebook is plt.show()

        # code copied from "02 - Churned Customer Study" notebook - "Variables Distribution by Churn" section
        # def plot_numerical(df, col, target_var):
        #     fig, axes = plt.subplots(figsize=(8, 5))
        #     sns.histplot(data=df, x=col, hue=target_var, kde=True, element="step")
        #     plt.title(f"{col}", fontsize=20, y=1.05)
        #     st.pyplot(fig)  # st.pyplot() renders image, in notebook is plt.show()

        # function created using "02 - Churned Customer Study" notebook code - Parallel Plot section
        # def parallel_plot_churn(df_eda):

        #     # hard coded from "disc.binner_dict_['tenure']"" result,
        #     tenure_map = [-np.Inf, 6, 12, 18, 24, np.Inf]
        #     # found at "02 - Churned Customer Study" notebook
        #     # under "Parallel Plot" section
        #     disc = ArbitraryDiscretiser(binning_dict={'tenure': tenure_map})
        #     df_parallel = disc.fit_transform(df_eda)

        #     n_classes = len(tenure_map) - 1
        #     classes_ranges = disc.binner_dict_['tenure'][1:-1]
        #     LabelsMap = {}
        #     for n in range(0, n_classes):
        #         if n == 0:
        #             LabelsMap[n] = f"<{classes_ranges[0]}"
        #         elif n == n_classes-1:
        #             LabelsMap[n] = f"+{classes_ranges[-1]}"
        #         else:
        #             LabelsMap[n] = f"{classes_ranges[n-1]} to {classes_ranges[n]}"

        #     df_parallel['tenure'] = df_parallel['tenure'].replace(LabelsMap)
        #     fig = px.parallel_categories(
        #         df_parallel, color="Churn", width=750, height=500)
        #     # we use st.plotly_chart() to render, in notebook is fig.show()
        #     st.plotly_chart(fig)
