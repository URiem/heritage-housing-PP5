import streamlit as st


def page_project_hypothesis_body():

    st.write("### Project Hypothesis and Validation")

    # conclusions taken from "02 - Churned Customer Study" notebook
    st.success(
        f"* 1. We hypothesise that a property's sale price correlates "
        f" strongly "
        f" with a subsection of the extensive features in the data set. \n\n"
        f"  * The correlation study of the data set supports that. \n\n"
        f"* 2. We hypothesise that the correlation is strongest with"
        f" common features of a home, such as square feet, overall condition"
        f" and overall quality. \n\n"
        f"  * The correlation study confirmed this and showed that the "
        f" sale price correlates most strongly with Overall Quality "
        f" (OverallQual), Groundlevel Living area (GrLivArea), Garage "
        f" Area (GarageArea), and Total Basement Area (TotalBsmtSF). \n\n"
        f"* 3. We hypothesis that we are able to predict a sale price with an "
        f" R2 value of at least 0.8.\n\n"
        f"  * The R2 analysis on the train and test sets confirms this."
    )
