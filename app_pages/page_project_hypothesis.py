import streamlit as st


def page_project_hypothesis_body():

    st.write("### Project Hypothesis and Validation")

    # conclusions taken from "02 - Churned Customer Study" notebook
    st.success(
        f"* We hypothesize that a property's sale price correlates "
        f" strongly "
        f" with a subsection of the extensive features in the dataset. \n\n"
        f"  * The correlation study of the dataset supports that. \n\n"
        f"* We hypothesize that the correlation is strongest with"
        f" common features of a home, such as total square footage, "
        f" overall condition and overall quality. \n\n"
        f"  * The correlation study confirmed this and showed that the "
        f" sale price correlates most strongly with Overall Quality "
        f" (OverallQual), Groundlevel Living area (GrLivArea), Garage "
        f" Area (GarageArea), Total Basement Area (TotalBsmtSF), "
        f" Year Built (YearBuilt), and 1st Floor squarefootage "
        f" (1stFlrSF). \n\n"
        f"* We hypothesize that we are able to predict a sale price with an "
        f" R2 value of at least 0.8.\n\n"
        f"  * The R2 analysis on the train and test sets confirms this."
    )
