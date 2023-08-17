import streamlit as st


def page_project_hypothesis_body():

    st.write("### Project Hypothesis and Validation")

    # conclusions taken from "02 - Churned Customer Study" notebook
    st.success(
        f"* 1. We suspect that a property's sale price correlates strongly "
        f" with a subsection of the extensive features in the data set. \n\n"
        f"  * The correlation study of the data set supports that. \n\n"
        f"* 2. We hypothesise that the correlation is strongest with"
        f" common features of a home, such as square footage, overall condition"
        f" and overall quality. \n\n"
        f"  * Did the study confirm this?\n\n"
        f"* 3. We hypothesis that we are able to predict a sale price with an "
        f" R2 value of 0.8.\n\n"
        f"  * Did the study confirm this?"
    )
