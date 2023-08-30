import streamlit as st
import matplotlib.pyplot as plt


def page_summary_body():

    # Image sources: https://commons.wikimedia.org/wiki/File:Ames_Iowa_Main_Street_%28bannerportada_esvoy%29.jpg
    # https://www.niche.com/colleges/iowa-state-university/
    image_main = plt.imread(f"media/amesiowa.jpg")
    image_isu = plt.imread(f"media/iowasu.jpg")

    st.image(image_main, caption='Mainstreet - Ames, Iowa.')

    st.write("### Project Summary")

    # text based on README file - "Dataset Content" section
    st.info(
        f"**Project Purpose and Motivation**\n\n"
        f" The general pupose of this project is to provide a tool that allows"
        f" a client to predict the potential sale price of a property in"
        f" Ames, Iowa, by providing detailed and typical information on the"
        f" real estate in question.\n\n"
        f" Specifically, a client has requested this app in order"
        f" to estimate the sale price for several inherited properties in "
        f" Ames, Iowa. The client has provided a publically available data set"
        f" which is used to train the machine learning model and "
        f" predict local real estate sale prices. \n \n"
        f"**Project Terminology**\n"
        f"* A **client** is a person who uses this service or product.\n"
        f"* The **sale price** is the estimated value of a home as it"
        f" might be realized in a typical and unencumbered real estate"
        f" transaction. \n"
        f"* The home, the value of which is being estimated, may be refered to"
        f" as **property, real estate, house, or home**. \n"
        f"* The **features** or **attributes** of a home are characteristics"
        f" used to describe the home. \n \n"
        f"**Project Dataset**\n"
        f"* The data set can be accessed at "
        f"[Kaggle](https://www.kaggle.com/datasets/codeinstitute/housing-prices-data)"
        f" where it is hosted by Code Institute.\n"
        f"* The dataset represents a record of approx. 1500 real estate "
        f" sales in Ames, Iowa. Each record contains 23 features indicating"
        f" the house profile, such as Floor Area, Basement, Garage, "
        f" Kitchen, Lot,"
        f" Porch, Wood Deck, and Year Built. It also contains the sale price."
        f" The features are extensive, so please visit the site for more"
        f" information.")

    # copied from README file - "Business Requirements" section
    st.success(
        f"**Business Requirements**\n\n"
        f"The project has 3 business requirements:\n"
        f"* 1. The client is interested in understanding the correlation "
        f" between a properties attributes/features and the sale price."
        f" Therefore, the client expects data visualization of the correlated"
        f" variables against the sale prices for illustration. \n"
        f"* 2. The client is interested in predicting the potential sale "
        f" prices"
        f" for properties in Ames, Iowa, and specifically, she wants to"
        f" determine a potential value for the properties she inherited. \n"
        f"* 3. The client would like to access the outcomes easily using"
        f" an online application."
    )

    # Link to README file, so the users can have access to full
    # project documentation
    st.write(
        f"* For additional information on this project please consult the "
        f"[README](https://github.com/URiem/heritage-housing-PP5/tree/main)"
        f" file for this project hosted on GitHub.\n"
        f"* The projecct was developed by Ulrike Riemenschneider. To find out"
        f" more information about the developer, please visit "
        f" [LinkedIn](https://www.linkedin.com/in/ulrikeseekingopportunities/)"
        f" or [GitHub](https://github.com/URiem). \n"
        f"* For additional information on Ames, Iowa, home of Iowa State"
        f" University and the Iowa State Cyclones, visit "
        f"[Wikipedia](https://en.wikipedia.org/wiki/Ames,_Iowa).")

    st.image(image_isu, caption='Iowa State Univesity - Ames, Iowa.')
