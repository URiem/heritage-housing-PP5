# Heritage Housing Issues

**Data Analysis and Predictive Modelling Study**

![Mockup image](#)

**Live Site:**

[Live webpage](#)

**Link to Repository:**

[Repository](https://github.com/URiem/heritage-housing-PP5)

**Developed by: Ulrike Riemenschneider**

## Table of Content

- [Heritage Housing Issues](#heritage-housing-issues)
  - [Table of Content](#table-of-content)
  - [Introduction](#introduction)
  - [Business Requirements](#business-requirements)
  - [Dataset Content](#dataset-content)
  - [Hypothesis and how to validate?](#hypothesis-and-how-to-validate)
  - [Mapping the business requirements to the Data Visualisations and ML tasks](#mapping-the-business-requirements-to-the-data-visualisations-and-ml-tasks)
  - [ML Business Case](#ml-business-case)
    - [Predict Sale Price](#predict-sale-price)
  - [Dashboard Design](#dashboard-design)
    - [Page 1: Project Summary](#page-1-project-summary)
    - [Page 2: Analysis of Sale Price](#page-2-analysis-of-sale-price)
    - [Page 3: Sale Price Prediction](#page-3-sale-price-prediction)
    - [Page 4: Hypothesis and Validation](#page-4-hypothesis-and-validation)
    - [Page 5: Machine Learning Model](#page-5-machine-learning-model)
  - [Unfixed Bugs](#unfixed-bugs)
  - [Deployment](#deployment)
    - [Heroku](#heroku)
  - [Main Data Analysis and Machine Learning Libraries](#main-data-analysis-and-machine-learning-libraries)
  - [Credits](#credits)
    - [Content](#content)
    - [Media](#media)
  - [Acknowledgements (optional)](#acknowledgements-optional)

## Introduction

This Machine Learning Project was developed as the fifth portfolio project during the Code Insititute's Diploma in Full Stack Development. It covers the Predictive Analytics specialization.

The Maching Learning and Data Analysis toolkit is applied to a real estate data set and developed with the specific purpose to allow a user to predict the sale value of a property based on certain features of the home. It also allows the user to see how certain features of a home correlate with the sale price of the home.

My personal motivation for choosing this project, is that I myself inherited two properties in a small town in the US in 2011 and subsequently became licensed and worked as a real estate agent for several years in order to gain experience in the real estate market and ultimately sell the properties I had inherited.

## Business Requirements

A client who has received an inheritance from a deceased great-grandfather located in Ames, Iowa, has requested help in maximising the sales price for the inherited properties.

The client has an excellent understanding of property prices in her own state and residential area, but she fears that basing her estimates for property worth on her current knowledge might lead to inaccurate appraisals. What makes a house desirable and valuable where she comes from might not be the same in Ames, Iowa. She found a public dataset with house prices for Ames, Iowa, and will provide you with that.

The client is interested in the following outcomes:

1. Discovering how the house attributes correlate with the sale price. Therefore, the client expects data visualisations of the correlated variables against the sale price to show that.
  
2. Predicting the house sale price from her four inherited houses and any other house in Ames, Iowa.

3. Delivery of the final product in the form of a deployed app that is easily accessible online and userfriendly.  

## Dataset Content

- The dataset is sourced from [Kaggle](https://www.kaggle.com/codeinstitute/housing-prices-data). The business requirements are based on a fictitious, but realisting, user story where predictive analytics can be applied in a real world scenario.

- The dataset has almost 1.5 thousand rows and represents housing records from Ames, Iowa, indicating house profile (Floor Area, Basement, Garage, Kitchen, Lot, Porch, Wood Deck, Year Built) and its respective sale price for houses built between 1872 and 2010.

|Variable|Meaning|Units|
|:----|:----|:----|
|1stFlrSF|First Floor square feet|334 - 4692|
|2ndFlrSF|Second-floor square feet|0 - 2065|
|BedroomAbvGr|Bedrooms above grade (does NOT include basement bedrooms)|0 - 8|
|BsmtExposure|Refers to walkout or garden level walls|Gd: Good Exposure; Av: Average Exposure; Mn: Minimum Exposure; No: No Exposure; None: No Basement|
|BsmtFinType1|Rating of basement finished area|GLQ: Good Living Quarters; ALQ: Average Living Quarters; BLQ: Below Average Living Quarters; Rec: Average Rec Room; LwQ: Low Quality; Unf: Unfinshed; None: No Basement|
|BsmtFinSF1|Type 1 finished square feet|0 - 5644|
|BsmtUnfSF|Unfinished square feet of basement area|0 - 2336|
|TotalBsmtSF|Total square feet of basement area|0 - 6110|
|GarageArea|Size of garage in square feet|0 - 1418|
|GarageFinish|Interior finish of the garage|Fin: Finished; RFn: Rough Finished; Unf: Unfinished; None: No Garage|
|GarageYrBlt|Year garage was built|1900 - 2010|
|GrLivArea|Above grade (ground) living area square feet|334 - 5642|
|KitchenQual|Kitchen quality|Ex: Excellent; Gd: Good; TA: Typical/Average; Fa: Fair; Po: Poor|
|LotArea| Lot size in square feet|1300 - 215245|
|LotFrontage| Linear feet of street connected to property|21 - 313|
|MasVnrArea|Masonry veneer area in square feet|0 - 1600|
|EnclosedPorch|Enclosed porch area in square feet|0 - 286|
|OpenPorchSF|Open porch area in square feet|0 - 547|
|OverallCond|Rates the overall condition of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|OverallQual|Rates the overall material and finish of the house|10: Very Excellent; 9: Excellent; 8: Very Good; 7: Good; 6: Above Average; 5: Average; 4: Below Average; 3: Fair; 2: Poor; 1: Very Poor|
|WoodDeckSF|Wood deck area in square feet|0 - 736|
|YearBuilt|Original construction date|1872 - 2010|
|YearRemodAdd|Remodel date (same as construction date if no remodelling or additions)|1950 - 2010|
|SalePrice|Sale Price|34900 - 755000|

## Hypothesis and how to validate?

- List here your project hypothesis(es) and how you envision validating it (them).

## Mapping the business requirements to the Data Visualisations and ML tasks

- Business Requirement 1: Data Visualization and Correlation Study.
  - We will inspect the data contained in the data set as it relates to the property sale prices in Ames, Iowa.
  - We will conduct a correlation study (Pearson and Spearman) to understand better how the variables are correlated to the sale price.
  - We will plot the most important and relevant data against the sale price to visualize the insights.

- Business Requirement 2: Regression and Data Analysis
  - We want to predict the sale price of homes in Ames, Iowa. For this purpose we will build a regression model with the sale price as the target value.

- Business Requirement 3: Online App and Deployement
  - We will build an app using streamlit that displays all the desired data analysis and visualization as well as a feature that will allow the client to predict the sale prices for her and any other property in Ames, Iowa.
  - We will deploy the app using Heroku.

## ML Business Case

### Predict Sale Price

- We want an ML model to predict sale price, in dollars, for a home in Ames, Iowa. The target variable is a continuous number. We firstly consider a regression model, which is supervised and uni-dimensional.
- Our ideal outcome is to provide a client with the ability to reliably predict the sale price of any home in Ames, Iowa, and more specifically the inherited properties the client is particularly concerned with.
- The model success metrics are:
  - At least 0.75 for R2 score, on train and test set.
  - The model is considered a failure if: after 12 months of usage, the model predictions are 50% off more than 30% of the time, and/or the R2 score is less than 0.75.
- The output is defined as a continuous value of sale price in dollars. Private parties/home owners/clients can access the app online and input data for their homes. The app can also be useful for real estate agents who want to give a quick estimate of saleprice to aprospective client, the can input the data on the fly while in live commonication with a prospective client.
- The training data come from a public data set, which contains over 1000 property sales records. It contains one target features: sale price, and all other variables (23 of them) are considered features.

## Dashboard Design

The project will be built using a Streamlit dashboard. The completed dashboard will satisfy the third Business Requrirement. It will contain the following pages:

### Page 1: Project Summary

This page will incude

- Project terms and jargon
- Description of the data set
- Statement of business requirements

### Page 2: Analysis of Sale Price

This page will include checkboxes so the client has the ability to display the following visual guides to the data features:

- Correlation between various features and the sale price.
- Spearman and Pearson correlations.
- Predictive Power Score analysis.
This will satisfy the first Business Requrirement.

### Page 3: Sale Price Prediction

This page will include

- Input feature of property attributes to produce a prediction on the sale price
- Display of the predicted sale price
- Predict the sale prices of the clients specific data in relation to her inherited properties.
This will satisfy the second Business Requirement.

### Page 4: Hypothesis and Validation

This page will include

- A list of the project's hypothesis and how they were validated

### Page 5: Machine Learning Model

This page will include

- Information on the ML pipeline used to train the model
- Demonstration of feature importance
- Review of the pipeline performance
- Considerations and conclusions after pipeline training

## Unfixed Bugs

- You will need to mention unfixed bugs and why they were not fixed. This section should include shortcomings of the frameworks or technologies used. Although time can be a big variable to consider, paucity of time and difficulty understanding implementation is not valid reason to leave bugs unfixed.

## Deployment

### Heroku

- The App live link is: https://YOUR_APP_NAME.herokuapp.com/
- Set the runtime.txt Python version to a [Heroku-20](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
- The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.

## Main Data Analysis and Machine Learning Libraries

- Here you should list the libraries you used in the project and provide example(s) of how you used these libraries.

## Credits

- In this section, you need to reference where you got your content, media and extra help from. It is common practice to use code from other repositories and tutorials, however, it is important to be very specific about these sources to avoid plagiarism.
- You can break the credits section up into Content and Media, depending on what you have included in your project.

### Content

- The text for the Home page was taken from Wikipedia Article A
- Instructions on how to implement form validation on the Sign-Up page was taken from [Specific YouTube Tutorial](https://www.youtube.com/)
- The icons in the footer were taken from [Font Awesome](https://fontawesome.com/)

### Media

- The photos used on the home and sign-up page are from This Open Source site
- The images used for the gallery page were taken from this other open-source site

## Acknowledgements (optional)

- In case you would like to thank the people that provided support through this project.

1. In the terminal type <code>jupyter notebook --NotebookApp.token='' --NotebookApp.password=''</code> to start the jupyter server.
