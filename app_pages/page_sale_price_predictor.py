import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date
from src.data_management import (
    load_house_prices_data,
    load_pkl_file,
    load_inherited_house_data)
from src.machine_learning.evaluate_regression import regression_performance
from src.machine_learning.predictive_analysis_ui import predict_sale_price


def page_sale_price_predictor_body():

    # load predict sale price files
    vsn = 'v2'
    sale_price_pipe = load_pkl_file(
        f"outputs/ml_pipeline/predict_sale_price/{vsn}/regression_pipeline.pkl")
    sale_price_features = (
        pd.read_csv(
            f"outputs/ml_pipeline/predict_sale_price/{vsn}/X_train.csv")
        .columns
        .to_list()
    )

    st.write("### Sale Price Predictor Interface")
    st.success(
        f"* The client is interested in predicting the potential sale "
        f" prices"
        f" for properties in Ames, Iowa, and specifically, she wants to"
        f" determine a potential value for the properties she inherited "
        f" (Business Requirement 2). \n"
    )
    st.info(
        f"The price prediction will be based on four "
        f" features of the property in question, which the client can input"
        f" using the selections below. These features were identified by"
        f" the machine learning model as the best features to predict Sale "
        f" Price. They are similar to, but may differ slightly from, the "
        f" variables "
        f" identified as most correlated in the initial data analysis. This "
        f" is because the model will carry out more complex analysis on the "
        f" variables behind the scenes and identify the best variables to use"
        f" for the prediction of the Sale Price. \n\n More information on the "
        f" machine learning model and feature importance can be found on the "
        f" **ML: Price Prediction** page. \n\n"
        f"**Information on categorical features used in the prediction**\n\n"
        f"* Overall Quality: 1 - Very Poor up to 10 - Very Excellent.\n\n"
        f"All three numerical features are measured in squarefeet."
    )
    st.write("---")

    # Generate Live Data
    X_live = DrawInputsWidgets()

    # predict on live data
    if st.button("Run Predictive Analysis"):
        predict_sale_price(X_live, sale_price_features, sale_price_pipe)

    st.write("---")

    st.write("### Price prediction for the clients inherited properties:")
    in_df = load_inherited_house_data()
    in_df = in_df.filter(sale_price_features)

    st.write("* Features of Inherited Homes")
    st.write(in_df)

    if st.button("Run Prediction on Inherited Homes"):
        inherited_price_prediction = predict_sale_price(
            in_df, sale_price_features, sale_price_pipe)
        total_value = inherited_price_prediction.sum()
        total_value = float(total_value.round(1))
        total_value = '${:,.2f}'.format(total_value)

        st.write(f"* The total value of the inherited homes is estimated"
                 f" to be:")
        st.write(f"**{total_value}**")


def DrawInputsWidgets():

    # load dataset
    df = load_house_prices_data()
    percentageMin, percentageMax = 0.4, 2.0

    # we create input widgets for the 4 best features
    col01, col02 = st.beta_columns(2)
    col03, col04 = st.beta_columns(2)

    # We are using these features to feed the ML pipeline -
    # values copied from check_variables_for_UI() result

    # create an empty DataFrame, which will be the live data
    X_live = pd.DataFrame([], index=[0])

    # from here on we draw the widget based on the variable type
    # (numerical or categorical)
    # and set initial values

    with col01:
        feature = "OverallQual"
        st_widget = st.number_input(
            label='Overall Quality',
            min_value=0,
            max_value=10,
            value=int(df[feature].median()),
            step=1
        )
    X_live[feature] = st_widget

    with col02:
        feature = "TotalBsmtSF"
        st_widget = st.number_input(
            label='Total Basement SQFT',
            min_value=int(df[feature].min()*percentageMin),
            max_value=int(df[feature].max()*percentageMax),
            value=int(df[feature].median()),
            step=20
        )
    X_live[feature] = st_widget

    with col03:
        feature = "2ndFlrSF"
        st_widget = st.number_input(
            label='2nd Floor SQFT',
            min_value=int(df[feature].min()*percentageMin),
            max_value=int(df[feature].max()*percentageMax),
            value=int(df[feature].median()),
            step=20
        )
    X_live[feature] = st_widget

    with col04:
        feature = "GarageArea"
        st_widget = st.number_input(
            label="Garage Area SQFT",
            min_value=int(df[feature].min()*percentageMin),
            max_value=int(df[feature].max()*percentageMax),
            value=int(df[feature].median()),
            step=20
        )
    X_live[feature] = st_widget

    # st.write(X_live)

    return X_live
