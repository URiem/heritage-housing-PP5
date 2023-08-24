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
        f"The price prediction will be based on various "
        f" features of the property in question, which the client can input"
        f" using the selections below. \n\n"
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

    # Generate Live Data
    X_live = DrawInputsWidgets()

    # predict on live data
    if st.button("Run Predictive Analysis"):
        predict_sale_price(X_live, sale_price_features, sale_price_pipe)

    st.write("---")

    st.write("## Price prediction for the clients inherited properties:")
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

    # we create input widgets for 24 features we will use 23 to start
    col01, col02, col03, col04 = st.beta_columns(4)
    # col05, col06, col07, col08 = st.beta_columns(4)
    # col09, col10, col11, col12 = st.beta_columns(4)
    # col13, col14, col15, col16 = st.beta_columns(4)
    # col17, col18, col19, col20 = st.beta_columns(4)
    # col21, col22, col23, col24 = st.beta_columns(4)

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
            step=50
        )
    X_live[feature] = st_widget

    with col03:
        feature = "2ndFlrSF"
        st_widget = st.number_input(
            label='2nd Floor SQFT',
            min_value=int(df[feature].min()*percentageMin),
            max_value=int(df[feature].max()*percentageMax),
            value=int(df[feature].median()),
            step=10
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

    # with col01:
    #     feature = "1stFlrSF"
    #     st_widget = st.number_input(
    #         label='1st Floor SQFT',
    #         min_value=int(df[feature].min()*percentageMin),
    #         max_value=int(df[feature].max()*percentageMax),
    #         value=int(df[feature].median()),
    #         step=50
    #     )
    # X_live[feature] = st_widget

    # with col03:
    #     feature = "BedroomAbvGr"
    #     st_widget = st.number_input(
    #         label='Bedrooms Above Ground',
    #         min_value=int(df[feature].min()*percentageMin),
    #         max_value=int(df[feature].max()*percentageMax),
    #         value=int(df[feature].median()),
    #         step=1
    #     )
    # X_live[feature] = st_widget

    # with col05:
    #     feature = "BsmtUnfSF"
    #     st_widget = st.number_input(
    #         label="Unfinished Basement SQFT",
    #         min_value=int(df[feature].min()*percentageMin),
    #         max_value=int(df[feature].max()*percentageMax),
    #         value=int(df[feature].median()),
    #         step=10
    #     )
    # X_live[feature] = st_widget

    # with col06:
    #     feature = "BsmtFinSF1"
    #     st_widget = st.number_input(
    #         label="Finished Basement SQFT",
    #         min_value=int(df[feature].min()*percentageMin),
    #         max_value=int(df[feature].max()*percentageMax),
    #         value=int(df[feature].median()),
    #         step=10
    #     )
    # X_live[feature] = st_widget

    # with col07:
    #     feature = "BsmtExposure"
    #     st_widget = st.selectbox(
    #         label="Basement Exposure",
    #         options=df[feature].unique()
    #     )
    # X_live[feature] = st_widget

    # with col08:
    #     feature = "BsmtFinType1"
    #     st_widget = st.selectbox(
    #         label="Basement Finish Type",
    #         options=df[feature].dropna().unique()
    #     )
    # X_live[feature] = st_widget

    # with col10:
    #     feature = "GarageFinish"
    #     st_widget = st.selectbox(
    #         label="Garage Finish",
    #         options=df[feature].dropna().unique()
    #     )
    # X_live[feature] = st_widget

    # with col11:
    #     feature = "GarageYrBlt"
    #     st_widget = st.number_input(
    #         label="Garage Year Built",
    #         min_value=int(df[feature].min()*percentageMin),
    #         max_value=date.today().year,
    #         value=int(df[feature].median()),
    #         step=1
    #     )
    # X_live[feature] = st_widget

    # with col12:
    #     feature = "GrLivArea"
    #     st_widget = st.number_input(
    #         label='Ground Living Area SQFT',
    #         min_value=int(df[feature].min()*percentageMin),
    #         max_value=int(df[feature].max()*percentageMax),
    #         value=int(df[feature].median()),
    #         step=25
    #     )
    # X_live[feature] = st_widget

    # with col13:
    #     feature = "LotArea"
    #     st_widget = st.number_input(
    #         label="Lot Area SQFT",
    #         min_value=int(df[feature].min()*percentageMin),
    #         max_value=int(df[feature].max()*percentageMax),
    #         value=int(df[feature].median()),
    #         step=25
    #     )
    # X_live[feature] = st_widget

    # with col14:
    #     feature = "LotFrontage"
    #     st_widget = st.number_input(
    #         label="Lot Frontage FT",
    #         min_value=int(df[feature].min()*percentageMin),
    #         max_value=int(df[feature].max()*percentageMax),
    #         value=int(df[feature].median()),
    #         step=25
    #     )
    # X_live[feature] = st_widget

    # with col15:
    #     feature = "MasVnrArea"
    #     st_widget = st.number_input(
    #         label="Masonry VnrArea",
    #         min_value=int(df[feature].min()*percentageMin),
    #         max_value=int(df[feature].max()*percentageMax),
    #         value=int(df[feature].median()),
    #         step=25
    #     )
    # X_live[feature] = st_widget

    # with col16:
    #     feature = "KitchenQual"
    #     st_widget = st.selectbox(
    #         label='Kitchen Quality',
    #         options=df[feature].unique()
    #     )
    # X_live[feature] = st_widget

    # with col17:
    #     feature = "OverallCond"
    #     st_widget = st.number_input(
    #         label='Overall Condition',
    #         min_value=0,
    #         max_value=10,
    #         value=int(df[feature].median()),
    #         step=1
    #     )
    # X_live[feature] = st_widget

    # with col19:
    #     feature = "OpenPorchSF"
    #     st_widget = st.number_input(
    #         label='Open Porch SQFT',
    #         min_value=int(df[feature].min()*percentageMin),
    #         max_value=int(df[feature].max()*percentageMax),
    #         value=int(df[feature].median()),
    #         step=20
    #     )
    # X_live[feature] = st_widget

    # with col20:
    #     feature = "YearBuilt"
    #     st_widget = st.number_input(
    #         label='Year Built',
    #         min_value=int(df[feature].min()*percentageMin),
    #         max_value=date.today().year,
    #         value=int(df[feature].median()),
    #         step=1
    #     )
    # X_live[feature] = st_widget

    # with col21:
    #     feature = "YearRemodAdd"
    #     st_widget = st.number_input(
    #         label='Year of Remodel/Addition',
    #         min_value=int(df[feature].min()*percentageMin),
    #         max_value=date.today().year,
    #         value=int(df[feature].median()),
    #         step=1
    #     )
    # X_live[feature] = st_widget

    # with col22:
    #     feature = "WoodDeckSF"
    #     st_widget = st.number_input(
    #         label='Wood Deck SQFT',
    #         min_value=int(df[feature].min()*percentageMin),
    #         max_value=int(df[feature].max()*percentageMax),
    #         value=int(df[feature].median()),
    #         step=20
    #     )
    # X_live[feature] = st_widget

    # with col23:
    #     feature = "EnclosedPorch"
    #     st_widget = st.number_input(
    #         label='Enclosed Porch SQFT',
    #         min_value=int(df[feature].min()*percentageMin),
    #         max_value=int(df[feature].max()*percentageMax),
    #         value=int(df[feature].median()),
    #         step=20
    #     )
    # X_live[feature] = st_widget

    # st.write(X_live)

    return X_live
