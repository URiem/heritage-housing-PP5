from src.machine_learning.predictive_analysis_ui import predict_sale_price
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_management import load_house_prices_data, load_pkl_file
from src.machine_learning.evaluate_clf import regression_performance


def page_sale_price_predictor_body():

    # load predict sale price files
    version = 'v1'
    sale_price_pipe = load_pkl_file(
        f"outputs/ml_pipeline/predict_sale_price/{version}/regression_pipeline.pkl")
    sale_price_features = (
        pd.read_csv(
            f"outputs/ml_pipeline/predict_sale_price/{version}/X_train.csv")
        .columns
        .to_list()
    )

    st.write("### Sale Price Predictor Interface")
    st.info(
        f"* The client is interested in determining the likely sale price of a "
        f" home in Ames, Iowa. The price prediction will be based on various "
        f" features of the property in question, which the client and input"
        f" using the selections below."
    )
    st.write("---")

    # Generate Live Data
    # check_variables_for_UI(sale_price_features)
    X_live = DrawInputsWidgets()

    # predict on live data
    if st.button("Run Predictive Analysis"):
        sale_price_prediction = predict_sale_price(
            X_live, churn_features, churn_pipe_dc_fe, churn_pipe_model)


# def check_variables_for_UI(sale_price_features):
#     import itertools

#     # The widgets inputs are the features used in sale price pipeline
#     combined_features = set(
#         list(
#             itertools.chain(sale_price_features)
#         )
#     )
#     st.write(
#         f"* There are {len(combined_features)} features for the UI: \n\n {combined_features}")


def DrawInputsWidgets():

    # load dataset
    df = load_house_prices_data()
    percentageMin, percentageMax = 0.4, 2.0

# we create input widgets only for 24 features we will use 21 to start
    col01, col02, col03, col04 = st.beta_columns(4)
    col05, col06, col07, col08 = st.beta_columns(4)
    col09, col10, col11, col12 = st.beta_columns(4)
    col13, col14, col15, col16 = st.beta_columns(4)
    col17, col18, col19, col20 = st.beta_columns(4)
    col21, col22, col23, col24 = st.beta_columns(4)

    # We are using these features to feed the ML pipeline - values copied from check_variables_for_UI() result

    # create an empty DataFrame, which will be the live data
    X_live = pd.DataFrame([], index=[0])

    # from here on we draw the widget based on the variable type (numerical or categorical)
    # and set initial values
    with col01:
        feature = "1stFlrSF"
        st_widget = st.number_input(
            label=feature,
            min_value=int(df[feature].min()*percentageMin),
            max_value=int(df[feature].max()*percentageMax),
            value=int(df[feature].median()),
            step=50
        )
    X_live[feature] = st_widget

    with col02:
        feature = "2ndFlrSF"
        st_widget = st.number_input(
            label=feature,
            min_value=int(df[feature].min()*percentageMin),
            max_value=int(df[feature].max()*percentageMax),
            value=int(df[feature].median()),
            step=10
        )
    X_live[feature] = st_widget

    with col03:
        feature = "BedroomAbvGr"
        st_widget = st.number_input(
            label=feature,
            min_value=int(df[feature].min()*percentageMin),
            max_value=int(df[feature].max()*percentageMax),
            value=int(df[feature].median()),
            step=1
        )
    X_live[feature] = st_widget

    with col04:
        feature = "TotalBsmtSF"
        st_widget = st.number_input(
            label=feature,
            min_value=int(df[feature].min()*percentageMin),
            max_value=int(df[feature].max()*percentageMax),
            value=int(df[feature].median()),
            step=50
        )
    X_live[feature] = st_widget

    with col05:
        feature = "BsmtUnfSF"
        st_widget = st.number_input(
            label=feature,
            min_value=int(df[feature].min()*percentageMin),
            max_value=int(df[feature].max()*percentageMax),
            value=int(df[feature].median()),
            step=10
        )
    X_live[feature] = st_widget

    with col06:
        feature = "BsmtFinSF1"
        st_widget = st.number_input(
            label=feature,
            min_value=int(df[feature].min()*percentageMin),
            max_value=int(df[feature].max()*percentageMax),
            value=int(df[feature].median()),
            step=10
        )
    X_live[feature] = st_widget

    with col07:
        feature = "BsmtExposure"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique()
        )
    X_live[feature] = st_widget

    with col08:
        feature = "BsmtFinType1"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique()
        )
    X_live[feature] = st_widget

    with col09:
        feature = "GarageArea"
        st_widget = st.number_input(
            label=feature,
            min_value=int(df[feature].min()*percentageMin),
            max_value=int(df[feature].max()*percentageMax),
            value=int(df[feature].median()),
            step=20
        )
    X_live[feature] = st_widget

    with col10:
        feature = "GarageFinish"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique()
        )
    X_live[feature] = st_widget

    with col11:
        feature = "GarageYrBlt"
        st_widget = st.number_input(
            label=feature,
            min_value=df[feature].min()*percentageMin,
            max_value=df[feature].max()*percentageMax,
            value=df[feature].median()
        )
    X_live[feature] = st_widget

    with col12:
        feature = "GrLivArea"
        st_widget = st.number_input(
            label=feature,
            min_value=df[feature].min()*percentageMin,
            max_value=df[feature].max()*percentageMax,
            value=df[feature].median()
        )
    X_live[feature] = st_widget

    with col13:
        feature = "LotArea"
        st_widget = st.number_input(
            label=feature,
            min_value=df[feature].min()*percentageMin,
            max_value=df[feature].max()*percentageMax,
            value=df[feature].median()
        )
    X_live[feature] = st_widget

    with col14:
        feature = "LotFrontage"
        st_widget = st.number_input(
            label=feature,
            min_value=df[feature].min()*percentageMin,
            max_value=df[feature].max()*percentageMax,
            value=df[feature].median()
        )
    X_live[feature] = st_widget

    with col15:
        feature = "MasVnrArea"
        st_widget = st.number_input(
            label=feature,
            min_value=df[feature].min()*percentageMin,
            max_value=df[feature].max()*percentageMax,
            value=df[feature].median()
        )
    X_live[feature] = st_widget

    with col16:
        feature = "KitchenQual"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique()
        )
    X_live[feature] = st_widget

    with col17:
        feature = "OverallCond"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique()
        )
    X_live[feature] = st_widget

    with col18:
        feature = "OverallQual"
        st_widget = st.selectbox(
            label=feature,
            options=df[feature].unique()
        )
    X_live[feature] = st_widget

    with col19:
        feature = "OpenPorchSF"
        st_widget = st.number_input(
            label=feature,
            min_value=df[feature].min()*percentageMin,
            max_value=df[feature].max()*percentageMax,
            value=df[feature].median()
        )
    X_live[feature] = st_widget

    with col20:
        feature = "YearBuilt"
        st_widget = st.number_input(
            label=feature,
            min_value=df[feature].min()*percentageMin,
            max_value=df[feature].max()*percentageMax,
            value=df[feature].median()
        )
    X_live[feature] = st_widget

    with col21:
        feature = "YearRemodAdd"
        st_widget = st.number_input(
            label=feature,
            min_value=df[feature].min()*percentageMin,
            max_value=df[feature].max()*percentageMax,
            value=df[feature].median()
        )
    X_live[feature] = st_widget

    # st.write(X_live)

    return X_live
