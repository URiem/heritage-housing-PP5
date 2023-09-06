import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from src.data_management import load_house_prices_data, load_pkl_file
from src.machine_learning.evaluate_regression import (
    regression_performance,
    regression_evaluation,
    regression_evaluation_plots)


def page_predict_price_ml_body():

    # load regression pipeline files
    version = 'v2'
    sale_price_pipe = load_pkl_file(
        f"outputs/ml_pipeline/predict_sale_price/{version}/regression_pipeline.pkl")  # noqa
    sale_price_feat_importance = plt.imread(
        f"outputs/ml_pipeline/predict_sale_price/{version}/features_importance.png")  # noqa
    X_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_sale_price/{version}/X_train.csv")
    X_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_sale_price/{version}/X_test.csv")
    y_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_sale_price/{version}/y_train.csv").squeeze()  # noqa
    y_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_sale_price/{version}/y_test.csv").squeeze()  # noqa

    st.write("### ML Pipeline: Predict Property Sale Price")
    # display pipeline training summary conclusions
    st.success(
        f" A Regressor model was trained to predict the sale price of"
        f" properties in Ames, Iowa. "
        f" The initial data set contained 23 features and 'SalePrice' as "
        f" the target."
        f" Two features were dropped due to around 90% of data points missing."
        f" Feature engineering was carried out on the remaining data. "
        f" The model was then tuned using a hyperparameter search and was "
        f" found to "
        f" **meet the project requirement** with an R2 Score of 0.8 or "
        f" better on "
        f" both train and test sets. The model identified the four most "
        f" important features necessary to acchieve the best predictive "
        f" power. ")
    st.write("---")

    # show pipeline steps
    st.write("### ML pipeline to predict property sale prices.")
    st.code(sale_price_pipe)
    st.write("---")

    # show best features
    st.write("### The features the model was trained on and their importance.")
    st.write(X_train.columns.to_list())
    st.image(sale_price_feat_importance)

    st.write(
        f"The model was ultimately trained on  the following four features: \n"
        f"* Overall Quality (OverallQual) \n"
        f"* Total Basement Area in squarefeet (TotalBsmtSF) \n"
        f"* 2nd Floor Area in squarefeet (2ndFlrSF) \n"
        f"* Garage Area in squarefeet (GarageArea) \n"
    )
    st.write("---")

    # evaluate performance on both sets
    st.write("### Pipeline Performance")
    regression_performance(X_train=X_train, y_train=y_train,
                           X_test=X_test, y_test=y_test,
                           pipeline=sale_price_pipe)

    st.write("**Performance Plot**")
    regression_evaluation_plots(X_train=X_train, y_train=y_train,
                                X_test=X_test,
                                y_test=y_test, pipeline=sale_price_pipe,
                                alpha_scatter=0.5)
