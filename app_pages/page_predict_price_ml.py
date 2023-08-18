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
    version = 'v1'
    sale_price_pipe = load_pkl_file(
        f"outputs/ml_pipeline/predict_sale_price/{version}/regression_pipeline.pkl")
    sale_price_feat_importance = plt.imread(
        f"outputs/ml_pipeline/predict_sale_price/{version}/feat_importance.png")
    model_perform_img = plt.imread(
        f"outputs/ml_pipeline/predict_sale_price/{version}/model_image.png")
    X_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_sale_price/{version}/X_train.csv")
    X_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_sale_price/{version}/X_test.csv")
    y_train = pd.read_csv(
        f"outputs/ml_pipeline/predict_sale_price/{version}/y_train.csv")
    y_test = pd.read_csv(
        f"outputs/ml_pipeline/predict_sale_price/{version}/y_test.csv")

    st.write("## ML Pipeline: Predict Property Sale Price")
    # display pipeline training summary conclusions
    st.success(
        f" A Regressor model was trained to predict the sale price of"
        f" properties in Ames, Iowa. "
        f" The initial data set contained 23 Features and SalePrice as "
        f" the target."
        f" Two features were dropped due to around 90% of data missing."
        f" Feature engineering was carried out on the remaining data. "
        f" The model was tuned using a hyperparameter search and was found to "
        f" **meet the project requirement**: and R2 Score of 0.8 or better on "
        f" both train and test sets. ")
    st.write("---")

    # show pipeline steps
    st.write("### ML pipeline to predict property sale prices.")
    st.code(sale_price_pipe)
    st.write("---")

    # show best features
    st.write("### The features the model was trained and their importance.")
    st.write(X_train.columns.to_list())
    st.image(sale_price_feat_importance)
    st.write("---")

    # evaluate performance on both sets
    st.write("### Pipeline Performance")
    regression_performance(X_train=X_train, y_train=y_train,
                           X_test=X_test, y_test=y_test,
                           pipeline=sale_price_pipe)

    st.write("**Performance Plot**")
    st.image(model_perform_img)
    # regression_evaluation_plots(X_train=X_train, y_train=y_train, X_test=X_test,
    #                             y_test=y_test, pipeline=sale_price_pipe,
    #                             alpha_scatter=0.5)
