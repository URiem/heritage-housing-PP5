import streamlit as st
from app_pages.multipage import MultiPage

# load pages scripts
from app_pages.page_summary import page_summary_body
# from app_pages.page_sale_price_analysis import page_sale_price_analysis_body
# from app_pages.page_predict_sale_price import page_predict_sale_price_body
from app_pages.page_project_hypothesis import page_project_hypothesis_body
# from app_pages.page_predict_price_ml import page_predict_price_ml_body

app = MultiPage(app_name= "Heritage Housing Sale Price Predictor") # Create an instance of the app 

# Add your app pages here using .add_page()
app.add_page("Quick Project Summary", page_summary_body)
# app.add_page("Sale Price Analysis", page_sale_price_analysis_body)
# app.add_page("Predict Sale Price", page_predict_sale_price_body)
app.add_page("Project Hypothesis and Validation", page_project_hypothesis_body)
# app.add_page("ML: Price Prediction", page_predict_price_ml_body)

app.run() # Run the  app