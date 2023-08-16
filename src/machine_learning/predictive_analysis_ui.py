import streamlit as st

# Change this to just contain one function to predict house prices

def predict_sale_price(X_live, property_features, sale_price_pipeline):

    # from live data, subset features related to this pipeline
    X_live_sale_price = X_live.filter(property_features)

    # predict
    sale_price_prediction = sale_price_pipeline.predict(X_live_sale_price)

    # create a logic to display the results
    # proba = tenure_prediction_proba[0, tenure_prediction][0]*100
    # tenure_levels = tenure_labels_map[tenure_prediction[0]]

    statement = (
        f"* Given the features provided for the property, the model has predicted"
        f"  a sale value of:"
        )

    st.write(statement)
    st.write(sale_price_prediction)
