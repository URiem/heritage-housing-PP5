import streamlit as st

# Function predicts house prices using the regression pipeline


def predict_sale_price(X_live, property_features, sale_price_pipeline):

    # from live data, subset features related to this pipeline
    # the features are filtered using the list of features from the pipeline
    # this is to avoid a scilent fail in case the input features
    # are not in the same order as in the dataset used to train the model.
    X_live_sale_price = X_live.filter(property_features)

    # predict
    sale_price_prediction = sale_price_pipeline.predict(X_live_sale_price)

    statement = (
        f"* Given the features provided for the property, the model has "
        f"  predicted a sale value of:"
    )

    # Format the value written to the page
    if len(sale_price_prediction) == 1:
        price = float(sale_price_prediction.round(1))
        price = '${:,.2f}'.format(price)

        st.write(statement)
        st.write(f"**{price}**")
    else:
        st.write(
            f"* Given the features provided for the inherited properties, "
            f" the model has predicted sale values of:")
        st.write(sale_price_prediction)

    return sale_price_prediction
