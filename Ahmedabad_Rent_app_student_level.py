import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# ---------------------------------------------------------
# 1. PAGE SETTINGS
# ---------------------------------------------------------

st.set_page_config(
    page_title="Ahmedabad Rent Predictor",
    layout="centered"
)

st.title("Ahmedabad House Rent Predictor")
st.write("Enter the details of a property to estimate its monthly rent.")

# ---------------------------------------------------------
# 2. LOAD THE DATASET
# ---------------------------------------------------------

rent = pd.read_csv("Ahmedabad_rent.csv")

# Remove exact duplicate rows
rent = rent.drop_duplicates()

# ---------------------------------------------------------
# 3. CLEAN THE PRICE COLUMN
# ---------------------------------------------------------

# Remove commas from values such as 20,000
rent["price"] = rent["price"].str.replace(",", "")

# Convert price from text to float
rent["price"] = rent["price"].astype(float)

# Convert small lakh-format values such as 1.1 to rupees
rent["price"] = np.where(
    rent["price"] < 100,
    rent["price"] * 100000,
    rent["price"]
)

# ---------------------------------------------------------
# 4. CLEAN THE BATHROOM COLUMN
# ---------------------------------------------------------

def get_bathroom(x):
    # Missing value
    if pd.isnull(x):
        return np.nan

    # Normal values look like "2 bathrooms"
    if "bathroom" in str(x):
        return float(str(x).split()[0])

    # Invalid values such as "North facing"
    return np.nan

rent["bathroom"] = rent["bathroom"].apply(get_bathroom)

# Fill missing bathroom values with the median
rent["bathroom"] = rent["bathroom"].fillna(rent["bathroom"].median())

# ---------------------------------------------------------
# 5. SAVE THE ORIGINAL CATEGORY NAMES
#    BEFORE ENCODING
# ---------------------------------------------------------

# These encoders change text categories into numbers.
le_seller = LabelEncoder()
le_layout = LabelEncoder()
le_property = LabelEncoder()
le_locality = LabelEncoder()
le_furnish = LabelEncoder()

# Fit the encoders on the dataset
le_seller.fit(rent["seller_type"])
le_layout.fit(rent["layout_type"])
le_property.fit(rent["property_type"])
le_locality.fit(rent["locality"])
le_furnish.fit(rent["furnish_type"])

# ---------------------------------------------------------
# 6. ENCODE THE CATEGORICAL COLUMNS
# ---------------------------------------------------------

rent["seller_type"] = le_seller.transform(rent["seller_type"])
rent["layout_type"] = le_layout.transform(rent["layout_type"])
rent["property_type"] = le_property.transform(rent["property_type"])
rent["locality"] = le_locality.transform(rent["locality"])
rent["furnish_type"] = le_furnish.transform(rent["furnish_type"])

# ---------------------------------------------------------
# 7. SELECT INPUTS (X) AND TARGET (y)
# ---------------------------------------------------------

X = rent.drop("price", axis=1)
y = rent["price"]

# Split data into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ---------------------------------------------------------
# 8. CREATE AND TRAIN LINEAR REGRESSION MODEL
# ---------------------------------------------------------

model = LinearRegression()

model.fit(X_train, y_train)

# ---------------------------------------------------------
# 9. TEST THE MODEL
# ---------------------------------------------------------

y_pred = model.predict(X_test)

mae = metrics.mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
r2 = metrics.r2_score(y_test, y_pred)

# ---------------------------------------------------------
# 10. SIDEBAR MENU
# ---------------------------------------------------------

page = st.sidebar.radio(
    "Choose a page",
    ["Rent Prediction", "Project Information"]
)

# ---------------------------------------------------------
# 11. RENT PREDICTION PAGE
# ---------------------------------------------------------

if page == "Rent Prediction":

    st.subheader("Enter Property Details")

    col1, col2 = st.columns(2)

    with col1:
        # Categorical choices
        seller_choice = st.selectbox(
            "Seller Type",
            le_seller.classes_
        )

        layout_choice = st.selectbox(
            "Layout Type",
            le_layout.classes_
        )

        property_choice = st.selectbox(
            "Property Type",
            le_property.classes_
        )

        locality_choice = st.selectbox(
            "Locality",
            le_locality.classes_
        )

    with col2:
        # Numerical choices
        bedroom = st.number_input(
            "Number of Bedrooms",
            min_value=1,
            max_value=20,
            value=2,
            step=1
        )

        area = st.number_input(
            "Area (sq.ft)",
            min_value=100,
            max_value=100000,
            value=1200,
            step=50
        )

        bathroom = st.number_input(
            "Number of Bathrooms",
            min_value=1,
            max_value=15,
            value=2,
            step=1
        )

        furnish_choice = st.selectbox(
            "Furnishing Type",
            le_furnish.classes_
        )

    st.write("")

    predict_button = st.button("Predict Monthly Rent")

    if predict_button:

        # Convert the selected text values into the same
        # numeric codes used when training the model.
        seller_code = le_seller.transform([seller_choice])[0]
        layout_code = le_layout.transform([layout_choice])[0]
        property_code = le_property.transform([property_choice])[0]
        locality_code = le_locality.transform([locality_choice])[0]
        furnish_code = le_furnish.transform([furnish_choice])[0]

        # Create one row containing the user's property details.
        input_data = pd.DataFrame(
            [[
                seller_code,
                bedroom,
                layout_code,
                property_code,
                locality_code,
                area,
                furnish_code,
                bathroom
            ]],
            columns=X.columns
        )

        # Predict the monthly rent
        prediction = model.predict(input_data)[0]

        st.success(
            f"Estimated Monthly Rent: ₹{prediction:,.0f}"
        )

        st.write(
            "This is an estimated rent based on the historical "
            "Ahmedabad rental data used in this project."
        )

# ---------------------------------------------------------
# 12. PROJECT INFORMATION PAGE
# ---------------------------------------------------------

else:

    st.subheader("Project Information")

    st.write(
        "This project uses Ahmedabad rental listings to estimate "
        "the monthly rent of a property."
    )

    st.write("Number of cleaned listings:", len(rent))
    st.write("Number of localities:", len(le_locality.classes_))
    st.write("Median monthly rent:", f"₹{rent['price'].median():,.0f}")
    st.write("Median area:", f"{rent['area'].median():,.0f} sq.ft")

    st.divider()

    st.subheader("Model Used")
    st.write("Linear Regression")

    st.write("MAE:", round(mae, 2))
    st.write("RMSE:", round(rmse, 2))
    st.write("R2 Score:", round(r2 * 100, 2), "%")

    st.divider()

    st.subheader("Features Used for Prediction")

    st.write("""
    - Seller Type
    - Number of Bedrooms
    - Layout Type
    - Property Type
    - Locality
    - Area
    - Furnishing Type
    - Number of Bathrooms
    """)

    st.info(
        "No pickle or joblib model file is used. "
        "The model is trained directly from Ahmedabad_rent.csv "
        "when the Streamlit application starts."
    )
