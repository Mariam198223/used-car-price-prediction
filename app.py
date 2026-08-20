import streamlit as st
import pandas as pd
import joblib


# ==========================================
# Page Configuration
# ==========================================

st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)


# ==========================================
# Load Model and Preprocessor
# ==========================================

@st.cache_resource
def load_files():

    model = joblib.load("final_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")

    return model, preprocessor


model, preprocessor = load_files()


# ==========================================
# Custom CSS
# ==========================================

st.markdown("""
<style>

    .block-container {
        max-width: 900px;
        padding-top: 3rem;
        padding-bottom: 2rem;
    }

    .main-title {
        text-align: center;
        font-size: 38px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .subtitle {
        text-align: center;
        opacity: 0.65;
        font-size: 16px;
        margin-bottom: 35px;
    }

    .section-title {
        font-size: 22px;
        font-weight: 650;
        margin-bottom: 20px;
    }

    label {
        font-weight: 500 !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="input"] > div {
        border-radius: 10px;
    }

    div.stButton > button {
        width: 100%;
        height: 48px;
        border-radius: 10px;
        font-size: 16px;
        font-weight: 600;
        margin-top: 15px;
    }

    .footer {
        text-align: center;
        opacity: 0.5;
        font-size: 12px;
        margin-top: 35px;
    }

</style>
""", unsafe_allow_html=True)


# ==========================================
# Header
# ==========================================

st.markdown(
    '<div class="main-title">🚗 Used Car Price Prediction</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Enter the car details below to get an estimated price.'
    '</div>',
    unsafe_allow_html=True
)


# ==========================================
# Car Information
# ==========================================

st.markdown(
    '<div class="section-title">Car Information</div>',
    unsafe_allow_html=True
)


# ==========================================
# Row 1
# ==========================================

col1, col2 = st.columns(2)

with col1:

    name = st.selectbox(
        "Car Name",
        [
            "Kia Forte",
            "Chevrolet Silverado 1500",
            "Toyota RAV4",
            "Honda Civic",
            "Honda Accord",
            "Mercedes-Benz GLC",
            "BMW 5 Series",
            "Jeep Wrangler"
        ]
    )


with col2:

    year = st.number_input(
        "Year",
        value=2020,
        step=1
    )

    if year > 2024:
        st.error("You really wanna convince me you can afford a car newer than a 2024 model?")
    elif year < 2000 :
        st.error("Sorry, nothing older than a 2000 model here. 💀")



# ==========================================
# Row 2
# ==========================================

col3, col4 = st.columns(2)

with col3:

    miles = st.number_input(
        "Miles",
        min_value=0,
        value=30000,
        step=100
    )

    if miles > 310000:
        st.warning(
            "Over 310,000 miles? Honestly, just save your money and don’t buy a car.🥸"
        )


with col4:

    accidents = st.number_input(
        "Accidents",
        min_value=0,
        value=0,
        step=1
    )

    if accidents > 5:
        st.warning(
            "5+ accidents? Looking for a car or scrap? 🤨"
        )


# ==========================================
# Row 3
# ==========================================

col5, col6 = st.columns(2)

with col5:

    Owner = st.number_input(
        "Number of Owners",
        min_value=1,
        value=1,
        step=1
    )

    if Owner > 8:
        st.warning(
            "8 owners is enough, dear! ✋🏻😔"
        )


with col6:

    exterior_color = st.selectbox(
        "Exterior Color",
        [
            "Black",
            "White",
            "Gray",
            "Silver",
            "Blue",
            "Red",
            "Green",
            "Unknown"
        ]
    )


# ==========================================
# Row 4
# ==========================================

col7, col8 = st.columns(2)

with col7:

    interior_color = st.selectbox(
        "Interior Color",
        [
            "Black",
            "Gray",
            "Beige",
            "Brown",
            "Red",
            "White",
            "Unknown"
        ]
    )


# ==========================================
# Prediction Button
# ==========================================

st.markdown("<br>", unsafe_allow_html=True)

predict = st.button(
    "Predict Price 🚀",
    use_container_width=True
)


# ==========================================
# Prediction
# ==========================================

if predict:

    # ======================================
    # Validate Inputs
    # ======================================

    if year < 2000 or year > 2024:

        st.error(
            "❌ Please enter a year between 2000 and 2024."
        )

    elif miles > 310000:

        st.error(
            "❌ Miles must be 310,000 or less."
        )

    elif accidents > 5:

        st.error(
            "❌ Accidents must be 5 or less."
        )

    elif Owner > 8:

        st.error(
            "❌ Number of owners must be 8 or less."
        )

    else:

        # ======================================
        # Create Input DataFrame
        # ======================================

        input_data = pd.DataFrame({
            "name": [name],
            "year": [year],
            "miles": [miles],
            "accidents": [accidents],
            "Owner": [Owner],
            "exterior_color": [exterior_color],
            "interior_color": [interior_color]
        })


        try:

            # ======================================
            # Apply Preprocessing
            # ======================================

            input_transformed = preprocessor.transform(
                input_data
            )


            # ======================================
            # Prediction
            # ======================================

            prediction = model.predict(
                input_transformed
            )

            predicted_price = prediction[0]


            # ======================================
            # Prediction Result
            # ======================================

            st.success(
                "Prediction completed successfully! 🚀"
            )

            st.metric(
                label="Estimated Car Price",
                value=f"${predicted_price:,.2f}"
            )

            st.caption(
                f"{name} • {year} • {miles:,} miles"
            )


        except Exception as e:

            st.error(
                "An error occurred while making the prediction."
            )

            st.code(str(e))


# ==========================================
# Footer
# ==========================================

st.markdown(
    """
    <div class="footer">
        Used Car Price Prediction |
        Machine Learning Regression Project |
        Developed by Mariam Ibrahiem - August 2026
    </div>
    """,
    unsafe_allow_html=True
)
