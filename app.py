import streamlit as st
import pandas as pd
import joblib

# ==========================
# LOAD MODELS
# ==========================

kmeans = joblib.load("kmeans_model.pkl")
scaler = joblib.load("scaler.pkl")
similarity_df = joblib.load("product_similarity.pkl")

# ==========================
# PAGE CONFIG
# ==========================

st.set_page_config(
    page_title="Shopper Spectrum",
    page_icon="🛍",
    layout="wide"
)

# ==========================
# TITLE
# ==========================

st.title("🛍 Shopper Spectrum")
st.subheader(
    "Customer Segmentation and Product Recommendation System"
)

# ==========================
# SIDEBAR
# ==========================

menu = st.sidebar.radio(
    "Select Module",
    [
        "🏠 Home",
        "🎯 Product Recommendation",
        "👥 Customer Segmentation"
    ]
)

# ==========================
# HOME
# ==========================

if menu == "🏠 Home":

    st.markdown("""
    ### Welcome

    This application provides:

    ✅ Product Recommendations

    ✅ Customer Segmentation using RFM Analysis

    Built using:
    - Python
    - Machine Learning
    - Streamlit
    """)

# ==========================
# PRODUCT RECOMMENDATION
# ==========================

elif menu == "🎯 Product Recommendation":

    st.header("🎯 Product Recommendation")

    product_name = st.text_input(
        "Enter Product Name"
    )

    if st.button("Get Recommendations"):

        if product_name not in similarity_df.index:

            st.error("Product Not Found")

        else:

            recommendations = (
                similarity_df[product_name]
                .sort_values(ascending=False)
                .iloc[1:6]
                .index
                .tolist()
            )

            st.success(
                "Top 5 Recommended Products"
            )

            for i, product in enumerate(
                recommendations,
                start=1
            ):
                st.write(f"{i}. {product}")

# ==========================
# CUSTOMER SEGMENTATION
# ==========================

elif menu == "👥 Customer Segmentation":

    st.header("👥 Customer Segmentation")

    recency = st.number_input(
        "Recency",
        min_value=0
    )

    frequency = st.number_input(
        "Frequency",
        min_value=0
    )

    monetary = st.number_input(
        "Monetary",
        min_value=0.0
    )

    if st.button("Predict Cluster"):

        data = [[
            recency,
            frequency,
            monetary
        ]]

        scaled_data = scaler.transform(
            data
        )

        cluster = kmeans.predict(
            scaled_data
        )[0]

        cluster_mapping = {
            0: "High-Value",
            1: "Regular",
            2: "Occasional",
            3: "At-Risk"
        }

        segment = cluster_mapping.get(
            cluster,
            "Unknown"
        )

        st.success(
            f"Customer Segment: {segment}"
        )