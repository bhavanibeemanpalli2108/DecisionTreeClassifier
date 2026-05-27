"""
Streamlit web app — interactive Iris flower species predictor.
"""

import sys
from pathlib import Path

# Add project root to path so src modules can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.prediction import IrisPredictor
from src.data_ingestion import CLASS_NAMES

# Page configuration
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered",
)

st.title("🌸 Iris Flower Species Classifier")
st.markdown(
    """
    Enter the flower measurements below and click **Predict** to classify
    the species using a **Decision Tree Classifier** model trained on the Iris dataset.

    **Classes:** Setosa · Versicolor · Virginica
    """
)

# Input fields for the four Iris features
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input(
        "Sepal Length (cm)",
        min_value=0.0,
        max_value=15.0,
        value=5.1,
        step=0.1,
        help="Length of the sepal in centimeters",
    )
    petal_length = st.number_input(
        "Petal Length (cm)",
        min_value=0.0,
        max_value=15.0,
        value=1.4,
        step=0.1,
        help="Length of the petal in centimeters",
    )

with col2:
    sepal_width = st.number_input(
        "Sepal Width (cm)",
        min_value=0.0,
        max_value=15.0,
        value=3.5,
        step=0.1,
        help="Width of the sepal in centimeters",
    )
    petal_width = st.number_input(
        "Petal Width (cm)",
        min_value=0.0,
        max_value=15.0,
        value=0.2,
        step=0.1,
        help="Width of the petal in centimeters",
    )

# Predict button
if st.button("Predict", type="primary", use_container_width=True):
    try:
        predictor = IrisPredictor()
        species = predictor.predict(
            sepal_length=sepal_length,
            sepal_width=sepal_width,
            petal_length=petal_length,
            petal_width=petal_width,
        )

        st.success(f"**Predicted Species: {species}**")

        # Show reference ranges for context
        st.info(
            f"This flower was classified as **{species}**, "
            f"one of the three Iris species: {', '.join(CLASS_NAMES)}."
        )
    except FileNotFoundError:
        st.error(
            "Model artifacts not found. Please run `python main.py` first "
            "to train the model and generate artifacts."
        )
    except Exception as exc:
        st.error(f"Prediction failed: {exc}")

# Sidebar with project info
with st.sidebar:
    st.header("About")
    st.markdown(
        """
        This app uses a **Decision Tree Classifier** trained on the
        scikit-learn Iris dataset.

        **Pipeline:**
        1. Data ingestion
        2. Cleaning & outlier handling
        3. Train-test split
        4. Decision Tree training (max_depth tuned)
        5. Prediction

        Run `python main.py` to (re)train the model.
        """
    )
