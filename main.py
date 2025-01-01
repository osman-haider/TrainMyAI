import streamlit as st
from src.utils import side_bar, binary_classification_ui

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main():
    """
    Main function to handle the Streamlit app logic.
    Includes dataset processing, model training, and inference functionality.
    """

    # Initialize session state variables
    if "process_completed" not in st.session_state:
        st.session_state["process_completed"] = False

    if "model_trained" not in st.session_state:
        st.session_state["model_trained"] = False

    if "binary_cl" not in st.session_state:
        st.session_state["binary_cl"] = None

    search_method, selected_parent, selected_child = side_bar.sidebar_ui(st)

    if st.session_state["process_completed"]:
        st.header("Model Training")

        if search_method == "CNN":
            if selected_parent == "Image Classification":
                if selected_child == "Binary Classification":

                    binary_classification_ui.binary_classification_cofig(st)

                elif selected_child == "Multi Class Classification":
                    st.info("CNN ---> Image Classification ---> Multi Class Classification")
            elif selected_parent == "Object Detection":
                st.info("CNN ---> Object Detection")
            elif selected_parent == "Image Segmentation":
                st.info("CNN ---> Image Segmentation")
            elif selected_parent == "Face Recognition":
                st.info("CNN ---> Face Recognition")
            elif selected_parent == "Style Transfer":
                st.info("CNN ---> Style Transfer")

        elif search_method == "ANN":
            if selected_parent == "Dataset Prediction":
                st.info("ANN ---> Dataset Prediction")
            elif selected_parent == "Credit Card Fraud Detection":
                st.info("ANN ---> Credit Card Fraud Detection")

        elif search_method == "RNN":
            if selected_parent == "Sentiment Analysis":
                st.info("RNN ---> Sentiment Analysis")
            elif selected_parent == "Time Series Forecasting":
                st.info("RNN ---> Time Series Forecasting")
            elif selected_parent == "Text Generation":
                st.info("RNN ---> Text Generation")
            elif selected_parent == "Machine Translation":
                st.info("RNN ---> Machine Translation")

        elif search_method == "LSTM":
            if selected_parent == "Text Summarization":
                st.info("LSTM ---> Text Summarization")
            elif selected_parent == "Language Modeling":
                st.info("LSTM ---> Language Modeling")
            elif selected_parent == "Time Series Forecasting":
                st.info("LSTM ---> Time Series Forecasting")
            elif selected_parent == "Dataset Prediction":
                st.info("LSTM ---> Dataset Prediction")

        elif search_method == "GAN":
            if selected_parent == "Image Generation":
                st.info("GAN ---> Image Generation")
            elif selected_parent == "Image Super-Resolution":
                st.info("GAN ---> Image Super-Resolution")
            elif selected_parent == "Facial Image Generation":
                st.info("GAN ---> Facial Image Generation")
            elif selected_parent == "Text-to-Image Generation":
                st.info("GAN ---> Text-to-Image Generation")

    else:
        st.info("Please complete the file upload and selections to start model training.")

if __name__ == "__main__":
    st.set_page_config(
        page_title="TrainMyAI",
        layout="wide"
    )
    main()