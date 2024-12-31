import streamlit as st
import os
import zipfile
from pathlib import Path

def extract_zip(zip_file):
    """Extracts the zip file contents and returns the list of file paths."""
    extracted_files = []
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall("extracted_folder")
        extracted_files = zip_ref.namelist()
    return extracted_files

def main():

    # Initialize session state
    if 'process_completed' not in st.session_state:
        st.session_state['process_completed'] = False
    search_method = None
    selected_parent = None
    selected_child = None

    # Create sidebar for data selection
    with st.sidebar:
        st.title("TrainMyAI")
        st.header("Select your favorite model")

        search_method = st.selectbox(
            "Select search method:",
            ["Select Model Type", "CNN", "ANN", "RNN", "LSTM", "GAN"],
            label_visibility="collapsed"
        )

        # Show options based on selection
        if search_method == "CNN":
            parent_options = ["Image Classification", "Object Detection", "Image Segmentation", "Face Recognition", "Style Transfer"]
            selected_parent = st.selectbox(
                "Select type of Main Model:",
                ["Select an option"] + parent_options,
                label_visibility="collapsed"
            )

            if selected_parent == "Image Classification":
                child_options = ["Binary Classification", "Multi Class Classification"]
                selected_child = st.selectbox(
                    "Select type of Classification Model:",
                    ["Select an option"] + child_options,
                    label_visibility="collapsed"
                )

        elif search_method == "ANN":
            parent_options = ["Dataset Prediction", "Credit Card Fraud Detection"]
            selected_parent = st.selectbox(
                "Select type of Main Model:",
                ["Select an option"] + parent_options,
                label_visibility="collapsed"
            )

        elif search_method == "RNN":
            parent_options = ["Sentiment Analysis", "Time Series Forecasting", "Text Generation", "Machine Translation"]
            selected_parent = st.selectbox(
                "Select type of Main Model:",
                ["Select an option"] + parent_options,
                label_visibility="collapsed"
            )

        elif search_method == "LSTM":
            parent_options = ["Text Summarization", "Language Modeling", "Time Series Forecasting", "Dataset Prediction"]
            selected_parent = st.selectbox(
                "Select type of Main Model:",
                ["Select an option"] + parent_options,
                label_visibility="collapsed"
            )

        elif search_method == "GAN":
            parent_options = ["Image Generation", "Image Super-Resolution", "Facial Image Generation", "Text-to-Image Generation"]
            selected_parent = st.selectbox(
                "Select type of Main Model:",
                ["Select an option"] + parent_options,
                label_visibility="collapsed"
            )

        # File upload section
        st.subheader("Upload Folder (zip file)")

        uploaded_file = st.file_uploader(
            "Upload your zip file containing data files",
            type='zip'  # Only accept zip files
        )

        # Submit button
        submit_button = st.button("Process Files")

        if submit_button:
            if not uploaded_file:
                st.error("Please upload a zip file")
            elif search_method == "Select Model Type" or selected_parent == "Select an option" or (selected_child is not None and selected_child == "Select an option"):
                st.error("Please make all required selections in the dropdown menus")
            else:
                st.success("Processing uploaded zip file")

                # Save the uploaded zip file to the session state
                st.session_state['uploaded_zip'] = uploaded_file

                # Extract the zip file
                extracted_files = extract_zip(uploaded_file)
                st.session_state['extracted_files'] = extracted_files

                # Update the process completed flag
                st.session_state['process_completed'] = True

                # Show the extracted files
                with st.expander("Extracted Files"):
                    for file in extracted_files:
                        st.write(f"📄 {file}")

    # Display content only after successful processing
    if st.session_state['process_completed']:
        st.header("Model Training")

        if search_method == "CNN":
            if selected_parent == "Image Classification":
                if selected_child == "Binary Classification":
                    st.info("CNN ---> Image Classification ---> Binary Classification")
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