import streamlit as st
import zipfile
from src.utils.cnn.image_classification import binary_classification
import json
import tensorflow as tf
from src.utils import traning_log

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def extract_zip(zip_file):
    """Extracts the zip file contents and returns the list of file paths."""
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall("extracted_folder")
        print("File is extracted successfully...")


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

    search_method = None
    selected_parent = None
    selected_child = None

    with st.sidebar:
        st.title("TrainMyAI")
        st.header("Select your favorite model")

        search_method = st.selectbox(
            "Select search method:",
            ["Select Model Type", "CNN", "ANN", "RNN", "LSTM", "GAN"],
            label_visibility="collapsed"
        )

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

        st.subheader("Upload Folder (zip file)")

        uploaded_file = st.file_uploader(
            "Upload your zip file containing data files",
            type='zip'
        )

        submit_button = st.button("Process Files")

        if submit_button:
            if not uploaded_file:
                st.error("Please upload a zip file")
            elif search_method == "Select Model Type" or selected_parent == "Select an option" or (selected_child is not None and selected_child == "Select an option"):
                st.error("Please make all required selections in the dropdown menus")
            else:
                st.success("Processing uploaded zip file")
                extract_zip(uploaded_file)
                st.session_state["process_completed"] = True

    if st.session_state["process_completed"]:
        st.header("Model Training")

        if search_method == "CNN":
            if selected_parent == "Image Classification":
                if selected_child == "Binary Classification":

                    binary_cl = binary_classification.Binary_Classification()

                    # Check if the model is already trained
                    if not st.session_state["model_trained"]:

                        # Use placeholders for temporary messages
                        preprocess_placeholder = st.empty()
                        split_placeholder = st.empty()
                        compile_placeholder = st.empty()
                        train_placeholder = st.empty()

                        st.write("Preprocessing the dataset...")
                        binary_cl.dataset_preprocessing()

                        st.write("Splitting the dataset...")
                        binary_cl.splitting_dataset()

                        st.write("Creating and compiling the model...")
                        binary_cl.model_creation()

                        st.write("Training the model...")
                        history = binary_cl.train_model(epochs=20)

                        # Clear temporary messages after training
                        preprocess_placeholder.empty()
                        split_placeholder.empty()
                        compile_placeholder.empty()
                        train_placeholder.empty()

                        # Save the model to session state
                        st.session_state["binary_cl"] = binary_cl
                        st.session_state["model_trained"] = True

                        # Extract metrics from the history object
                        training_logs = []
                        for epoch in range(len(history.history['loss'])):
                            log = {
                                "epoch": epoch + 1,
                                "steps": len(binary_cl.train),
                                "loss": history.history['loss'][epoch],
                                "accuracy": history.history['accuracy'][epoch],
                                "val_loss": history.history['val_loss'][epoch],
                                "val_accuracy": history.history['val_accuracy'][epoch],
                            }
                            training_logs.append(log)

                        st.session_state["training_logs"] = json.dumps(training_logs, indent=6)

                    # Use the trained model from session state
                    binary_cl = st.session_state["binary_cl"]

                    traning_log.logs(st)

                    # Display training metrics
                    st.subheader("Training Metrics")
                    col1, col2 = st.columns(2)

                    with col1:
                        st.pyplot(binary_cl.plot_loss())
                    with col2:
                        st.pyplot(binary_cl.plot_accuracy())

                    # Add "Inference" section
                    st.subheader("Inference")
                    st.write("Upload an image for inference:")

                    # Image uploader
                    uploaded_image = st.file_uploader("Choose an image", type=["jpeg", "jpg", "png", "bmp"], key="image_uploader")

                    # Submit button for inference
                    inference_button = st.button("Submit for Inference", key="inference_button")

                    if inference_button and uploaded_image is not None:
                        # Read the uploaded image file
                        img_bytes = uploaded_image.read()  # Get the binary content
                        img_array = tf.image.decode_image(img_bytes, channels=3)  # Decode to tensor
                        result = binary_cl.inference(img_array)  # Pass the tensor to the inference function

                        # Display the result in large, bold text
                        st.markdown(f"<h2 style='text-align: center; color: black;'>Prediction: {result.upper()}</h2>",
                                    unsafe_allow_html=True)

                        # Display the uploaded image
                        st.image(uploaded_image, caption="Uploaded Image for Inference", use_container_width=True)
                        # st.write(f"Prediction:  {result}")


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