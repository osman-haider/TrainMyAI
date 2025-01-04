from src.utils.cnn.image_classification import plots
from src.utils.cnn.object_detection import object_detection_model
import tensorflow as tf
import json
from src.utils import traning_log
import os
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def object_detection_cofig(st, input_value):
    """
    This function configures and trains a binary classification model using the specified input parameters.

    Parameters:
    - st: The Streamlit session object for handling session states and UI updates.
    - input_value: The number of epochs for training the model.

    Returns:
    - None. The function updates the Streamlit session state with the trained model and training logs.
    """
    object_detection_cl = object_detection_model.object_detection()

    if not st.session_state["model_trained"]:
        preprocess_placeholder = st.empty()
        split_placeholder = st.empty()
        compile_placeholder = st.empty()
        train_placeholder = st.empty()

        st.write("data preprocessing...")
        object_detection_cl.preprocess_dataset()

        st.write("train dataset in creating...")
        object_detection_cl.train_dataset_method()

        st.write("val dataset in creating...")
        object_detection_cl.val_dataset_method()

        st.write("Creating the model...")
        object_detection_cl.model_creation()

        st.write("Training the model...")
        object_detection_cl.train_model(input_value)

        history = object_detection_cl.history

        preprocess_placeholder.empty()
        split_placeholder.empty()
        compile_placeholder.empty()
        train_placeholder.empty()

        st.session_state["model_obj"] = object_detection_cl
        st.session_state["model_trained"] = True

        training_logs = []
        for epoch in range(len(history.history['loss'])):
            log = {
                "epoch": epoch + 1,
                "loss": history.history['loss'][epoch],
                "accuracy": history.history['accuracy'][epoch],
                "val_loss": history.history['val_loss'][epoch],
                "val_accuracy": history.history['val_accuracy'][epoch],
            }
            training_logs.append(log)

        st.session_state["training_logs"] = json.dumps(training_logs, indent=6)

    object_detection_cl = st.session_state["model_obj"]

    traning_log.logs(st)

    st.subheader("Training Metrics")
    col1, col2 = st.columns(2)
    plot = plots.training_metrics(object_detection_cl.history)
    with col1:
        st.pyplot(plot.plot_loss())
    with col2:
        st.pyplot(plot.plot_accuracy())

    st.subheader("Download Trained Model")
    download_option = st.radio("Do you want to download the trained model?", ("No", "Yes"))

    if download_option == "Yes":
        import io
        import h5py

        model_buffer = io.BytesIO()

        # Save the model to the buffer in HDF5 format
        object_detection_cl.model.save('model.h5')

        model_buffer.seek(0)

        st.download_button(
            label="Download Model as .h5",
            data=model_buffer,
            file_name="trained_model.h5",
            mime="application/octet-stream"
        )

    # Inference Section
    st.subheader("Inference")
    st.write("Upload an image for inference:")

    uploaded_image = st.file_uploader("Choose an image", type=["jpeg", "jpg", "png", "bmp"], key="image_uploader")

    inference_button = st.button("Submit for Inference", key="inference_button")

    if inference_button and uploaded_image is not None:
        # Define the directory to save the image
        save_dir = "extracted_folder"
        Path(save_dir).mkdir(parents=True, exist_ok=True)  # Create the folder if it doesn't exist

        # Save the uploaded image to the "extracted" folder
        image_path = os.path.join(save_dir, uploaded_image.name)

        # Write the file
        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())

        img_with_bbox = object_detection_cl.inference(image_path)

        col1, col2 = st.columns(2)


        with col1:
            st.markdown("<div style='height:57px;'></div>", unsafe_allow_html=True)  # Add 30px vertical space
            st.image(uploaded_image, caption="Uploaded Image", width=300)

        with col2:
            st.image(img_with_bbox, caption="Inference Result", width=300)  # Same fixed width

