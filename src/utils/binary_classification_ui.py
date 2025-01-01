from src.utils.cnn.image_classification import binary_classification_model_creation
import tensorflow as tf
import json
from src.utils import traning_log

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def binary_classification_cofig(st):
    binary_cl = binary_classification_model_creation.Binary_Classification()

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

    # Provide an option to download the trained model
    st.subheader("Download Trained Model")
    download_option = st.radio("Do you want to download the trained model?", ("No", "Yes"))

    if download_option == "Yes":
        # Save the model to a buffer as .h5
        import io
        import h5py

        model_buffer = io.BytesIO()

        # Save the model to the buffer using h5py
        with h5py.File(model_buffer, 'w') as f:
            binary_cl.model.save(f)

        model_buffer.seek(0)  # Reset the buffer pointer

        # Provide a download button dynamically
        st.download_button(
            label="Download Model as .h5",
            data=model_buffer,
            file_name="trained_model.h5",
            mime="application/octet-stream"
        )

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

