from src.utils.cnn.image_classification import multiclass_classification_model, plots
import tensorflow as tf
import json
from src.utils import traning_log

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def multiclass_classification_cofig(st, input_value):
    """
    This function configures and trains a binary classification model using the specified input parameters.

    Parameters:
    - st: The Streamlit session object for handling session states and UI updates.
    - input_value: The number of epochs for training the model.

    Returns:
    - None. The function updates the Streamlit session state with the trained model and training logs.
    """
    multiclass_cl = multiclass_classification_model.multiclass_classifiction()

    if not st.session_state["model_trained"]:
        preprocess_placeholder = st.empty()
        split_placeholder = st.empty()
        compile_placeholder = st.empty()
        train_placeholder = st.empty()

        st.write("train dataset in creating...")
        multiclass_cl.train_data_generator()

        st.write("val dataset in creating...")
        multiclass_cl.val_data_generator()

        st.write("Creating the model...")
        multiclass_cl.cnn_model_buliding()

        st.write("Training the model...")
        history = multiclass_cl.model_fit(epochs=input_value)

        preprocess_placeholder.empty()
        split_placeholder.empty()
        compile_placeholder.empty()
        train_placeholder.empty()

        st.session_state["model_obj"] = multiclass_cl
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

    multiclass_cl = st.session_state["model_obj"]

    traning_log.logs(st)

    st.subheader("Training Metrics")
    col1, col2 = st.columns(2)
    plot = plots.training_metrics(multiclass_cl.history)
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

        with h5py.File(model_buffer, 'w') as f:
            multiclass_cl.model.save(f)

        model_buffer.seek(0)

        st.download_button(
            label="Download Model as .h5",
            data=model_buffer,
            file_name="trained_model.h5",
            mime="application/octet-stream"
        )

    st.subheader("Inference")
    st.write("Upload an image for inference:")

    uploaded_image = st.file_uploader("Choose an image", type=["jpeg", "jpg", "png", "bmp"], key="image_uploader")

    inference_button = st.button("Submit for Inference", key="inference_button")

    if inference_button and uploaded_image is not None:
        img_bytes = uploaded_image.read()
        img_tensor = tf.image.decode_image(img_bytes, channels=3)  # Decode the image to a tensor
        result = multiclass_cl.inference(img_tensor)
        st.write(f"Predicted class: {result}")

        st.markdown(f"<h2 style='text-align: center; color: black;'>Prediction: {result.upper()}</h2>",
                    unsafe_allow_html=True)

        st.image(uploaded_image, caption="Uploaded Image for Inference", use_container_width=True)