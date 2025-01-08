from src.utils.cnn import plots
from src.utils.cnn.style_transfor import style_transfor_model
import json
from src.utils import traning_log
import os
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

"""
This function handles the configuration and training of the style transformer model, as well as
performing inference and allowing users to download the trained model. It includes uploading
content and style images, preprocessing data, training the model, and generating stylized images.
"""
def style_transfor_cofig(st, input_value):
    """
    Configure and train the style transformer model.

    Parameters:
    - st: Streamlit session object for UI updates.
    - input_value: Number of epochs for training.
    """
    style_transfor_cl = style_transfor_model.StyleTransformer()

    if not st.session_state["model_trained"]:
        preprocess_placeholder = st.empty()
        split_placeholder = st.empty()
        compile_placeholder = st.empty()
        train_placeholder = st.empty()

        """
        Preprocess the dataset and prepare content and style images for training.
        """
        st.write("data preprocessing...")
        content_images, style_images = style_transfor_cl.prepare_dataset()

        """
        Train the model with the specified number of epochs.
        """
        st.write("Training the model...")
        style_transfor_cl.train(input_value)

        preprocess_placeholder.empty()
        split_placeholder.empty()
        compile_placeholder.empty()
        train_placeholder.empty()

        """
        Save the trained model and log training metrics.
        """
        st.session_state["model_obj"] = style_transfor_cl
        st.session_state["model_trained"] = True

        training_logs = []
        for epoch in range(input_value):
            log = {
                "epoch": epoch + 1,
                "content loss": style_transfor_cl.content_losses[epoch],
                "style loss": style_transfor_cl.style_losses[epoch],
            }
            training_logs.append(log)

        st.session_state["training_logs"] = json.dumps(training_logs, indent=6)

    style_transfor_cl = st.session_state["model_obj"]

    """
    Display training logs and metrics.
    """
    traning_log.logs(st)

    st.subheader("Training Metrics")

    plot = style_transfor_cl.plot_losses()
    st.pyplot(plot)

    """
    Allow the user to download the trained model in HDF5 format.
    """
    st.subheader("Download Trained Model")
    download_option = st.radio("Do you want to download the trained model?", ("No", "Yes"))

    if download_option == "Yes":
        import io

        model_buffer = io.BytesIO()

        style_transfor_cl.transformer.save('model.h5')

        model_buffer.seek(0)

        st.download_button(
            label="Download Model as .h5",
            data=model_buffer,
            file_name="trained_model.h5",
            mime="application/octet-stream"
        )

    """
    Inference Section: Allow users to upload content and style images for generating stylized images.
    """
    st.subheader("Style Transfer Inference")
    st.write("Upload a content image and a style image:")

    content_uploaded_image = st.file_uploader("Choose a Content Image", type=["jpeg", "jpg", "png"],
                                              key="content_uploader")
    style_uploaded_image = st.file_uploader("Choose a Style Image", type=["jpeg", "jpg", "png"], key="style_uploader")

    inference_button = st.button("Submit for Inference", key="inference_button")

    if inference_button and content_uploaded_image and style_uploaded_image:
        """
        Save uploaded images to respective directories and perform style transfer.
        """
        content_img_dir = "extracted_folder/Inference_Content"
        style_img_dir = "extracted_folder/Inference_Style"
        Path(content_img_dir).mkdir(parents=True, exist_ok=True)
        Path(style_img_dir).mkdir(parents=True, exist_ok=True)

        content_img_path = os.path.join(content_img_dir, content_uploaded_image.name)
        with open(content_img_path, "wb") as f:
            f.write(content_uploaded_image.getbuffer())

        style_img_path = os.path.join(style_img_dir, style_uploaded_image.name)
        with open(style_img_path, "wb") as f:
            f.write(style_uploaded_image.getbuffer())

        stylized_image_fig = style_transfor_cl.stylize_image(content_img_path, style_img_path)

        st.pyplot(stylized_image_fig)