from src.utils.cnn import plots
from src.utils.cnn.style_transfor import style_transfor_model
import json
from src.utils import traning_log
import os
from pathlib import Path

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def style_transfor_cofig(st, input_value):
    """
    This function configures and trains a object detection model using the specified input parameters.

    Parameters:
    - st: The Streamlit session object for handling session states and UI updates.
    - input_value: The number of epochs for training the model.

    Returns:
    - None. The function updates the Streamlit session state with the trained model and training logs.
    """
    style_transfor_cl = style_transfor_model.StyleTransformer()

    if not st.session_state["model_trained"]:
        preprocess_placeholder = st.empty()
        split_placeholder = st.empty()
        compile_placeholder = st.empty()
        train_placeholder = st.empty()

        st.write("data preprocessing...")
        content_images, style_images = style_transfor_cl.prepare_dataset()

        st.write("Training the model...")
        style_transfor_cl.train(input_value)

        preprocess_placeholder.empty()
        split_placeholder.empty()
        compile_placeholder.empty()
        train_placeholder.empty()

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

    traning_log.logs(st)

    st.subheader("Training Metrics")

    plot = style_transfor_cl.plot_losses()
    st.pyplot(plot)

    st.subheader("Download Trained Model")
    download_option = st.radio("Do you want to download the trained model?", ("No", "Yes"))

    if download_option == "Yes":
        import io

        model_buffer = io.BytesIO()

        # Save the model to the buffer in HDF5 format
        style_transfor_cl.model.save('model.h5')

        model_buffer.seek(0)

        st.download_button(
            label="Download Model as .h5",
            data=model_buffer,
            file_name="trained_model.h5",
            mime="application/octet-stream"
        )

    # Inference Section
    st.subheader("Style Transfer Inference")
    st.write("Upload a content image and a style image:")

    # File uploaders for content and style images
    content_uploaded_image = st.file_uploader("Choose a Content Image", type=["jpeg", "jpg", "png"],
                                              key="content_uploader")
    style_uploaded_image = st.file_uploader("Choose a Style Image", type=["jpeg", "jpg", "png"], key="style_uploader")

    # Button to trigger the inference process
    inference_button = st.button("Submit for Inference", key="inference_button")

    if inference_button and content_uploaded_image and style_uploaded_image:
        # Directories to save the uploaded images
        content_img_dir = "extracted_folder/Inference_Content"
        style_img_dir = "extracted_folder/Inference_Style"
        Path(content_img_dir).mkdir(parents=True, exist_ok=True)  # Create the content folder if it doesn't exist
        Path(style_img_dir).mkdir(parents=True, exist_ok=True)  # Create the style folder if it doesn't exist

        # Save the uploaded content image
        content_img_path = os.path.join(content_img_dir, content_uploaded_image.name)
        with open(content_img_path, "wb") as f:
            f.write(content_uploaded_image.getbuffer())

        # Save the uploaded style image
        style_img_path = os.path.join(style_img_dir, style_uploaded_image.name)
        with open(style_img_path, "wb") as f:
            f.write(style_uploaded_image.getbuffer())

        # Perform style transfer
        stylized_image_fig = style_transfor_cl.stylize_image(content_img_path, style_img_path)

        # Display the resulting stylized image
        st.pyplot(stylized_image_fig)
