from src.utils.lstm.text_generation import text_generation_model
import json
from src.utils import traning_log
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

def text_generation_cofig(st, input_value):
    """
    Configure the sentiment analysis model, including training and tracking the state.
    """

    if "model_trained" not in st.session_state:
        st.session_state["model_trained"] = False
    if "model_obj" not in st.session_state:
        st.session_state["model_obj"] = None
    if "training_logs" not in st.session_state:
        st.session_state["training_logs"] = ""

    text_generation_cl = text_generation_model.TextGenModel()

    if not st.session_state["model_trained"]:
        preprocess_placeholder = st.empty()
        split_placeholder = st.empty()
        compile_placeholder = st.empty()
        train_placeholder = st.empty()

        st.write("Dataset Reading...")
        text_generation_cl.read_data()

        st.write("Preprocess data...")
        text_generation_cl.preprocess()

        st.write("Model Building...")
        text_generation_cl.build_model()

        st.write("Model Traning...")
        text_generation_cl.train_model(epochs=input_value)

        history = text_generation_cl.history
        preprocess_placeholder.empty()
        split_placeholder.empty()
        compile_placeholder.empty()
        train_placeholder.empty()

        st.session_state["model_obj"] = text_generation_cl
        st.session_state["model_trained"] = True

        training_logs = []
        for epoch in range(input_value):
            log = {
                "epoch": epoch + 1,
                "accuracy": history.history['accuracy'][epoch],
                "loss": history.history['loss'][epoch],
            }
            training_logs.append(log)

        st.session_state["training_logs"] = json.dumps(training_logs, indent=6)

    text_generation_cl = st.session_state["model_obj"]

    if text_generation_cl:
        st.write("Training Logs")
        traning_log.logs(st)

        st.subheader("Training Metrics")
        loss_image = text_generation_cl.plot_loss()
        accuracy_image = text_generation_cl.plot_accuracy()

        col1, col2 = st.columns(2)
        with col1:
            st.pyplot(loss_image)
        with col2:
            st.pyplot(accuracy_image)

        st.subheader("Download Trained Model")
        download_option = st.radio("Do you want to download the trained model?", ("No", "Yes"))

        if download_option == "Yes":
            import io
            import h5py

            model_buffer = io.BytesIO()

            with h5py.File(model_buffer, 'w') as f:
                text_generation_cl.model.save(f)

            model_buffer.seek(0)

            st.download_button(
                label="Download Model as .pth",
                data=model_buffer,
                file_name="trained_model.pth",
                mime="application/octet-stream"
            )

        st.subheader("Inference")
        user_text_input = st.text_input("Enter your text:", key="user_input")

        # Display the number input with a default value and help text.
        user_num_words = st.number_input(
            "Number of words to generate:",
            min_value=1,
            max_value=100,
            value=10,
            step=1,
            key="num_words",
            help="Default value is 10. Adjust to generate more or fewer words."
        )

        inference_button = st.button("Text Generation", key="inference_button")
        if inference_button and user_text_input:
            predictions = text_generation_cl.predict_next_words(user_text_input, int(user_num_words))
            for prediction in predictions:
                st.write(prediction)