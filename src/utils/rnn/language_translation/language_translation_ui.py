from src.utils.rnn.language_translation import language_translation_model
import json
from src.utils import traning_log
import warnings
from src.utils.cnn import plots

warnings.filterwarnings("ignore", category=DeprecationWarning)

def language_translation_cofig(st, input_value):


    if "model_trained" not in st.session_state:
        st.session_state["model_trained"] = False
    if "model_obj" not in st.session_state:
        st.session_state["model_obj"] = None
    if "training_logs" not in st.session_state:
        st.session_state["training_logs"] = ""

    language_translation_cl = language_translation_model.LanguageTranslationModel()

    if not st.session_state["model_trained"]:
        preprocess_placeholder = st.empty()
        split_placeholder = st.empty()
        compile_placeholder = st.empty()
        train_placeholder = st.empty()

        st.write("Dataset Reading & perprocessing...")
        language_translation_cl.preprocess()

        st.write("Initialize the model...")
        language_translation_cl.build_model()

        st.write("Training the model...")
        language_translation_cl.train(epochs=input_value)

        history = language_translation_cl.history
        preprocess_placeholder.empty()
        split_placeholder.empty()
        compile_placeholder.empty()
        train_placeholder.empty()

        st.session_state["model_obj"] = language_translation_cl
        st.session_state["model_trained"] = True

        training_logs = []
        for epoch in range(input_value):
            log = {
                "epoch": epoch + 1,
                "loss": history.history['loss'][epoch],
                "val_loss": history.history['val_loss'][epoch],
                "accuracy": history.history['accuracy'][epoch],
                "val_accuracy": history.history['val_accuracy'][epoch],
            }
            training_logs.append(log)

        st.session_state["training_logs"] = json.dumps(training_logs, indent=6)

    language_translation_cl = st.session_state["model_obj"]

    if language_translation_cl:
        st.write("Training Logs")
        traning_log.logs(st)

        st.subheader("Training Metrics")
        col1, col2 = st.columns(2)
        plot = plots.training_metrics(language_translation_cl.history)
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
                language_translation_cl.model.save(f)

            model_buffer.seek(0)

            st.download_button(
                label="Download Model as .h5",
                data=model_buffer,
                file_name="trained_model.h5",
                mime="application/octet-stream"
            )

        st.subheader("Inference")
        user_input = st.text_input("Enter your message:", key="user_input")

        inference_button = st.button("Translate Text", key="inference_button")
        if inference_button and user_input:
            prediction_text = language_translation_cl.predict(user_input)

            st.write("Original Text: ", user_input)
            st.write("Predicted Text: ", prediction_text)