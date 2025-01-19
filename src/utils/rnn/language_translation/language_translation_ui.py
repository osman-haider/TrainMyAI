from src.utils.rnn.language_translation import language_translation_model
import json
from src.utils import traning_log
import warnings

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

    sentiment_analysis_cl = st.session_state["model_obj"]

    if sentiment_analysis_cl:
        st.write("Training Logs")
        traning_log.logs(st)

        st.subheader("Training & Val Metrics")
        image = sentiment_analysis_cl.plot_losses()

        st.image(image, caption="Training Loss Per Epoch", use_container_width=True)

        st.subheader("Download Trained Model")
        download_option = st.radio("Do you want to download the trained model?", ("No", "Yes"))

        if download_option == "Yes":
            import io
            import torch

            model_buffer = io.BytesIO()
            torch.save(sentiment_analysis_cl.net.state_dict(), model_buffer)
            model_buffer.seek(0)

            st.download_button(
                label="Download Model as .pth",
                data=model_buffer,
                file_name="trained_model.pth",
                mime="application/octet-stream"
            )

        st.subheader("Inference")
        user_input = st.text_input("Enter your message:", key="user_input")

        inference_button = st.button("Analyze Sentiment", key="inference_button")
        if inference_button and user_input:
            predicted_sentiment = sentiment_analysis_cl.predict(user_input)
            prediction_text = None
            if predicted_sentiment == 1:
                prediction_text = "Positive"
            else:
                prediction_text = "Negative"

            st.write("Message: ", user_input)
            st.write("Predicted Sentiment: ", prediction_text)