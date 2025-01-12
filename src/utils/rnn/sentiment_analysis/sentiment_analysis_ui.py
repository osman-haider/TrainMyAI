from src.utils.rnn.sentiment_analysis import sentiment_analysis_model
from src.utils.cnn import plots
import tensorflow as tf
import json
from src.utils import traning_log
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def sentiment_analysis_cofig(st, input_value):
    """
    Configures the dataset prediction process, including data preprocessing, model training,
    evaluation, and inference. It also handles session state management and user interactions
    for training and downloading the model.
    """

    if "model_trained" not in st.session_state:
        st.session_state["model_trained"] = False
    if "model_obj" not in st.session_state:
        st.session_state["model_obj"] = None
    if "training_logs" not in st.session_state:
        st.session_state["training_logs"] = ""

    sentiment_analysis_cl = sentiment_analysis_model.SentimentAnalysis()

    if not st.session_state["model_trained"]:
        preprocess_placeholder = st.empty()
        split_placeholder = st.empty()
        compile_placeholder = st.empty()
        train_placeholder = st.empty()

        st.write("Dataset Reading...")
        sentiment_analysis_cl.load_data()

        st.write("Initialize the model...")
        sentiment_analysis_cl.init_model()

        st.write("Training the model...")
        sentiment_analysis_cl.train(epochs=input_value)
        history = dataset_prediction_cl.history
        preprocess_placeholder.empty()
        split_placeholder.empty()
        compile_placeholder.empty()
        train_placeholder.empty()

        st.session_state["model_obj"] = dataset_prediction_cl
        st.session_state["model_trained"] = True

        training_logs = []
        for epoch in range(len(history.history['loss'])):
            log = {
                "epoch": epoch + 1,
                "loss": history.history['loss'][epoch],
                "val_loss": history.history['val_loss'][epoch],
            }
            training_logs.append(log)

        st.session_state["training_logs"] = json.dumps(training_logs, indent=6)

    dataset_prediction_cl = st.session_state["model_obj"]

    if dataset_prediction_cl:
        """
        Displays training logs, evaluation metrics, and plots. It also provides an option 
        to download the trained model and perform inference on the dataset.
        """

        st.write("Training Logs")
        traning_log.logs(st)

        evaluate_model_matrics = dataset_prediction_cl.evaluate_model()

        st.subheader("Evaluation Metrics Training Logs")
        st.session_state["training_logs"] = evaluate_model_matrics
        traning_log.logs(st)

        st.subheader("Training Metrics")
        image = dataset_prediction_cl.plot_training_loss()

        st.image(image, caption="Training Loss Per Epoch", use_container_width=True)

        st.subheader("Download Trained Model")
        download_option = st.radio("Do you want to download the trained model?", ("No", "Yes"))

        if download_option == "Yes":
            import io
            import h5py

            model_buffer = io.BytesIO()

            with h5py.File(model_buffer, 'w') as f:
                dataset_prediction_cl.model.save(f)

            model_buffer.seek(0)

            st.download_button(
                label="Download Model as .h5",
                data=model_buffer,
                file_name="trained_model.h5",
                mime="application/octet-stream"
            )

        st.subheader("Inference")
        inference_button = st.button("Inference", key="inference_button")
        if inference_button:
            predicted_dataset = dataset_prediction_cl.prediction()
            st.dataframe(predicted_dataset)