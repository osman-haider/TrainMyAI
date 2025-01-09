from src.utils.ann.dataset_prediction import dataset_prediction_model
from src.utils.cnn import plots
import tensorflow as tf
import json
from src.utils import traning_log

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)


def dataset_prediction_cofig(st, input_value):
    """
    This function configures and trains a face recognition model using the specified input parameters.

    Parameters:
    - st: The Streamlit session object for handling session states and UI updates.
    - input_value: The number of epochs for training the model.

    Returns:
    - None. The function updates the Streamlit session state with the trained model and training logs.
    """
    # Initialize session state keys
    if "model_trained" not in st.session_state:
        st.session_state["model_trained"] = False
    if "model_obj" not in st.session_state:
        st.session_state["model_obj"] = None
    if "training_logs" not in st.session_state:
        st.session_state["training_logs"] = ""

    dataset_prediction_cl = dataset_prediction_model.DatasetPrediction()

    if not st.session_state["model_trained"]:
        preprocess_placeholder = st.empty()
        split_placeholder = st.empty()
        compile_placeholder = st.empty()
        train_placeholder = st.empty()

        st.write("Dataset Reading...")
        data_head = dataset_prediction_cl.get_data_head()

        # Display the head of the data
        st.write("Data Head:")
        st.dataframe(data_head)

        # Extract column names
        column_names = data_head.columns.tolist()

        # Dropdown for multiple column selection
        drop_columns_name = st.multiselect("Select all columns you want to drop:", column_names)

        # Filter columns for single-column dropdown
        filtered_columns = [col for col in column_names if col not in drop_columns_name]

        # Dropdown for single column selection
        output_column = st.selectbox(
            "Select column you want to make a prediction on it",
            options=["Select a column"] + filtered_columns
        )

        if drop_columns_name and output_column != "Select a column":
            st.write("Dataset in preprocessing...")
            dataset_prediction_cl.preprocess(drop_columns=drop_columns_name, target_column=output_column)

            st.write("Splitting the dataset...")
            dataset_prediction_cl.split_data(output_column)

            st.write("Creating the model...")
            dataset_prediction_cl.build_model()

            st.write("Training the model...")
            dataset_prediction_cl.train_model(epochs=input_value)
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
