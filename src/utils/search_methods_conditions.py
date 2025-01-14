from src.utils.cnn.image_classification import binary_classification_ui, multiclass_classfication_ui
from src.utils.cnn.object_detection import object_detection_ui
from src.utils.cnn.face_recognition import face_recognition_ui
from src.utils.cnn.style_transfor import style_transfor_ui
from src.utils.ann.dataset_prediction import dataset_prediction_ui
from src.utils.rnn.sentiment_analysis import sentiment_analysis_ui
from src.utils.rnn.text_generation import text_generation_ui

def medthos_traning(st, search_method, selected_parent, selected_child, input_value):
    """
    Executes a training workflow based on the specified search method, parent category, and child category.

    Parameters:
    - st: Streamlit session object for managing UI and states.
    - search_method: The type of machine learning method (e.g., CNN, ANN, RNN, LSTM, GAN).
    - selected_parent: The main category selected by the user (e.g., Image Classification, Object Detection).
    - selected_child: The subcategory selected by the user (e.g., Binary Classification, Multi Class Classification).
    - input_value: Additional input parameter, such as the number of epochs or other training configurations.

    Returns:
    - None. Displays relevant UI elements and logs training information based on the selected configurations.
    """
    if search_method == "CNN":
        if selected_parent == "Image Classification":
            if selected_child == "Binary Classification":
                binary_classification_ui.binary_classification_cofig(st, input_value)

            elif selected_child == "Multi Class Classification":
                multiclass_classfication_ui.multiclass_classification_cofig(st, input_value)
        elif selected_parent == "Object Detection":
            object_detection_ui.object_detection_cofig(st, input_value)
        elif selected_parent == "Face Recognition":
            face_recognition_ui.face_recognition_cofig(st, input_value)
        elif selected_parent == "Style Transfer":
            style_transfor_ui.style_transfor_cofig(st, input_value)

    elif search_method == "ANN":
        if selected_parent == "Dataset Prediction":
            dataset_prediction_ui.dataset_prediction_cofig(st, input_value)

    elif search_method == "RNN":
        if selected_parent == "Sentiment Analysis":
            sentiment_analysis_ui.sentiment_analysis_cofig(st, input_value)
        elif selected_parent == "Text Generation":
            text_generation_ui.text_generation_cofig(st, input_value)
        elif selected_parent == "Machine Translation":
            st.info("RNN ---> Machine Translation")

    elif search_method == "LSTM":
        if selected_parent == "Text Summarization":
            st.info("LSTM ---> Text Summarization")
        elif selected_parent == "Language Modeling":
            st.info("LSTM ---> Language Modeling")

    elif search_method == "GAN":
        if selected_parent == "Image Generation":
            st.info("GAN ---> Image Generation")
        elif selected_parent == "Image Super-Resolution":
            st.info("GAN ---> Image Super-Resolution")
        elif selected_parent == "Facial Image Generation":
            st.info("GAN ---> Facial Image Generation")
        elif selected_parent == "Text-to-Image Generation":
            st.info("GAN ---> Text-to-Image Generation")