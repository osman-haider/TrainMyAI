from src.utils import extract_file

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def sidebar_ui(st):
    with st.sidebar:
        st.title("TrainMyAI")
        st.header("Select your favorite model")

        search_method = None
        selected_parent = None
        selected_child = None

        search_method = st.selectbox(
            "Select search method:",
            ["Select Model Type", "CNN", "ANN", "RNN", "LSTM", "GAN"],
            label_visibility="collapsed"
        )

        if search_method == "CNN":
            parent_options = ["Image Classification", "Object Detection", "Image Segmentation", "Face Recognition", "Style Transfer"]
            selected_parent = st.selectbox(
                "Select type of Main Model:",
                ["Select an option"] + parent_options,
                label_visibility="collapsed"
            )

            if selected_parent == "Image Classification":
                child_options = ["Binary Classification", "Multi Class Classification"]
                selected_child = st.selectbox(
                    "Select type of Classification Model:",
                    ["Select an option"] + child_options,
                    label_visibility="collapsed"
                )

        elif search_method == "ANN":
            parent_options = ["Dataset Prediction", "Credit Card Fraud Detection"]
            selected_parent = st.selectbox(
                "Select type of Main Model:",
                ["Select an option"] + parent_options,
                label_visibility="collapsed"
            )

        elif search_method == "RNN":
            parent_options = ["Sentiment Analysis", "Time Series Forecasting", "Text Generation", "Machine Translation"]
            selected_parent = st.selectbox(
                "Select type of Main Model:",
                ["Select an option"] + parent_options,
                label_visibility="collapsed"
            )

        elif search_method == "LSTM":
            parent_options = ["Text Summarization", "Language Modeling", "Time Series Forecasting", "Dataset Prediction"]
            selected_parent = st.selectbox(
                "Select type of Main Model:",
                ["Select an option"] + parent_options,
                label_visibility="collapsed"
            )

        elif search_method == "GAN":
            parent_options = ["Image Generation", "Image Super-Resolution", "Facial Image Generation", "Text-to-Image Generation"]
            selected_parent = st.selectbox(
                "Select type of Main Model:",
                ["Select an option"] + parent_options,
                label_visibility="collapsed"
            )

        st.subheader("Upload Folder (zip file)")

        uploaded_file = st.file_uploader(
            "Upload your zip file containing data files",
            type='zip'
        )

        submit_button = st.button("Process Files")

        if submit_button:
            if not uploaded_file:
                st.error("Please upload a zip file")
            elif search_method == "Select Model Type" or selected_parent == "Select an option" or (selected_child is not None and selected_child == "Select an option"):
                st.error("Please make all required selections in the dropdown menus")
            else:
                st.success("Processing uploaded zip file")
                extract_file.extract_zip(uploaded_file)
                st.session_state["process_completed"] = True
    return search_method, selected_parent, selected_child