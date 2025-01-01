from src.utils import extract_file

def sidebar_ui(st):
    with st.sidebar:
        st.title("TrainMyAI")
        st.header("Select your favorite model")

        search_method = None
        selected_parent = None
        selected_child = None
        input_value = 20  # Default value for input_value

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

        # Add an input field for an integer value
        input_value = st.number_input(
            "Enter Num of epoch: Def:20",
            min_value=0,  # Optional: Set a minimum value
            max_value=1000,  # Optional: Set a maximum value
            value=20,  # Default value
            step=1  # Increment step
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
                # Reset session state variables for a fresh start
                st.session_state["process_completed"] = False
                st.session_state["model_trained"] = False
                st.session_state["binary_cl"] = None

                st.success("Processing uploaded zip file")
                extract_file.extract_zip(uploaded_file)
                st.session_state["process_completed"] = True

    return search_method, selected_parent, selected_child, input_value