from src.utils import extract_file
import os
import shutil

def sidebar_ui(st):
    """
    Creates a sidebar UI for the TrainMyAI application to facilitate model selection, parameter input,
    and file upload for processing.

    Parameters:
    - st: Streamlit object for handling the sidebar UI and session states.

    Returns:
    - Tuple containing:
        - search_method: Selected search method (e.g., CNN, ANN).
        - selected_parent: Selected main model type (e.g., Image Classification).
        - selected_child: Selected sub-model type, if applicable (e.g., Binary Classification).
        - input_value: Number of epochs input by the user.
    """
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
            parent_options = ["Image Classification", "Object Detection", "Face Recognition", "Style Transfer"]
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
            parent_options = ["Dataset Prediction"]
            selected_parent = st.selectbox(
                "Select type of Main Model:",
                ["Select an option"] + parent_options,
                label_visibility="collapsed"
            )

        elif search_method == "RNN":
            parent_options = ["Sentiment Analysis", "Text Generation", "Machine Translation"]
            selected_parent = st.selectbox(
                "Select type of Main Model:",
                ["Select an option"] + parent_options,
                label_visibility="collapsed"
            )

        elif search_method == "LSTM":
            parent_options = ["Language Modeling"]
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
                st.session_state["process_completed"] = False
                st.session_state["model_trained"] = False
                st.session_state["model_obj"] = None

                extracted_folder_path = "extracted_folder"
                if os.path.exists(extracted_folder_path) and os.path.isdir(extracted_folder_path):
                    for filename in os.listdir(extracted_folder_path):
                        file_path = os.path.join(extracted_folder_path, filename)
                        try:
                            if os.path.isfile(file_path) or os.path.islink(file_path):
                                os.unlink(file_path)
                            elif os.path.isdir(file_path):
                                shutil.rmtree(file_path)
                        except Exception as e:
                            st.error(f"Failed to delete {file_path}. Reason: {e}")

                st.success("Processing uploaded zip file")
                extract_file.extract_zip(uploaded_file)
                st.session_state["process_completed"] = True

    return search_method, selected_parent, selected_child, input_value