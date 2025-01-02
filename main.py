import streamlit as st
from src.utils import side_bar, search_methods_conditions

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main():
    """
    Main function to handle the Streamlit app logic.
    Includes dataset processing, model training, and inference functionality.
    """

    # Initialize session state variables
    if "process_completed" not in st.session_state:
        st.session_state["process_completed"] = False

    if "model_trained" not in st.session_state:
        st.session_state["model_trained"] = False

    if "binary_cl" not in st.session_state:
        st.session_state["binary_cl"] = None

    search_method, selected_parent, selected_child, input_value = side_bar.sidebar_ui(st)

    if st.session_state["process_completed"]:
        st.header("Model Training")

        search_methods_conditions.medthos_traning(st, search_method, selected_parent, selected_child, input_value)

    else:
        st.info("Please complete the file upload and selections to start model training.")

if __name__ == "__main__":
    st.set_page_config(
        page_title="TrainMyAI",
        layout="wide"
    )
    main()