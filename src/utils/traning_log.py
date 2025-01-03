def logs(st):
    """
    Displays training logs in a scrollable JSON-like format on the Streamlit app.

    Parameters:
    - st: Streamlit object for rendering the UI elements.

    Returns:
    - None. Outputs the training logs directly to the Streamlit interface.
    """
    st.subheader("Training Logs")
    st.markdown(
        """
        <style>
            .scrollable-json {
                max-height: 300px;
                overflow-y: auto;
                padding: 10px;
                border: 1px solid #ddd;
                border-radius: 5px;
                background-color: #f9f9f9;
                white-space: pre;
                font-family: monospace;
                line-height: 1.5;
                margin: 0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
            <div class="scrollable-json">{st.session_state["training_logs"]}</div>
            """,
        unsafe_allow_html=True,
    )