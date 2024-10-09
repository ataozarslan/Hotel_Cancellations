import streamlit as st

pages = st.navigation({
    "Project": [
        st.Page("pages/project.py", title="Description", icon="💻"),
        st.Page("pages/dataset.py", title="About Data", icon="📁"),
        st.Page("pages/charts.py", title="Charts", icon="📊")
    ],
    "Modeling": [
        st.Page("pages/evaluation.py", title="Evaluation", icon="🔍"),
        st.Page("pages/predict.py", title="Prediction", icon="🤖")
    ],
})

pages.run()