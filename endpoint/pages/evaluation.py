import pandas as pd
import plotly.express as px
import streamlit as st

# Sayfa Ayarları
st.set_page_config(
    page_title="Hotel Cancellations",
    page_icon="images/hotel-service.png",
    layout="wide",
    menu_items={
        "Get help": "mailto:ata.ozarslan@istdsa.com",
        "About": "For More Information\n" + "https://istdatascience.com/"
    }
)

# -----------------------------------------------------------------------

conf_matrix = pd.DataFrame([[4506, 333], [469, 1947]],
                           columns=["Not Canceled", "Canceled"],
                           index=["Not Canceled", "Canceled"])

conf_matrix = px.imshow(
    conf_matrix,
    text_auto=True,
    aspect="auto",
    color_continuous_scale=["red","yellow","green"],
    labels={"color": "Count"}
).update_layout(title="Confusion Matrix")

conf_matrix.update_xaxes(title_text="Prediction")
conf_matrix.update_yaxes(title_text="Actual")

st.plotly_chart(conf_matrix, use_container_width=True)

# -----------------------------------------------------------------------

st.markdown("**Evaluation Metrics**")

column_names = ["Customer Group", "Accuracy", "Recall", "Precision", "F1 Score", "AUC Score"]

metric0 = pd.DataFrame(["Not Canceled" ,0.889, 0.931, 0.906, 0.918, 0.869]).T
metric0.columns = column_names

metric1 = pd.DataFrame(["Canceled", 0.889, 0.806, 0.854, 0.829, 0.869]).T
metric1.columns = column_names

st.table(metric0)
st.table(metric1)

# -----------------------------------------------------------------------

st.markdown("**Model Explanation**")

st.image("images/shap_bar_plot.png", caption="Shap Feature Importances", use_column_width="always")
st.image("images/shap_summary_plot.png", caption="Shap Summary Plot", use_column_width="always")