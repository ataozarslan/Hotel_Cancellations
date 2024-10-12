import sqlite3
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
import streamlit as st
from joblib import load
from datetime import date, datetime
import shap
import pickle
from io import BytesIO

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

#---------------------------------------------------------------------------------------------------------------------

st.header("🛜 Online Prediction")
st.markdown("**Choose** the features to see the prediction!")

# Sidebarda Kullanıcıdan Girdileri Alma
name = st.text_input("Name", help="Please capitalize the first letter of your name!")
surname = st.text_input("Surname", help="Please capitalize the first letter of your surname!")
arrival_date = st.date_input("Arrival Date", format="DD/MM/YYYY")
no_of_weekend_nights = st.slider("Weekend Nights", min_value=0, max_value=10)
no_of_week_nights = st.slider("Weekday Nights", min_value=0, max_value=50)
market_segment_type = st.selectbox("Booking Type", ['Other','Online'], index=1)
no_of_adults =  st.number_input("Adults", min_value=0)
lead_time = st.number_input("Lead Time", min_value=1, format="%d")
no_of_special_requests = st.number_input("Number of Special Requests", min_value=0)
avg_price_per_room = st.number_input("Average Room Price", min_value=0)

#---------------------------------------------------------------------------------------------------------------------

# Pickle kütüphanesi kullanarak eğitilen modelin tekrardan kullanılması
xgb_model = load('hotel_cancellation_model.pkl')

arrival_day = arrival_date.day
arrival_month = arrival_date.month
arrival_year = arrival_date.year

if market_segment_type == 'Online':
    market_segment_type = 1
else:
    market_segment_type = 0

input_data = {
    "lead_time": lead_time,
    "no_of_special_requests": no_of_special_requests,
    "avg_price_per_room": avg_price_per_room,
    "market_segment_type_Online": market_segment_type,
    "arrival_month": arrival_month,
    "arrival_date": arrival_day,
    "arrival_year": arrival_year,
    "no_of_weekend_nights": no_of_weekend_nights,
    "no_of_week_nights": no_of_week_nights,
    "no_of_adults": no_of_adults
}

#---------------------------------------------------------------------------------------------------------------------

st.header("Results")

# Sonuç Ekranı
if st.button("Submit"):

    response = requests.post(f"http://127.0.0.1:8000/predict", json=input_data)
    result = response.json()

    # Info mesajı oluşturma
    st.info("You can find the result below.")

    today = date.today()
    time = datetime.now().strftime("%H:%M:%S")

    # Sonuçları Görüntülemek için DataFrame
    online_results_df = pd.DataFrame({
    "Name": [name],
    "Surname": [surname],
    "Date": [today],
    "Time": [time],
    "Prediction": result['Is Canceled'],
    "Cancellation Probability": '%' + str(np.round((result['Cancelation Probability (%)']*100),2)),
    "Arrival Date": [arrival_day],
    "Arrival Month": [arrival_month],
    "Arrival Year": [arrival_year],
    "No of Weekend Nights": [no_of_weekend_nights],
    "No of Week Nights": [no_of_week_nights],
    "Market Segment Type Online": [market_segment_type],
    "No of Adults": [no_of_adults],
    "Lead Time": [lead_time],
    "No of Special Requests": [no_of_special_requests],
    "Avg Price Per Room": [avg_price_per_room]
    })

    st.dataframe(online_results_df)

    with sqlite3.connect("hotel_db.sqlite") as conn:
        cursor = conn.cursor()
        online_results_df.to_sql("predictions", conn, if_exists="append", index=False)

    with open("explainer.pkl", "rb") as explainer:
        explainer = pickle.load(explainer)
    
    test_data = pd.DataFrame(input_data, index=[0])

    shap_values = explainer(test_data)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.waterfall_plot(shap_values[0], show=False)

    st.info("You can find the Shap Explanation of your prediction!")
    st.pyplot(fig)

    # Görseli geçici belleğe kaydetmek için BytesIO kullanma
    buffer = BytesIO()
    fig.savefig(buffer, format="png")
    buffer.seek(0)  # Bellek konumunu başa al

    # Download butonu ekleme
    st.download_button(
        label="Download SHAP Explanation",
        data=buffer,
        file_name="shap_local_explanation.png",
        mime="image/png"
        )

else:
    st.markdown("Please click the *Submit Button*!")

#---------------------------------------------------------------------------------------------------------------------

st.header("📥 Batch Prediction")

uploaded_file = st.file_uploader("**Upload your file**", type="csv")

if uploaded_file is not None:

    batch_df = pd.read_csv(uploaded_file)

    columns = ["lead_time", "no_of_special_requests", "avg_price_per_room", "market_segment_type", "arrival_month",
               "arrival_date", "arrival_year", "no_of_weekend_nights", "no_of_week_nights", "no_of_adults"]
    batch_df = batch_df[columns]

    for booking_type in batch_df["market_segment_type"]:
        if booking_type == "Online":
            batch_df["market_segment_type_Online"] = 1
        else:
            batch_df["market_segment_type_Online"] = 0

    batch_df.drop(columns="market_segment_type", inplace=True)
    batch_df = batch_df[["lead_time", "no_of_special_requests", "avg_price_per_room", "market_segment_type_Online", "arrival_month",
                         "arrival_date", "arrival_year", "no_of_weekend_nights", "no_of_week_nights", "no_of_adults"]]

    batch_pred = xgb_model.predict(batch_df)
    batch_pred_probability = xgb_model.predict_proba(batch_df)

    predictions = pd.DataFrame({
    "Prediction": batch_pred,
    "Cancellation Probability": batch_pred_probability[:,1] * 100
    })

    st.info("You can find the result below.")

    pred_df = pd.concat([predictions, batch_df], ignore_index=True, axis=1)
    pred_columns = ["Prediction", "Cancellation Probability (%)", "Lead Time", "No of Special Requests",
                    "Avg Price Per Room", "Market Segment Type Online", "Arrival Month", "Arrival Date",
                    "Arrival Year", "No of Weekend Nights", "No of Week Nights", "No of Adults"]
    pred_df.columns = pred_columns

    pred_df["Prediction"] = pred_df["Prediction"].apply(lambda pred: "Will Not be Canceled" if pred==0 else "Will be Canceled")

    st.dataframe(pred_df)