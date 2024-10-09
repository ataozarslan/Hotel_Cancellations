import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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

df = pd.read_csv("hotel_bookings_prepared.csv")

# -----------------------------------------------------------------------

year_cancelations = df.groupby(["arrival_year","booking_status"])["Booking_ID"].count().reset_index()

booking_dist2017 = px.pie(names=year_cancelations[year_cancelations["arrival_year"]==2017]["booking_status"],
                      values=year_cancelations[year_cancelations["arrival_year"]==2017]["Booking_ID"],
                      title="Customer Cancellation Status in 2017",
                      color=year_cancelations[year_cancelations["arrival_year"]==2017]["booking_status"],
                      color_discrete_map={"Canceled":"red",
                                          "Not_Canceled":"blue"}).update_layout(legend_title_text='Cancellation Status').update_traces(textposition='inside', textinfo='percent+label')

booking_dist2018 = px.pie(names=year_cancelations[year_cancelations['arrival_year']==2018]['booking_status'],
                          values=year_cancelations[year_cancelations['arrival_year']==2018]['Booking_ID'],
                          title='Customer Cancellation Status in 2018',
                          color=year_cancelations[year_cancelations["arrival_year"]==2018]["booking_status"],
                          color_discrete_map={"Canceled":"red",
                                              "Not_Canceled":"blue"}).update_layout(legend_title_text='Cancellation Status').update_traces(textposition='inside', textinfo='percent+label')

st.plotly_chart(booking_dist2017, use_container_width=True)
st.plotly_chart(booking_dist2018, use_container_width=True)

# -----------------------------------------------------------------------

corr_matrix = df.corr(numeric_only=True)

corr_matrix = px.imshow(
    corr_matrix,
    zmin=-1,
    zmax=1,
    text_auto=".2f",
    aspect="auto",
    color_continuous_scale='RdBu_r',
    labels={"color": "coef"}
).update_layout(title="Correlation Matrix")

st.plotly_chart(corr_matrix, use_container_width=True)

# -----------------------------------------------------------------------

lead_time_dist = go.Figure()
lead_time_dist.add_trace(go.Histogram(x=df[df["booking_status"]=="Not_Canceled"]["lead_time"], name="Not Canceled", marker=dict(color="blue")))
lead_time_dist.add_trace(go.Histogram(x=df[df["booking_status"]=="Canceled"]["lead_time"], name="Canceled", marker=dict(color="red")))

lead_time_dist.update_xaxes(title_text="Lead Time (Days) Before Arrival")
lead_time_dist.update_yaxes(title_text="Customer Count")
lead_time_dist.update_layout(title="Lead Time Distribution", barmode="overlay")
lead_time_dist.update_traces(opacity=0.6)

st.plotly_chart(lead_time_dist, use_container_width=True)