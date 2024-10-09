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

# Başlık Ekleme
st.title("Can we predict hotel cancellations of our customers with their reasons?")

st.markdown("- According to [Procuswire](https://www.phocuswire.com/One-in-five-hotel-bookings-on-the-web-are-cancelled), did you know that **almost 20% of hotels rooms booked online are cancelled** before the guest arrives?")
st.markdown("- That’s **1 of every 5 bookings** you get!")
st.markdown("- It not only directly impacts the revenue, but also leads to poor marketing, since Online Travel Agencies keep track of cancellations drawn by a hotel.")
st.markdown("- Specifically, **cancellations made close to the time of service are the most damaging** for hotels because they leave management with no time to react!")

# Resim Ekleme
st.image("images/cancelled-holidays.jpg")

st.markdown("After the latest developments in the artificial intelligence industry, they expect us to develop a **machine learning model** in line with their needs and help them with their research.")
st.markdown("In order to fill this gap, this study aims to identify individuals who are likely to cancel in the short term with AI techniques using the dataset shared with us by the hotel.")
st.markdown("*Let's help them!*")