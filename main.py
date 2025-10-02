import streamlit as st

# Define pages (by file or function)
page1 = st.Page("Pages/1_time_series.py", title="Time Series Analysis", icon="📊")
page2 = st.Page("Pages/2_about_me.py", title="Overview", icon="👤")
page3 = st.Page("Pages/3-Challenges_faced.py", title="Challenges Faced", icon="🧗")

# Create navigation
pg = st.navigation([page1, page2])