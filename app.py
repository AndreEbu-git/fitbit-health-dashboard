import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import numpy as np

# Load data and simple model
df = pd.read_csv('fitbit_processed.csv')
df['Date'] = pd.to_datetime(df['Date'])
model = joblib.load('calorie_predictor_simple.pkl')

# Ensure TotalActiveMinutes
if 'TotalActiveMinutes' not in df.columns:
    df['TotalActiveMinutes'] = df['VeryActiveMinutes'] + df['FairlyActiveMinutes'] + df['LightlyActiveMinutes']

st.set_page_config(page_title="Fitbit Health Dashboard", layout="wide")
st.title("🏃‍♂️ Fitbit Health & Fitness Insights Dashboard")
st.markdown("### Interactive analysis of wearable activity, sleep, and calorie data with AI predictions")

# User selection
unique_ids = sorted(df['Id'].unique())
user_id = st.selectbox("Select a User ID", unique_ids, index=0)
user_data = df[df['Id'] == user_id]

# Key Averages
st.subheader("Key Averages")
avg_steps = user_data['TotalSteps'].mean()
avg_very = user_data['VeryActiveMinutes'].mean()
avg_sleep = user_data['TotalMinutesAsleep'].mean() / 60 if user_data['TotalMinutesAsleep'].notna().any() else 0
avg_calories = user_data['Calories'].mean()

st.markdown(f"""
**Average Daily Steps**: {avg_steps:,.0f}  
**Avg Very Active Minutes**: {avg_very:.0f}  
**Avg Sleep**: {avg_sleep:.1f} hours  
**Avg Calories Burned**: {avg_calories:.0f}
""")

# Charts using native Streamlit (they work perfectly on Streamlit Cloud)
st.subheader(f"Daily Steps Trend for User {user_id}")
st.line_chart(user_data.set_index('Date')['TotalSteps'])

st.subheader("Activity Level Distribution")
st.bar_chart(user_data['ActivityLevel'].value_counts())

st.subheader("Steps vs Sleep")
sleep_data = user_data.dropna(subset=['TotalMinutesAsleep'])
if not sleep_data.empty:
    st.scatter_chart(sleep_data[['TotalMinutesAsleep', 'TotalSteps']].rename(columns={'TotalMinutesAsleep': 'Sleep Minutes'}))
else:
    st.info("No sleep data for this user.")

# Simple Predictor
st.header("🔮 Predict Calories Burned")
st.markdown("Adjust to simulate a day!")

col1, col2 = st.columns(2)
with col1:
    steps = st.slider("Total Steps", 0, 30000, int(avg_steps))
    very = st.slider("Very Active Minutes", 0, 180, int(avg_very))
    fairly = st.slider("Fairly Active Minutes", 0, 180, 20)
with col2:
    lightly = st.slider("Lightly Active Minutes", 0, 720, 200)
    sedentary = st.slider("Sedentary Minutes", 0, 1440, 600)

total_active = very + fairly + lightly

input_data = np.array([[steps, very, fairly, lightly, sedentary, total_active]])
prediction = model.predict(input_data)[0]

delta = prediction - avg_calories
st.markdown(f"**Predicted Calories: {int(prediction)}** 🔥")
st.caption(f"{abs(int(delta))} calories {'more' if delta > 0 else 'less'} than your average")

st.success("Dashboard deployed successfully!")
