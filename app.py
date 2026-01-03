import streamlit as st
import pandas as pd
import altair as alt
import joblib
import numpy as np

# Load data and model
df = pd.read_csv('fitbit_processed.csv')
df['Date'] = pd.to_datetime(df['Date'])
model_bundle = joblib.load('calorie_predictor_v2.pkl')
model = model_bundle['model']
preprocessor = model_bundle['preprocessor']

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

# Altair Charts (reliable and beautiful)
# Steps Trend
steps_chart = alt.Chart(user_data).mark_line(color='cyan', strokeWidth=3).encode(
    x='Date:T',
    y='TotalSteps:Q'
).properties(title=f"Daily Steps Trend for User {user_id}", width=800, height=400)
st.altair_chart(steps_chart, use_container_width=True)

# Activity Level Distribution
activity_chart = alt.Chart(user_data).mark_bar().encode(
    x='ActivityLevel:N',
    y='count()',
    color=alt.Color('ActivityLevel:N', scale=alt.Scale(scheme='category10'))
).properties(title="Activity Level Distribution", width=600)
st.altair_chart(activity_chart, use_container_width=True)

# Steps vs Sleep
if not user_data.dropna(subset=['TotalMinutesAsleep']).empty:
    sleep_chart = alt.Chart(user_data.dropna(subset=['TotalMinutesAsleep'])).mark_circle(size=100, opacity=0.8).encode(
        x='TotalMinutesAsleep:Q',
        y='TotalSteps:Q',
        color=alt.value('orange'),
        tooltip=['Date', 'TotalSteps', 'TotalMinutesAsleep']
    ).properties(title="Steps vs Sleep (minutes)", width=700, height=500)
    st.altair_chart(sleep_chart, use_container_width=True)
else:
    st.info("No sleep data available for this user.")

# AI Predictor
st.header("🔮 Predict Tomorrow's Calories Burned")
st.markdown("Sliders start at your averages — adjust to plan!")

avg_steps = user_data['TotalSteps'].mean()
avg_very = user_data['VeryActiveMinutes'].mean()
avg_fairly = user_data['FairlyActiveMinutes'].mean()
avg_lightly = user_data['LightlyActiveMinutes'].mean()
avg_sedentary = user_data['SedentaryMinutes'].mean()

col1, col2 = st.columns(2)
with col1:
    steps = st.slider("Total Steps", 0, 30000, int(avg_steps), step=500)
    very = st.slider("Very Active Minutes", 0, 180, int(avg_very))
    fairly = st.slider("Fairly Active Minutes", 0, 180, int(avg_fairly))
with col2:
    lightly = st.slider("Lightly Active Minutes", 0, 720, int(avg_lightly))
    sedentary = st.slider("Sedentary Minutes", 0, 1440, int(avg_sedentary))

total_active = very + fairly + lightly

# Prediction
input_cat = pd.DataFrame({'Id': [user_id]})
input_activity = pd.DataFrame([[steps, very, fairly, lightly, sedentary, total_active]],
                              columns=['TotalSteps', 'VeryActiveMinutes', 'FairlyActiveMinutes',
                                       'LightlyActiveMinutes', 'SedentaryMinutes', 'TotalActiveMinutes'])
input_df = pd.concat([input_cat, input_activity], axis=1)

X_input = preprocessor.transform(input_df)
prediction = model.predict(X_input)[0]

delta = prediction - avg_calories
st.markdown(f"**Predicted Calories Burned: {int(prediction)}** 🔥")
st.caption(f"That's {abs(int(delta))} calories **{'more' if delta > 0 else 'less'}** than your average")

if delta > 50:
    st.success("Excellent — higher burn day!")
elif delta < -50:
    st.warning("Lower activity — try boosting intensity!")
else:
    st.info("Similar to average — consistent progress!")

st.balloons()
st.success("Your final AI-powered dashboard is ready!")
