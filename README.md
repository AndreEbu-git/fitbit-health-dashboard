# Fitbit Health & Fitness Insights Dashboard 🏃‍♂️

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](fitbit-health-dashboard.streamlit.app/)

An interactive web dashboard built with **Streamlit** that analyzes real Fitbit wearable data to provide personalized health and fitness insights, including activity trends, sleep correlations, and AI-powered calorie burn predictions.

## Live Demo
👉 [Open the dashboard here](fitbit-health-dashboard.streamlit.app/)

## Features
- **User Selection**: Explore data from 33 real Fitbit users.
- **Key Metrics**: Average daily steps, very active minutes, sleep hours, and calories burned.
- **Visualizations**:
  - Daily steps trend over time
  - Activity level distribution (Sedentary to Highly Active)
  - Steps vs Sleep scatter plot
- **AI Calorie Predictor**: Adjust sliders (steps, activity intensity, sedentary time) to simulate a day and get a personalized calorie burn estimate using a trained Random Forest model.

## Dataset
- Source: [FitBit Fitness Tracker Data on Kaggle](https://www.kaggle.com/datasets/arashnic/fitbit) (public dataset from 33 users over ~1 month).
- Includes daily activity, calories, sleep, and intensity minutes.

## Tech Stack
- **Python** with Pandas for data processing and feature engineering
- **Scikit-learn** (Random Forest Regressor) for calorie prediction
- **Streamlit** for the interactive web dashboard
- Deployed on **Streamlit Community Cloud**

## How to Run Locally
1. Clone the repo:
   ```bash
   git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
   cd YOUR_REPO_NAME
