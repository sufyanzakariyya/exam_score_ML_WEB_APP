import streamlit as st
import sklearn
import pandas as pd
import numpy as np


import pickle

# Streamlit App Interface
st.write("""
# AI-Powered Exam Score Prediction for Smarter Insights.

""")
st.write('---')
df = pd.read_csv("StudentPerformanceFactors.csv") 
st.sidebar.header("User Input Features")

def user_input_features():
    Hourse_Studied = st.sidebar.slider("Hourse Studied", int(df['Hours_Studied'].min()), int(df['Hours_Studied'].max()), int(df['Hours_Studied'].mean()))
    Attendance = st.sidebar.slider("Attendance", int(df["Attendance"].min()), int(df["Attendance"].max()), int(df["Attendance"].mean()))
    Parental_Involvement = st.sidebar.selectbox("Parental Involvement", ('Low', 'Medium', 'High')) 
    Access_to_Resources = st.sidebar.selectbox("Access To Resources", ('High', 'Medium', 'Low'))
    Extracurricular_Activities = st.sidebar.selectbox("Extracurricular Activities", ('No', 'Yes'))
    Sleep_Hours = st.sidebar.selectbox("Sleep Hourse", ( 7,  8,  6, 10,  9,  5,  4))
    Previous_Scores = st.sidebar.slider("Previous Score", int(df['Previous_Scores'].min()), int(df['Previous_Scores'].max()), int(df['Previous_Scores'].mean()))
    Motivation_Level = st.sidebar.selectbox("Motivation Level", ('Low', 'Medium', 'High'))
    Internet_Access = st.sidebar.selectbox("Internet Access", ('No', 'Yes'))
    Tutoring_Sessions = st.sidebar.selectbox("Tutoring Sessions", (0, 2, 1, 3, 4, 5, 6, 7, 8))
    Family_Income = st.sidebar.selectbox("Family Income", ('Low', 'Medium', 'High'))
    Teacher_Quality = st.sidebar.selectbox("Teacher Quality", ('Low', 'Medium', 'High'))
    School_Type = st.sidebar.selectbox("School Type", ('Public', 'Private'))
    Peer_Influence = st.sidebar.selectbox("Peer Influence", ('Positive', 'Negative', 'Neutral'))
    Physical_Activity = st.sidebar.selectbox("Physical Activity", (3, 4, 2, 1, 5, 0, 6))
    Learning_Disabilities = st.sidebar.selectbox("Learning_Disabilities", (3, 4, 2, 1, 5, 0, 6))
    Parental_Education_Level = st.sidebar.selectbox("Parental Educational Level", ('High School', 'College', 'Postgraduate'))
    Distance_from_Home = st.sidebar.selectbox("Distance From Home", ('Near', 'Moderate', 'Far'))
    Gender = st.sidebar.selectbox("Gender", ("Male", "Female"))

    data = {
        'Hours_Studied': Hourse_Studied,
        'Attendance': Attendance,
        'Parental_Involvement': Parental_Involvement,
        'Access_to_Resources': Access_to_Resources,
        'Extracurricular_Activities':Extracurricular_Activities,
        'Sleep_Hours':Sleep_Hours,
        'Previous_Scores': Previous_Scores,
        'Motivation_Level': Motivation_Level,
        'Internet_Access': Internet_Access,
        'Tutoring_Sessions':Tutoring_Sessions,
        'Family_Income': Family_Income,
        'Teacher_Quality': Teacher_Quality,
        'School_Type': School_Type,
        'Peer_Influence':Peer_Influence,
        'Physical_Activity': Physical_Activity,
        'Learning_Disabilities': Learning_Disabilities,
        'Parental_Education_Level' :Parental_Education_Level,
        'Distance_from_Home':Distance_from_Home,
        'Gender':Gender
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display User Input Features 
st.subheader("User Input features")
st.write(input_df)

# Load the saved model
load_clf = pickle.load(open("RG_MODEL.pkl", "rb"))

# Only predict when the button is clicked
if st.button("Predict"):
    prediction = load_clf.predict(input_df)

    st.success(f"Predicted Exam Score: {round(prediction[0])}")