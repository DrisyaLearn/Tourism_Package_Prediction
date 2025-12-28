import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the model
model_path = hf_hub_download(repo_id="DrishVij/Tourism-Package-Prediction", 
                             filename="final_tourism_prediction_model_v1.joblib",
                             repo_type="model")
model = joblib.load(model_path)

# Streamlit UI for Machine Failure Prediction
st.title("Tourism Package Prediction App")
st.write("""
This application predicts the likelihood of a customer byuing a Tour Package based on the pitch.
Please enter the customer and pitch data below to get a prediction.
""")

st.markdown("---")
st.subheader("Customer's Personal Details")

# Age: Range from 18-65
age = st.number_input("Age (18-65 years)",
                       min_value=18,
                       max_value=65,
                       value=36  # median value in dataset
                       )

# Gender: Male, Female
gender = st.selectbox("Gender",["Female", "Male"]
                     )

# MaritalStatus: Single, Married, Divorced
marital_status = st.selectbox("Marital Status",["Single", "Divorced", "Married"]
                             )

# City Tier: 1, 2, 3
city_tier = st.selectbox("City Tier",[1, 2, 3]
                         )

st.markdown("---")
st.subheader("Customer's Professional Details")

# Occupation: Salaried, Small Business, Large Business, Free Lancer
occupation = st.selectbox("Occupation Type",
                          ["Salaried", "Free Lancer", "Small Business", "Large Business"]
                          )

# Designation: Executive, Manager, Senior Manager, AVP, VP
designation = st.selectbox("Designation",
                          ["Manager", "Executive", "Senior Manager", "AVP", "VP"]
                          )

# MonthlyIncome: Range from 1000 to 100000
monthly_income = st.number_input("Monthly Income (₹1,000 - ₹1,00,000)",
                                        min_value=1000,
                                        max_value=100000,
                                        value=22418,  # median value in dataset
                                        step=1000
                                        )

st.markdown("---")
st.subheader("Customer's Travel Preferences")

# NumberOfTrips: Range from 1-25
num_trips = st.slider("Number of Trips taken Annually (1-25)",
                             min_value=1,
                             max_value=25,
                             value=3  # median value in dataset
                     )


# Passport: 0 or 1
passport = st.selectbox("Does customer have a valid passport?",
                              [0, 1],
                              format_func=lambda x: "Yes" if x == 1 else "No",
                              index=1  # median is 1
                        )

# OwnCar: 0 or 1
own_car = st.selectbox("Does customer own a car?",
                              [0, 1],
                              format_func=lambda x: "Yes" if x == 1 else "No",
                              index=1  # median is 1
                        )

# PreferredPropertyStar: 3, 4, 5
preferred_property_star = st.selectbox("Preferred Hotel Rating (3-5 stars)",
                                              [3, 4, 5],
                                              index=0  # median is 3
                                      )

st.markdown("---")
st.subheader("Trip Details")

# NumberOfPersonVisiting: Range from 1-5
num_persons = st.number_input("Number of Persons Visiting in the group (1-5)",
                               min_value=1,
                               max_value=5,
                               value=3  # median
                       )

# NumberOfChildrenVisiting: Range from 0-3
num_children = st.number_input("Number of Children under 5 years (0-3)",
                                min_value=0,
                                max_value=3,
                                value=1  # median
                        )

st.markdown("---")
st.subheader("Interaction Details")

# TypeofContact: Company Invited, Self Enquiry
type_of_contact = st.selectbox("How was the customer contacted?",
                                      ["Self Enquiry", "Company Invited"]
                              )

# ProductPitched: Basic, Standard, Deluxe, Super Deluxe, King
product_pitched = st.selectbox("Type of Package Pitched to the customer",
                                      ["Deluxe", "Basic", "Standard", "Super Deluxe", "King"]
                              )

# DurationOfPitch: Range from 5-127 minutes
duration_of_pitch = st.slider("Sales pitch duration in minutes (1-130)",
                                     min_value=1,
                                     max_value=130,
                                     value=14  # median
                              )

# NumberOfFollowups: Range from 1-6
num_followups = st.slider("Noumber of follow-ups after initial pitch (1-6)",
                                 min_value=1,
                                 max_value=6,
                                 value=4  # median
                          )

# PitchSatisfactionScore: Range from 1-5
pitch_satisfaction = st.slider("Customer Pitch Satisfaction Score (1=Very Low, 5=Very High)",
                                      min_value=1,
                                      max_value=5,
                                      value=3  # median
                              )

# Assemble input into DataFrame
input_data = pd.DataFrame([{
        'Age': age,
        'CityTier': city_tier,
        'DurationOfPitch': duration_of_pitch,
        'NumberOfPersonVisiting': num_persons,
        'NumberOfFollowups': num_followups,
        'PreferredPropertyStar': preferred_property_star,
        'NumberOfTrips': num_trips,
        'Passport': passport,
        'PitchSatisfactionScore': pitch_satisfaction,
        'NumberOfChildrenVisiting': num_children,
        'MonthlyIncome': monthly_income,
        'TypeofContact': type_of_contact,
        'Occupation': occupation,
        'Gender': gender,
        'OwnCar': own_car,
        'ProductPitched': product_pitched,
        'MaritalStatus': marital_status,
        'Designation': designation
}])

if st.button("Predict Package Buying"):
    prediction = model.predict(input_data)[0]
    result = "Buying" if prediction == 1 else "Not Buying"
    st.subheader("Prediction Result:")
    st.success(f"The model predicts: **{result}**")
