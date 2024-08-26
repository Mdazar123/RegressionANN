import streamlit as st
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
# from tensorflow.keras.layers import InputLayer
# model = tf.keras.models.load_model('my_model.keras', custom_objects={'InputLayer': InputLayer})

# Load the trained model
# model = tf.keras.models.load_model('my_model.keras')
# # model=load_model('model.h5')

# # Load the encoders and scaler
# model = tf.keras.models.load_model('model.h5')
# from tensorflow.keras.models import load_model

# # Suppress warnings
# tf.get_logger().setLevel('ERROR')

# Load the model
# model = load_model('model.h5')
# Load the encoders and scaler
with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)
with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)
with open('scaler_reg.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Starting with Streamlit
st.title("Customer Salary Estimation")

# # User input
geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider('Age', 18, 92)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.slider('Number of Products', 1, 4)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
is_Exited = st.selectbox('Is Exited', [0, 1])

# Encoding the categorical data
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
gender_encoded = label_encoder_gender.transform([gender])[0]

# Prepare the feature vector
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [gender_encoded],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'Exited': [is_Exited]
})

# Adding the encoded geography features
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# Scaling the features
input_data_scaled = scaler.transform(input_data)

# Making predictions
model = tf.keras.models.load_model('regression_model.h5')
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# Display the prediction result
st.write(f"Estimated salary is : {prediction_proba}")
