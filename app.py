import streamlit as st
import numpy as np 
import pandas as pd 
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import tensorflow as tf 
import pickle 

#Load the trained model 
model = tf.keras.models.load_model('model.h5')

#Load the encoder and scalar \
with open('one_hot_encoder_geography.pkl','rb') as file :
    one_hot_encoder_geography = pickle.load(file)

with open('scalar.pkl','rb') as file :
    scalar = pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file :
    label_encoder_gender= pickle.load(file) 

## Streamlit app 
st.title('Customer churn Prediction')

## User inputs
geography = st.selectbox('Country', one_hot_encoder_geography.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)
age = st.slider("Age",18,90)
balance = st.number_input('Balance')
credit_score = st.number_input('Credit Score')
estimated_salary = st.number_input('Estimated Salary')
tenure = st.slider('Tenure', 0,10)
num_of_products = st.slider('Number of Products',1,4)
has_cr_card = st.selectbox('Has Credit Card',[0,1])
is_active_member = st.selectbox('Is Active Member', [0,1])

#Prepare the input data 
input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary':[estimated_salary]
})

## One Hot Encoder "Geography"
geo_encoded = one_hot_encoder_geography.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns= one_hot_encoder_geography.get_feature_names_out(['Geography']))

## Combimne the data wit geography data 
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df] ,axis=1)

## Scale the data 
input_data_scaled =scalar.transform(input_data)

## Predict Churn 

prediction =model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

## Have a button for prediction 
if st.button(label = "Predict" ):

    if prediction_proba >0.5 :
        st.write("The customer is likely to churn")
    else :
        st.write("the customer is not likely to churn")



