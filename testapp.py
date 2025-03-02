import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import joblib

datatype_dict = {'CreditScore': 'float64',
 'Age':'float64',
 'Tenure': 'float64',
 'Balance': 'float64',
 'NumOfProducts': 'int64',
 'HasCrCard': 'int64',
 'IsActiveMember': 'int64',
 'EstimatedSalary': 'float64',
 'Geography_Germany': 'float64',
 'Geography_Spain': 'float64',
 'Gender_Male': 'float64'}

### version number and read model and scaler object
v = 0
annmodel = load_model("Annmodel_v_"+ str(v) +".h5", compile=False) ## model read
sc = joblib.load('scaler.pkl')  ## scaler 

###### inference code 
def modelInference(tdf,model,sc):

    col2scale = ['CreditScore', 'Age', 'Tenure', 'Balance', 'EstimatedSalary']
    tdf = tdf.astype(datatype_dict) ## change datatype

    tdf[col2scale] = sc.transform( tdf[col2scale])

    res = annmodel.predict(tdf)
    return res[0][0]
# Streamlit app
st.title("Prediction App")

# Create a form for user input
with st.form("prediction_form"):
    st.header("Enter Input Values")
    CreditScore = st.number_input("Credit Score", value=0.0)
    Age = st.number_input("Age", value=0.0)
    Tenure = st.number_input("Tenure", value=0.0)
    Balance = st.number_input("Balance", value=0.0)
    NumOfProducts = st.number_input("Number of Products", value=0.0)
    HasCrCard = st.selectbox("Has Credit Card", [0, 1])
    IsActiveMember = st.selectbox("Is Active Member", [0, 1])
    EstimatedSalary = st.number_input("Estimated Salary", value=0.0)
    Geography_Germany = st.selectbox("Geography: Germany", [0, 1])
    Geography_Spain = st.selectbox("Geography: Spain", [0, 1])
    Gender_Male = st.selectbox("Gender: Male", [0, 1])
    
    # Submit button
    submitted = st.form_submit_button("Predict")
    
    if submitted:
        inputs = np.array([CreditScore, Age, Tenure, Balance, NumOfProducts, HasCrCard,
                           IsActiveMember, EstimatedSalary, Geography_Germany,
                           Geography_Spain, Gender_Male])
        
        print(inputs.shape)
        inputs = inputs.reshape(1,-1)

        ### convert to dataframe 
        tdf = pd.DataFrame(index = range(1),columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard',
       'IsActiveMember', 'EstimatedSalary', 'Geography_Germany',
       'Geography_Spain', 'Gender_Male'])
        
        tdf.iloc[0] = inputs.reshape(1,-1)



        prediction = modelInference(tdf,annmodel,sc)
        st.success(f"Predicted Value is: {prediction:.2f}")
