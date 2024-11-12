import streamlit as st
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
import seaborn as sn
from matplotlib import pyplot as plt
import joblib
import os


le=preprocessing.LabelEncoder() #Encoding the string values into labels 




def load_prediction_models(model_file):
	loaded_model = joblib.load(open(os.path.join('LR.pkl'),"rb"))
	return loaded_model



st.title("Loan Stream")



activities = ['View DATA','Prediction']
choices = st.sidebar.selectbox("Select",activities)
 
    
if choices == 'Prediction':
    Emp_dur = st.slider("Enter the Employment duration",0,20)
    own_type = st.radio("Select the ownership type",('RENT','MORTGAGE','OWN'))
    income_type = st.radio("Select your level of income",('Low','Med','High'))
    app_type = st.radio("Select applicant type",('Individual','Joint'))
    interestpay = st.radio("Select your level of interest",('Low','Med','High'))
    grade = st.selectbox("Grade",['A','B','C','D','E','F'])
    annual_pay = st.number_input("Enter The Income")
    loan_amt = st.number_input("Enter the Loan amount")
    intrate = st.number_input("Enter the interest rate")
    duration = st.radio("Select the duration of the Loan",('36 months','60 months'))
    dti = st.number_input("Enter Debt to Income")
    tot_payment = st.number_input("Enter The total money paid back")
    totalrec = st.number_input("Enter the total amount recieved")
    rec = st.number_input("Enter the amount recovered")
    installments = st.number_input("Enter the installments")

    DATA = le.fit_transform([Emp_dur,own_type,income_type,app_type,interestpay,grade,annual_pay,loan_amt,intrate,duration,dti,tot_payment,totalrec,rec,installments])

    DATA.astype(np.float64)
    D=DATA.reshape(1,-1)







    predictor = load_prediction_models("LR.pkl")
    prediction = predictor.predict(D)

    if st.button('Predict'):
        if prediction == 1:
            st.write('Will Default')
        else:
            st.write('Will not default')

        
        
        
        
elif choices == 'View DATA':
    
    st.subheader("Data is Shown Below")
    data=pd.read_csv (r'C:\Users\Srikkanth\Desktop\Files\LoanDefaultData.csv',nrows=300000)
    st.dataframe(data.head(20))

    if st.checkbox("Show Summary of Dataset"):
        st.write(data.describe())
    


    
    
    
    
    




