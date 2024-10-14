import streamlit as st
from prediction_helper import predict
st.title("Lauki Finance:Credit Risk Modelling")
row1=st.columns(3)
row2=st.columns(3)
row3=st.columns(3)
row4=st.columns(3)

with row1[0]:
    age=st.number_input('Age',min_value=18,max_value=100,step=1)
with row1[1]:
    income=st.number_input('Income',min_value=0,max_value=12000000)
with row1[2]:
    loan_amount=st.number_input('Loan Amount',min_value=1,value=2560000)


# Calculate loan to income ratio
loan_to_income_ratio=loan_amount/income if income>0 else 0

with row2[0]:
    st.text("Loan to Income Ratio")
    st.text(f"{loan_to_income_ratio:.2f}")
with row2[1]:
    loan_tenure=st.number_input('Loan Tenure(months)',min_value=1,max_value=50,step=1)
with row2[2]:
    avg_dpd=st.number_input("AVG DPD",min_value=0,max_value=1000)


with row3[0]:
    deliquency_ratio=st.number_input("Deliquency Ratio",min_value=0,max_value=100,step=1,value=30)
with row3[1]:
    credit_utilization_ratio=st.number_input("Credit Utilization Ratio",min_value=0,max_value=100,step=1,value=30)
with row3[2]:
    num_open_accounts=st.number_input("Open Loan Accounts",min_value=1,max_value=4,step=1,value=2)

with row4[0]:
    residence_type=st.selectbox("Residence Type",['Owned','Rented','Mortgage'])
with row4[1]:
    loan_purpose=st.selectbox("Loan Purpose",['Education','Home','Auto','Personl'])
        
with row4[2]:
    loan_type=st.selectbox("Loan Type",['Unsecured','Secured']) 


if st.button("calculate Risk"):
    propability,creditscore,rating=predict(age,income,loan_amount,loan_tenure,avg_dpd,deliquency_ratio,credit_utilization_ratio,num_open_accounts,residence_type,loan_purpose,loan_type)

    st.write(f"Default Propability: {propability:.2%}")
    st.write(f"Credit score:{creditscore}")
    st.write(f"Rating:{rating}")