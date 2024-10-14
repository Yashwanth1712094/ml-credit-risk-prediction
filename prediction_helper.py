import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
MODEL_PATH='artifacts/model_data.joblib'

model_data=joblib.load(MODEL_PATH)
model=model_data['model']
scaler=model_data['scaler']
features=model_data['features']
cols_to_scale=model_data['cols_to_scale']


def prepare_df(age,income,loan_amount,loan_tenure,avg_dpd,deliquency_ratio,credit_utilization_ratio,num_open_accounts,residence_type,loan_purpose,loan_type):

    input_data={
        'age':age,
        'loan_tenure_months':loan_tenure,
        'number_of_open_accounts':num_open_accounts,
        'credit_utilization_ratio':credit_utilization_ratio,
        'loan_to_income':loan_amount/income if income>0 else 0,
        'delinquent_ratio':deliquency_ratio,
        'avg_dpd_per_deliquency':avg_dpd,
        'residence_type_Owned':1 if residence_type=='Owned' else 0,
        'residence_type_Rented':1 if residence_type=='Rented' else 0,
        'loan_purpose_Education':1 if loan_purpose=='Education' else 0,
        'loan_purpose_Home':1 if loan_purpose=='Home' else 0,
        'loan_purpose_Personal':1 if loan_purpose=='Personal' else 0,
        'loan_type_Unsecured':1 if loan_type=='Unsecured' else 0,
        #dummies
        'number_of_dependants':1,
        'years_at_current_address':1,
        'zipcode':1,
        'sanction_amount':1,
        'processing_fee':1,
        'gst':1,
        'net_disbursement':1,
        'principal_outstanding':1,
        'bank_balance_at_application':1,
        'number_of_closed_accounts':1,
        'enquiry_count':1,
        }
    df=pd.DataFrame([input_data])
    df[cols_to_scale]=scaler.transform(df[cols_to_scale])
    
    df=df[features]
    return df


def calculate_credit_score(input_df,base_score=300,scale_length=600):
     
    x=np.dot(input_df.values,model.coef_.T)+model.intercept_

    default_propability=1/(1+np.exp(-x))
    non_default_propability=1-default_propability

    credit_score=base_score+non_default_propability.flatten()*scale_length

    def get_rating(score):
        if 300<=score<500:
            return "poor"
        elif 500<=score<650:
            return "Average"
        elif 650<=score<750:
            return "Good"
        elif 750<=score<900:
            return "Excellent"
        else:
            return "Undefined"
    rating=get_rating(credit_score)
    return default_propability.flatten()[0],int(credit_score),rating


def predict(age,income,loan_amount,loan_tenure,avg_dpd,deliquency_ratio,credit_utilization_ratio,num_open_accounts,residence_type,loan_purpose,loan_type):

    input_df=prepare_df(age,income,loan_amount,loan_tenure,avg_dpd,deliquency_ratio,credit_utilization_ratio,num_open_accounts,residence_type,loan_purpose,loan_type)


    
    propability= 40
    credit_score=300

    rating="poor"
    propability,credit_score,rating=calculate_credit_score(input_df)
    return propability,credit_score,rating

# if __name__=='__main__':
#     age=25
#     income=450000
#     print