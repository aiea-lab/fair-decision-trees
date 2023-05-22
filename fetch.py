import numpy as np
import pandas as pd
from aif360.sklearn.datasets import fetch_adult
from aif360.sklearn.datasets import fetch_bank
from aif360.sklearn.datasets import fetch_compas
from aif360.sklearn.datasets import fetch_german
from aif360.sklearn.datasets import fetch_meps

def convert_features_to_one_hot(df, feature_name_list):
  for feature_name in feature_name_list:
    df = pd.get_dummies(df, columns=[feature_name])
  
  return df

def adult():
    # Fetching adult dataset
    # https://archive.ics.uci.edu/ml/datasets/adult
    # https://aif360.readthedocs.io/en/latest/modules/generated/aif360.datasets.AdultDataset.html
    adult = fetch_adult(numeric_only=True)
    adult_income = adult.y
    X = adult.X.values
    y = adult_income.values
    # Combination of input output
    Xy = np.append(X, y[:, None], axis=1)
    columns = ["age", "education", "race", "sex", "capital-gain", "capital-loss", "hours-per-week"]
    class_names = ["<$50k",">$50k"]
    disc_index = 3 # sex (1: Male, 0: Female)
    return X, y, Xy, columns, class_names, disc_index

def bank():
    # Fetching bank dataset
    # https://archive.ics.uci.edu/ml/datasets/bank+marketing
    # https://aif360.readthedocs.io/en/latest/modules/generated/aif360.sklearn.datasets.fetch_bank.html
    bank = fetch_bank(numeric_only=True, dropna=True)
    bank_income = bank.y
    X = bank.X.values
    X[:,0] = X[:,0] >= 25
    y = bank_income.values
    # Combination of input output
    Xy = np.append(X, y[:, None], axis=1)
    columns = ["age", "education", "balance", "day", "campaign", "pdays", "previous"]
    class_names = ["No Deposit","Yes Deposit"] # has the client subscribed a term deposit?
    disc_index = 0 # Age (1: >=25, 0: <25)
    return X, y, Xy, columns, class_names, disc_index

def compas(disc="race"):
    # Fetching compas dataset
    # https://aif360.readthedocs.io/en/latest/modules/generated/aif360.sklearn.datasets.fetch_compas.html
    compas = fetch_compas(numeric_only=True, dropna=True, binary_race=True)
    compas_income = compas.y
    X = compas.X.values
    X = np.delete(X, 2, 1)
    y = compas_income.values
    # Combination of input output
    Xy = np.append(X, y[:, None], axis=1)
    columns = ['sex', 'age', 'age_cat', 'juv_fel_count', 'juv_misd_count', 'juv_other_count', 'priors_count', 'c_charge_degree']
    class_names = ["Recidivated","Survived"] # has the client subscribed a term deposit?
    if disc=="race":
        disc_index = 2 # Race (1: Caucasian, 0: African-American)
    else:
        disc_index = 0 # Sex (1: Female, 0: Male)
    return X, y, Xy, columns, class_names, disc_index
  
def german(disc="foreign"):
    # Fetching german dataset
    # https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
    # https://aif360.readthedocs.io/en/latest/modules/generated/aif360.sklearn.datasets.fetch_german.html
    german = fetch_german(numeric_only=True, dropna=True, binary_age=True)
    german_income = german.y
    X = german.X.values
    y = german_income.values
    # Combination of input output
    Xy = np.append(X, y[:, None], axis=1)
    columns = ["duration", "credit_amount", "installment_commitment", "residence_since", "age", "existing_credits", "num_dependents", "foreign_worker", "sex"]
    class_names = ["Bad Credit","Good Credit"] # Good or bad credit score
    if disc == "foreign":
        disc_index = 7 # Foreign_worker (1: Domestic, 0: Foreign)
    else:
        disc_index = 8 # Sex (1: Male, 0: Female)

    # Binary Age
    # disc_index = 4 # Age (1: >=25, 0: <25)
    # X[:,4] = X[:,4] >= 25
    return X, y, Xy, columns, class_names, disc_index

def credit(disc="sex", file='data/default_credit.csv'):
    # fetching credit default dataset
    # https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients
    df = pd.read_csv(file)
    
    # [LIMIT_BAL, SEX, EDUCATION, MARRIAGE, AGE, PAY_0, PAY_2, PAY_3, PAY_4, PAY_5, PAY_6, BILL_AMT1, BILL_AMT2, BILL_AMT3, BILL_AMT4, BILL_AMT5, BILL_AMT6, PAY_AMT1, PAY_AMT2, PAY_AMT3, PAY_AMT4, PAY_AMT5, PAY_AMT6, default payment next month]
    columns = list(df.columns)
    credit = df.to_numpy()
    X = credit[:,:-1]
    X[:,1]=np.abs((X[:,1]-1)-1)
    X[:,2] = X[:, 2]<=1
    y = np.abs(credit[:,-1]-1)
    Xy = np.append(X, y[:, None], axis=1)
    class_names = ["Defaulted", "Not Defaulted"]
    if disc == 'sex':
        disc_index = 1 # sex (1: Male, 0: Female)
    else:
        disc_index = 2 # marital (1: Married, 0: Other)
    return X, y, Xy, columns, class_names, disc_index

def fraud(file='data/bank_fraud.csv', seed=42):
    # fetching bank fraud dataset
    # https://www.kaggle.com/datasets/sgpjesus/bank-account-fraud-dataset-neurips-2022
    df = pd.read_csv(file)
    df = convert_features_to_one_hot(df, ['payment_type','employment_status','housing_status','source','device_os'])
    # [income,name_email_similarity,prev_address_months_count,current_address_months_count,customer_age,days_since_request,intended_balcon_amount,payment_type,zip_count_4w,velocity_6h,velocity_24h,velocity_4w,bank_branch_count_8w,date_of_birth_distinct_emails_4w,employment_status,credit_risk_score,email_is_free,housing_status,phone_home_valid,phone_mobile_valid,bank_months_count,has_other_cards,proposed_credit_limit,foreign_request,source,session_length_in_minutes,device_os,keep_alive_session,device_distinct_emails_8w,device_fraud_count,month]
    # ['income', 'name_email_similarity', 'prev_address_months_count', 'current_address_months_count', 'customer_age', 'days_since_request', 'intended_balcon_amount', 'zip_count_4w', 'velocity_6h', 'velocity_24h', 'velocity_4w', 'bank_branch_count_8w', 'date_of_birth_distinct_emails_4w', 'credit_risk_score', 'email_is_free', 'phone_home_valid', 'phone_mobile_valid', 'bank_months_count', 'has_other_cards', 'proposed_credit_limit', 'foreign_request', 'session_length_in_minutes', 'keep_alive_session', 'device_distinct_emails_8w', 'device_fraud_count', 'month', 'payment_type_AA', 'payment_type_AB', 'payment_type_AC', 'payment_type_AD', 'payment_type_AE', 'employment_status_CA', 'employment_status_CB', 'employment_status_CC', 'employment_status_CD', 'employment_status_CE', 'employment_status_CF', 'employment_status_CG', 'housing_status_BA', 'housing_status_BB', 'housing_status_BC', 'housing_status_BD', 'housing_status_BE', 'housing_status_BF', 'housing_status_BG', 'source_INTERNET', 'source_TELEAPP', 'device_os_linux', 'device_os_macintosh', 'device_os_other', 'device_os_windows', 'device_os_x11']
    columns = list(df.columns)
    credit = df.to_numpy()
    np.random.seed(seed)
    fraud_index = np.nonzero(credit[:,0]==1)[0]
    non_fraud_index = np.nonzero(credit[:,0]==0)[0]
    random_nfraud = np.random.choice(non_fraud_index, size=fraud_index.shape[0])
    combined_index = np.append(fraud_index, random_nfraud, axis=0)
    X = credit[combined_index,1:]
    # Binary Age
    X[:,4] = X[:, 4]<=30
    y = np.abs(credit[combined_index,0]-1)
    Xy = np.append(X, y[:, None], axis=1)
    class_names = ["Fraud", "No Fraud"]
    disc_index = 4 # sex (1: <=30 years, 0: >30 years)
    return X, y, Xy, columns, class_names, disc_index