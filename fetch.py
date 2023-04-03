import numpy as np
from aif360.sklearn.datasets import fetch_adult
from aif360.sklearn.datasets import fetch_bank
from aif360.sklearn.datasets import fetch_compas
from aif360.sklearn.datasets import fetch_german
from aif360.sklearn.datasets import fetch_meps

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
