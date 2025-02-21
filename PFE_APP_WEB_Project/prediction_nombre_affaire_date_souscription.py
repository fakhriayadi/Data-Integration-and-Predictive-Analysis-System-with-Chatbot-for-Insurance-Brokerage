# prediction_nombre_affaire_date_souscription.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load the sum_nb_aff file
sum_nb_aff = pd.read_excel('C:/Users/dell/OneDrive/Bureau/9raya_esprit/PFE/Fichiers_Analyse_Prédictive/sum_nb_aff.xls')

# Create train and test sets for number of transactions prediction
X_transactions = (sum_nb_aff[sum_nb_aff['date_souscription_aff'] < '2024-01-19']['date_souscription_aff'] - sum_nb_aff['date_souscription_aff'].min()) / np.timedelta64(1, 'D')
X_transactions = X_transactions.values.reshape(-1, 1)
y_transactions = np.log1p(sum_nb_aff[sum_nb_aff['date_souscription_aff'] < '2024-01-19']['somme_nb_aff'])
X_train_transactions, X_test_transactions, y_train_transactions, y_test_transactions = train_test_split(X_transactions, y_transactions, test_size=0.3, random_state=42)

# Create and train the linear regression model for number of transactions
model_transactions = LinearRegression()
model_transactions.fit(X_train_transactions, y_train_transactions)

def predict_transactions(selected_date):
    days_since_min_date = (selected_date - sum_nb_aff['date_souscription_aff'].min()) / np.timedelta64(1, 'D')
    days_since_min_date = np.array([[days_since_min_date]])
    predicted_log_transactions = model_transactions.predict(days_since_min_date)
    predicted_transactions = np.expm1(predicted_log_transactions)
    rounded_predicted_transactions = int(round(predicted_transactions[0]))
    return rounded_predicted_transactions
