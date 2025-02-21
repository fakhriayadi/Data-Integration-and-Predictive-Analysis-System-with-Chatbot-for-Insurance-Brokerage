# prediction_chiffre_affaire_date_souscription.py

import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Chargement des données
df_gamme_ca_affg = pd.read_excel('C:/Users/dell/OneDrive/Bureau/9raya_esprit/PFE/Fichiers_Analyse_Prédictive/df_gamme_ca_affg.xls')

# Convertir la colonne 'Date' en un format compatible avec la régression linéaire
df_gamme_ca_affg['Date'] = pd.to_datetime(df_gamme_ca_affg['Date'])
df_gamme_ca_affg['Date_numeric'] = (df_gamme_ca_affg['Date'] - df_gamme_ca_affg['Date'].min()) / np.timedelta64(1, 'D')

# Créer un modèle de régression linéaire pour 'mca' (Chiffre d'affaires moyen)
model_mca = LinearRegression()

# Séparer les données en données d'entraînement pour 'mca'
X_train_mca = df_gamme_ca_affg[df_gamme_ca_affg['Date'] < '2024-01-31']['Date_numeric'].values.reshape(-1, 1)
y_train_mca = np.log1p(df_gamme_ca_affg[df_gamme_ca_affg['Date'] < '2024-01-31']['mca'])

# Adapter le modèle aux données d'entraînement pour 'mca'
model_mca.fit(X_train_mca, y_train_mca)

def predict_revenue(selected_date):
    days_since_min_date = (selected_date - df_gamme_ca_affg['Date'].min()) / np.timedelta64(1, 'D')
    days_since_min_date = np.array([[days_since_min_date]])
    predicted_log_revenue = model_mca.predict(days_since_min_date)
    predicted_revenue = np.expm1(predicted_log_revenue)
    rounded_predicted_revenue = round(predicted_revenue[0], 2)
    return f"{rounded_predicted_revenue:.2f} Euros"  # Return the rounded prediction to two decimal places concatenated with "Euros"
