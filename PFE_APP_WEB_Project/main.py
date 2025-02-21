from flask import Flask, render_template, request, redirect, session, flash , jsonify
import mysql.connector
import os
#import pickle
#from model import preprocess_input, model
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# -------------------------Base de Données MYSQL---------------------------------------------

app = Flask(__name__)
app.secret_key = os.urandom(24)


conn = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",
    database="pfe"
)
#users
cursor = conn.cursor()
#model = pickle.load(open("model.pkl", "rb"))

# création de la table users
cursor.execute("""CREATE TABLE IF NOT EXISTS users (
                id INT AUTO_INCREMENT PRIMARY KEY,
                Name VARCHAR(255) NOT NULL,
                email VARCHAR(255) NOT NULL,
                password VARCHAR(255) NOT NULL)""")




# -------------------------Prediction Nombre d'affaires par date de souscription---------------------------------------------

from flask import Flask, request, render_template
from prediction_nombre_affaire_date_souscription import predict_transactions
import pandas as pd

@app.route('/predict2', methods=['GET', 'POST'])
def predictt():
    if request.method == 'POST':
        selected_date = pd.to_datetime(request.form['selected_date'])
        rounded_predicted_transactions = predict_transactions(selected_date)
        return str(rounded_predicted_transactions) + " Affaires"  # Return the predicted number of transactions as a string

    return render_template('Prediction_date_souscription.html')  # Redirect to Prediction_date_souscription.html if the method is not POST

# -------------------------Prediction Chiffre d'affaires par date de souscription---------------------------------------------

from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


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



from prediction_chiffre_affaire_date_souscription import predict_revenue

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        selected_date = pd.to_datetime(request.form['selected_date'])
        predicted_revenue = predict_revenue(selected_date)
        return predicted_revenue

    return render_template('Prediction_date_souscription.html')  # Redirect to Prediction_date_souscription.html if the method is not POST


#-------------------------------------------------------------------------------------------

# -------------------------Prediction Chiffre d'affaires par date et gamme---------------------------------------------

@app.route('/predict4', methods=['GET','POST'])
def predict4():
    gammes = df_gamme_ca_affg['nom_gamme_prod'].unique().tolist()
    predicted_mca = None
    form_id = str(uuid.uuid4())  # Générer un identifiant unique

    if request.method == 'POST':
        selected_date = pd.to_datetime(request.form['selected_date'])
        selected_gamme = request.form['selected_gamme']

        # Convertir la date en un format compatible avec la régression linéaire
        selected_date_numeric = (selected_date - df_gamme_ca_affg['Date'].min()) / np.timedelta64(1, 'D')
        selected_date_numeric = np.array([[selected_date_numeric]])

        # Filtrer les données pour la gamme sélectionnée
        filtered_df = df_gamme_ca_affg[df_gamme_ca_affg['nom_gamme_prod'] == selected_gamme]

        # Prévoir le chiffre d'affaires pour la date sélectionnée
        predicted_mca_log = model_mca.predict(selected_date_numeric)
        predicted_mca = round(np.expm1(predicted_mca_log)[0], 2)


        print(f"Le chiffre d'affaires prédit selon Gamme et Date est : {predicted_mca} Euros")

    return render_template('Prediction_date_gamme.html', gammes=gammes, prediction=predicted_mca, form_id=form_id)


# -------------------------Prediction Nombre d'affaires par date et gamme---------------------------------------------

from flask import Flask, render_template, request
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import uuid


# Chargement des données
nb_aff_date_gamme_copie = pd.read_excel('C:/Users/dell/OneDrive/Bureau/9raya_esprit/PFE/Fichiers_Analyse_Prédictive/nb_aff_date_gamme_copie.xls')

gamme_list1 = nb_aff_date_gamme_copie['Nom_Gamme_Produit'].unique().tolist()
print(gamme_list1)


# Convertir la colonne 'Date' en un format compatible avec la régression linéaire
nb_aff_date_gamme_copie['Date'] = pd.to_datetime(nb_aff_date_gamme_copie['Date'])
nb_aff_date_gamme_copie['Date_numeric'] = (nb_aff_date_gamme_copie['Date'] - nb_aff_date_gamme_copie['Date'].min()) / np.timedelta64(1, 'D')

# Créer un modèle de régression linéaire pour 'mca' (Chiffre d'affaires moyen)
model_nb = LinearRegression()

# Séparer les données en données d'entraînement pour 'mca'
X_train_nb1 = nb_aff_date_gamme_copie[nb_aff_date_gamme_copie['Date'] < '2024-01-31']['Date_numeric'].values.reshape(-1, 1)
y_train_nb1 = np.log1p(nb_aff_date_gamme_copie[nb_aff_date_gamme_copie['Date'] < '2024-01-31']['Nombre_affaire'])

# Adapter le modèle aux données d'entraînement pour 'mca'
model_nb.fit(X_train_nb1, y_train_nb1)


import uuid

@app.route('/predict5', methods=['GET','POST'])
def predict5():
    gammes = nb_aff_date_gamme_copie['Nom_Gamme_Produit'].unique().tolist()
    predicted_nb = None
    form_id1 = str(uuid.uuid4())  # Générer un identifiant unique
    selected_gamme = None
    formatted_date = None
    if request.method == 'POST':
        selected_date = pd.to_datetime(request.form['selected_date'])
        selected_gamme = request.form['selected_gamme']

        # Convertir la date en un format compatible avec la régression linéaire
        selected_date_numeric = (selected_date - nb_aff_date_gamme_copie['Date'].min()) / np.timedelta64(1, 'D')
        selected_date_numeric = np.array([[selected_date_numeric]])
        # Filtrer les données pour la gamme sélectionnée
        filtered_df1 = nb_aff_date_gamme_copie[nb_aff_date_gamme_copie['Nom_Gamme_Produit'] == selected_gamme]

       # Prévoir le chiffre d'affaires pour la date sélectionnée
        predicted_nb_log = model_nb.predict(selected_date_numeric)
        predicted_nb = np.expm1(predicted_nb_log)[0]
        predicted_nb = round(predicted_nb)
        print(f"Le nombre d'affaires prédit est : {predicted_nb} Affaires")

        # Formater la date au format jour mois année
        formatted_date = selected_date.strftime('%d %B %Y').lstrip('0')

    return render_template('Prediction_date_gamme.html', gammes=gammes, prediction1=predicted_nb, form_id=form_id1, selected_gamme=selected_gamme, selected_date=formatted_date)



# -------------------------Prediction Chiffre d'affaire par Date et Courtier---------------------------------------------


# Charger les données à partir du fichier excel
data_courtier_nb_aff_cleaned_cl = pd.read_excel(r'C:/Users/dell/OneDrive/Bureau/9raya_esprit/PFE/Fichiers_Analyse_Prédictive/data_courtier_nb_aff_cleaned_cl.xls')

courtier_list = data_courtier_nb_aff_cleaned_cl['courtier'].unique().tolist()
print(courtier_list)

# Convertir la colonne 'Date' en un format compatible avec la régression linéaire
data_courtier_nb_aff_cleaned_cl['date'] = pd.to_datetime(data_courtier_nb_aff_cleaned_cl['date'])
data_courtier_nb_aff_cleaned_cl['Date_numeric'] = (data_courtier_nb_aff_cleaned_cl['date'] - data_courtier_nb_aff_cleaned_cl['date'].min()) / np.timedelta64(1, 'D')

# Créer un modèle de régression linéaire pour 'mca' (Chiffre d'affaires moyen)
model_mca_courtier = LinearRegression()

# Séparer les données en données d'entraînement pour 'mca'
X_train_mca_courtier = data_courtier_nb_aff_cleaned_cl[data_courtier_nb_aff_cleaned_cl['date'] < '2024-01-19']['Date_numeric'].values.reshape(-1, 1)
y_train_mca_courtier = np.log1p(data_courtier_nb_aff_cleaned_cl[data_courtier_nb_aff_cleaned_cl['date'] < '2024-01-19']['mca'])

# Adapter le modèle aux données d'entraînement pour 'mca'
model_mca_courtier.fit(X_train_mca_courtier, y_train_mca_courtier)

import uuid


@app.route('/predict6', methods=['GET', 'POST'])
def predict6():
    courtiers = data_courtier_nb_aff_cleaned_cl['courtier'].unique().tolist()
    predicted_mca_courtier = None
    form_id2 = str(uuid.uuid4())  # Générer un identifiant unique
    selected_courtier = None
    formatted_date = None
    if request.method == 'POST':
        selected_date = pd.to_datetime(request.form['selected_date'])
        selected_courtier = request.form['selected_courtier']

        # Convertir la date en un format compatible avec la régression linéaire
        selected_date_numeric = (selected_date - data_courtier_nb_aff_cleaned_cl['date'].min()) / np.timedelta64(1, 'D')
        selected_date_numeric = np.array([[selected_date_numeric]])
        # Filtrer les données pour la gamme sélectionnée
        filtered_df = data_courtier_nb_aff_cleaned_cl[data_courtier_nb_aff_cleaned_cl['courtier'] == selected_courtier]

        # Prévoir le chiffre d'affaires pour la date sélectionnée
        predicted_mca_log_courtier = model_mca_courtier.predict(selected_date_numeric)
        predicted_mca_courtier = np.expm1(predicted_mca_log_courtier)[0]
        predicted_mca_courtier = round(predicted_mca_courtier, 3)
        print(f"Le chiffre d'affaires prédit selon courtier et Date est : {predicted_mca_courtier} Euros")

        # Formater la date au format jour mois année
        formatted_date = selected_date.strftime('%d %B %Y').lstrip('0')
    return render_template('Prediction_date_courtier.html', courtiers=courtiers, prediction=predicted_mca_courtier, form_id=form_id2,
                           selected_courtier=selected_courtier, selected_date=formatted_date)


# -------------------------Prediction Nombre d'affaire par Date et Courtier---------------------------------------------


# Charger les données à partir du fichier excel
df_courtier_nb_aff = pd.read_excel(r'C:\Users\dell\OneDrive\Bureau\9raya_esprit\PFE\Fichiers_Analyse_Prédictive\data_courtier_nb_aff_cleaned_cl.xls')

courtier_list1 = df_courtier_nb_aff['courtier'].unique().tolist()
#print(courtier_list1)


# Convertir la colonne 'Date' en un format compatible avec la régression linéaire
df_courtier_nb_aff['date'] = pd.to_datetime(df_courtier_nb_aff['date'])
df_courtier_nb_aff['Date_numeric'] = (df_courtier_nb_aff['date'] - df_courtier_nb_aff['date'].min()) / np.timedelta64(1, 'D')

# Créer un modèle de régression linéaire pour 'mca' (Chiffre d'affaires moyen)
model_nb_courtier = LinearRegression()

# Séparer les données en données d'entraînement pour 'mca'
X_train_nb1_courtier = df_courtier_nb_aff[df_courtier_nb_aff['date'] < '2024-01-19']['Date_numeric'].values.reshape(-1, 1)
y_train_nb1_courtier = np.log1p(df_courtier_nb_aff[df_courtier_nb_aff['date'] < '2024-01-19']['Nombre_affaire'])

# Adapter le modèle aux données d'entraînement pour 'mca'
model_nb_courtier.fit(X_train_nb1_courtier, y_train_nb1_courtier)


import uuid

@app.route('/predict7', methods=['GET','POST'])
def predict7():
    courtiers = df_courtier_nb_aff['courtier'].unique().tolist()
    predicted_nb = None
    form_id3 = str(uuid.uuid4())  # Générer un identifiant unique
    selected_courtier = None
    formatted_date = None
    if request.method == 'POST':
        selected_date = pd.to_datetime(request.form['selected_date'])
        selected_courtier = request.form['selected_courtier']

        # Convertir la date en un format compatible avec la régression linéaire
        selected_date_numeric = (selected_date - df_courtier_nb_aff['date'].min()) / np.timedelta64(1, 'D')
        selected_date_numeric = np.array([[selected_date_numeric]])
        # Filtrer les données pour la gamme sélectionnée
        filtered_df1 = df_courtier_nb_aff[df_courtier_nb_aff['courtier'] == selected_courtier]

       # Prévoir le chiffre d'affaires pour la date sélectionnée
        predicted_nb_log = model_nb_courtier.predict(selected_date_numeric)
        predicted_nb = np.expm1(predicted_nb_log)[0]
        predicted_nb = round(predicted_nb)
        print(f"Le nombre d'affaires prédit est : {predicted_nb} ")

        # Formater la date au format jour mois année
        formatted_date = selected_date.strftime('%d %B %Y').lstrip('0')

    return render_template('Prediction_date_courtier.html', courtiers=courtiers, prediction1=predicted_nb, form_id=form_id3, selected_courtier=selected_courtier, selected_date=formatted_date)


# -------------------------Routes---------------------------------------------


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        return redirect('/home')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def about():
   if request.method == 'POST':
        return redirect('/')
   return render_template('register.html')


@app.route('/home', methods=['GET', 'POST'])
def home():
    if 'id' in session:
        return render_template('home.html')
    else:
        return redirect('/')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')


@app.route('/login_validation', methods=['POST'])
def login_validation():
    email = request.form.get('email')
    password = request.form.get('password')

    cursor.execute("""SELECT * FROM users WHERE email = %s AND password = %s""", (email, password))
    users = cursor.fetchall()
    if len(users) > 0:
        session['id']=users[0][0]
        return redirect('/home')
    else:
        flash('please check your login details and try again')
        return redirect('/')

@app.route('/add_user', methods=['POST'])
def add_user():
    name=request.form.get('uname')
    email=request.form.get('uemail')
    password=request.form.get('upassword')

    cursor.execute("""INSERT INTO users (id,`name`,`email`,`password`) VALUES (NULL,'{}','{}','{}')""".format(name,email,password))
    conn.commit()
    return redirect('/')

@app.route('/logout')
def logout():
    if 'id' in session:
        session.pop('id')
    return redirect('/')


#----------------------------Chatbot----------------------------------------


from flask import Flask, request
from chatbot import chatbot_response
@app.route("/get")
def get_bot_response():
    user_input = request.args.get("msg")
    response = chatbot_response(user_input)
    return str(response)



if __name__ == '__main__':
    app.run(debug=True)


