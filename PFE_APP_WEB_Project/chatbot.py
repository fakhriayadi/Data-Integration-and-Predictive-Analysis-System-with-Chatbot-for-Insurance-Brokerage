# -------------------------Chatbot---------------------------------------------

#################################################################################################################
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
import pandas as pd

# Charger le dataframe df_gamme_ca_affg
df_gamme_ca_affg = pd.read_excel(
    r'C:/Users/dell/OneDrive/Bureau/9raya_esprit/PFE/Fichiers_Analyse_Prédictive/df_gamme_ca_affg.xls')

#####################################################################################################################

gamme_list = df_gamme_ca_affg['nom_gamme_prod'].unique().tolist()
print(gamme_list)
print(df_gamme_ca_affg['nom_gamme_prod'])

# Charger les DataFrames à partir des fichiers Excel
#Dataframe_Visuels

excel_path_courtier = r'C:\Users\dell\OneDrive\Bureau\9raya_esprit\PFE\Fichiers_vue_ensemble\df_Aff_Courtier.xls'
df_courtier = pd.read_excel(excel_path_courtier)

excel_path_etat = r'C:\Users\dell\OneDrive\Bureau\9raya_esprit\PFE\Fichiers_vue_ensemble\df_Aff_Etat.xls'
df_etat = pd.read_excel(excel_path_etat)

excel_path_tag = r'C:\Users\dell\OneDrive\Bureau\9raya_esprit\PFE\Fichiers_vue_ensemble\df_Aff_Tag.xls'
df_tag = pd.read_excel(excel_path_tag)

excel_path_gamme = r'C:\Users\dell\OneDrive\Bureau\9raya_esprit\PFE\Fichiers_vue_ensemble\df_gamme.xls'
df_gamme = pd.read_excel(excel_path_gamme)

excel_path_code_dep = r'C:\Users\dell\OneDrive\Bureau\9raya_esprit\PFE\Fichiers_vue_ensemble\df_code_dep.xls'
df_code_dep = pd.read_excel(excel_path_code_dep)


excel_path_nb_aff_Tranche = r'C:\Users\dell\OneDrive\Bureau\9raya_esprit\PFE\Fichiers_vue_ensemble\df_tranche_age.xls'
df_aff_Tranche = pd.read_excel(excel_path_nb_aff_Tranche)

excel_path_montant_echeancier_date = r'C:\Users\dell\OneDrive\Bureau\9raya_esprit\PFE\Fichiers_vue_ensemble\df_montant_echeancier_par_date.xls'
df_montant_echeancier = pd.read_excel(excel_path_montant_echeancier_date)

excel_path_opp_date = r'C:\Users\dell\OneDrive\Bureau\9raya_esprit\PFE\Fichiers_vue_ensemble\df_opportunites_annee_mois.xls'
df_opp_date = pd.read_excel(excel_path_opp_date)


excel_path_CA_5courtier = r'C:\Users\dell\OneDrive\Bureau\9raya_esprit\PFE\Fichiers_Courtiers\df_CA_5Courtiers.xls'
df_CA_5courtier = pd.read_excel(excel_path_CA_5courtier)

excel_path_CA_5users  = r'C:\Users\dell\OneDrive\Bureau\9raya_esprit\PFE\Fichiers_Courtiers\df_CA_5User.xls'
df_CA_5users = pd.read_excel(excel_path_CA_5users)


excel_path_user_courtiers  = r'C:\Users\dell\OneDrive\Bureau\9raya_esprit\PFE\Fichiers_Courtiers\df_Utilisateur_courtier.xls'
df_user_courtier = pd.read_excel(excel_path_user_courtiers)


excel_path_aff_produits = r'C:\Users\dell\OneDrive\Bureau\9raya_esprit\PFE\Fichiers_Gammes\df_aff_produit.xls'
df_aff_produits = pd.read_excel(excel_path_aff_produits)

excel_path_CA_aff_produits = r'C:\Users\dell\OneDrive\Bureau\9raya_esprit\PFE\Fichiers_Gammes\df_ca_aff_produit.xls'
df_CA_aff_produits = pd.read_excel(excel_path_CA_aff_produits)


excel_path_aff_compagnie = r'C:\Users\dell\OneDrive\Bureau\9raya_esprit\PFE\Fichiers_Gammes\df_aff_compagnie.xls'
df_aff_compagnie = pd.read_excel(excel_path_aff_compagnie)

excel_path_CA_compagnie = r'C:\Users\dell\OneDrive\Bureau\9raya_esprit\PFE\Fichiers_Gammes\df_ca_aff_compagnie.xls'
df_ca_aff_compagnie = pd.read_excel(excel_path_CA_compagnie)


excel_path_CA_aff_gamme = r'C:\Users\dell\OneDrive\Bureau\9raya_esprit\PFE\Fichiers_Gammes\df_ca_aff_gamme.xls'
df_ca_aff_gamme= pd.read_excel(excel_path_CA_aff_gamme)

excel_path_aff_utilisateurs = r'C:\Users\dell\OneDrive\Bureau\9raya_esprit\PFE\Fichiers_Courtiers\df_Aff_Utilisateur.xls'
df_aff_utilisateurs= pd.read_excel(excel_path_aff_utilisateurs)


excel_path_utilisateurs_civilite = r'C:\Users\dell\OneDrive\Bureau\9raya_esprit\PFE\Fichiers_Courtiers\df_user_civilite.xls'
df_user_civilite= pd.read_excel(excel_path_utilisateurs_civilite)







#-----------------------------------------------------------------------------------------------
#Dataframe_KPIS
#Chiffre d'affaire par Année
excel_path_ca_annee = r'C:\Users\dell\OneDrive\Bureau\9raya_esprit\PFE\Fichiers_vue_ensemble\KPIS\df_Chiffre_affaires_année.xls'
df_ca_ann = pd.read_excel(excel_path_ca_annee)
#Chiffre d'affaire par Mois
excel_path_ca_mois = r'C:\Users\dell\OneDrive\Bureau\9raya_esprit\PFE\Fichiers_vue_ensemble\KPIS\df_CA_aff_mois.xls'
df_ca_mois = pd.read_excel(excel_path_ca_mois)
#Chiffre d'affaire par Date de Souscription
excel_path_ca_date_souscription = r'C:\Users\dell\OneDrive\Bureau\9raya_esprit\PFE\Fichiers_vue_ensemble\KPIS\df_CA_Date_souscription.xls'
df_ca_souscription = pd.read_excel(excel_path_ca_date_souscription)

#Nombre d'affaires résiliés par Année
excel_path_aff_resilies = r'C:\Users\dell\OneDrive\Bureau\9raya_esprit\PFE\Fichiers_vue_ensemble\KPIS\df_Affaire_resilies_annee.xls'
df_Affaire_resilies_annee = pd.read_excel(excel_path_aff_resilies)

#Nombre d'affaires par Année
excel_path_nb_aff_annee = r'C:\Users\dell\OneDrive\Bureau\9raya_esprit\PFE\Fichiers_vue_ensemble\KPIS\df_nombreaffaires_année.xls'
df_nb_aff_annee = pd.read_excel(excel_path_nb_aff_annee)

#TauxConversion par Année
excel_path_TauxConversion_annee  = r'C:\Users\dell\OneDrive\Bureau\9raya_esprit\PFE\Fichiers_vue_ensemble\KPIS\df_TauxConversion_année.xls'
df_TauxConversion_annee = pd.read_excel(excel_path_TauxConversion_annee)

#TauxConversion par Mois
excel_path_TauxConversion_mois  = r'C:\Users\dell\OneDrive\Bureau\9raya_esprit\PFE\Fichiers_vue_ensemble\KPIS\df_tauxconversion_mois.xls'
df_tauxconversion_mois = pd.read_excel(excel_path_TauxConversion_mois)


#Affaire par Cycle Produit
excel_path_Aff_Cycle_prod  = r'C:\Users\dell\OneDrive\Bureau\9raya_esprit\PFE\Fichiers_vue_ensemble\KPIS\df_concatenated_aff_cycle_prod.xls'
df_aff_cycle_prod = pd.read_excel(excel_path_Aff_Cycle_prod)

#Chiffre d'Affaire par Cycle Produit
excel_path_CA_Cycle_prod  = r'C:\Users\dell\OneDrive\Bureau\9raya_esprit\PFE\Fichiers_vue_ensemble\KPIS\df_concatenated_ca_cycle_prod.xls'
df_CA_cycle_prod = pd.read_excel(excel_path_CA_Cycle_prod)


#Affaire par Type Contrat
excel_path_Aff_type_contrat  = r'C:\Users\dell\OneDrive\Bureau\9raya_esprit\PFE\Fichiers_vue_ensemble\KPIS\df_concatenated_aff_type_contrat.xls'
df_aff_type_contrat = pd.read_excel(excel_path_Aff_type_contrat)

#Chiffre d'Affaire par Type Contrat
excel_path_CA_type_contrat  = r'C:\Users\dell\OneDrive\Bureau\9raya_esprit\PFE\Fichiers_vue_ensemble\KPIS\df_concatenated_mca_type_contrat.xls'
df_CA_type_contrat = pd.read_excel(excel_path_CA_type_contrat)








#-------------------------------------------------------------------------------


#KPIS

# Créer des paires de questions-réponses pour le chatbot pour KPI (Chiffre d'Affaire par Année )
nb_ca_ann_pairs = []
for index, row in df_ca_ann.iterrows():
    annee = int(row['annee'])
    question = f"Chiffre d'affaires pour l'année {annee}"
    answer = [f"Le chiffre d'affaires par année {annee} est {row['mca']} Euros"]
    nb_ca_ann_pairs.append([question, answer])
print(nb_ca_ann_pairs)


# Créer des paires de questions-réponses pour le chatbot pour KPI (Chiffre d'Affaire par Mois )
nb_ca_mois_pairs = []
for index, row in df_ca_mois.iterrows():
    question = f"Chiffre d'affaires pour le mois {row['Mois']}"
    answer = [f"Le chiffre d'affaires par mois {row['Mois']} est {row['mca']} Euros"]
    nb_ca_mois_pairs.append([question, answer])


# Créer des paires de questions-réponses pour le chatbot pour le chiffre d'affaires par date de souscription
df_ca_souscription_pairs = []
for index, row in df_ca_souscription.iterrows():
    question = f"Donne-moi la somme du chiffre d'affaires par date de souscription {row['date_souscription_aff']}"
    answer = [f"Le chiffre d'affaires pour {row['date_souscription_aff']} est {row['somme_mca']} Euros"]
    df_ca_souscription_pairs.append([question, answer])



# Créer des paires de questions-réponses pour le chatbot pour le nombre d'affaires résiliées par Année
nb_Aff_Res_ann_pairs = []
for index, row in df_Affaire_resilies_annee.iterrows():
    question = f"Nombre d'affaires résiliées pour l'année {int(row['annee'])}"
    answer = [f"Le nombre d'affaires résiliées pour l'année {int(row['annee'])} est {row['Nombre_affaires_resiliees']}"]
    nb_Aff_Res_ann_pairs.append([question, answer])

print(nb_Aff_Res_ann_pairs)



# Créer des paires de questions-réponses pour le chatbot pour KPI (Nombre Affaire par Année )
nb_aff_ann_pairs = []
for index, row in df_nb_aff_annee.iterrows():
    question = f"nombre d'affaires pour l'année {row['annee']}"
    answer = [f"Le nombre d'affaires par année {row['annee']} est {row['Nombre_Affaires']}"]
    nb_aff_ann_pairs.append([question, answer])
print(nb_aff_ann_pairs)


# Créer des paires de questions-réponses pour le chatbot pour KPI (Taux Conversion par Année)

# Créer des paires de questions-réponses pour le chatbot pour KPI (Taux Conversion par Année)
nb_TC_ann_pairs = []
for index, row in df_TauxConversion_annee.iterrows():
    question = f"Taux Conversion des opportunités en Affaires pour l'année {int(row['annee'])}"
    taux_conversion = float(row['Taux_Conversion_Opportunite_Affairee']) * 100
    answer = [f"Taux Conversion des opportunités en Affaires par année {int(row['annee'])} est {taux_conversion:.2f}%"]
    nb_TC_ann_pairs.append([question, answer])
print(nb_TC_ann_pairs)



# Créer des paires de questions-réponses pour le chatbot pour KPI (Taux Conversion par Mois )
nb_TC_mois_pairs = []
for index, row in df_tauxconversion_mois.iterrows():
    question = f"Taux Conversion des opportunités en Affaires pour le mois {row['Mois']}"
    taux_conversion = float(row['Taux_Conversion_Opportunite_Affaire']) * 100
    answer = [f"Taux Conversion des opportunités en Affaires par mois {row['Mois']} est {taux_conversion:.2f}%"]
    nb_TC_mois_pairs.append([question, answer])
print(nb_TC_mois_pairs)



# Créer des paires de questions-réponses pour le chatbot pour KPI (Nombre d'affaires par Cycle Produit )
aff_cycle_prod_pairs = []
for index, row in df_aff_cycle_prod.iterrows():
    question = f"nombre d'affaires pour le cycle produit {row['cycle_prod']}"
    answer = [f"Le nombre d'affaires par cycle produit est {row['Nombre_affaires']}"]
    aff_cycle_prod_pairs.append([question, answer])
print(aff_cycle_prod_pairs)

# Créer des paires de questions-réponses pour le chatbot pour KPI (Chiffre d'affaires par Cycle Produit )
ca_cycle_prod_pairs = []
for index, row in df_CA_cycle_prod.iterrows():
    question = f"chiffre d'affaires pour le cycle produit {row['cycle_prod']}"
    answer = [f"Le chiffre d'affaires par cycle produit est {row['mca']} Euros"]
    ca_cycle_prod_pairs.append([question, answer])

print(ca_cycle_prod_pairs)



# Créer des paires de questions-réponses pour le chatbot pour KPI (Nombre d'affaires par Type Contrat)
aff_type_contrat_pairs = []
for index, row in df_aff_type_contrat.iterrows():
    question = f"nombre d'affaires pour le type contrat{row['type_contrat']}"
    answer = [f"Le nombre d'affaires par type contrat est {row['Nombre_affaires']}"]
    aff_type_contrat_pairs.append([question, answer])

print(aff_type_contrat_pairs)



# Créer des paires de questions-réponses pour le chatbot pour KPI (Chiffre d'affaires par Type Contrat )
ca_type_contrat_pairs = []
for index, row in df_CA_type_contrat.iterrows():
    question = f"chiffre d'affaires pour le type contrat{row['type_contrat']}"
    answer = [f"Le chiffre d'affaires par type contrat est {row['mca']} Euros"]
    ca_type_contrat_pairs.append([question, answer])

print(ca_type_contrat_pairs)





#-------------------------------------------------------------------------------

#Visuels Vue Ensemble

# Créer des paires de questions-réponses pour le chatbot pour les courtiers
courtier_pairs = []
for index, row in df_courtier.iterrows():
    question = f"nombre d'affaires pour le courtier {row['Courtier']}"
    #answer = [str(row['Nombre d\'Affaires'])]
    answer = [f"Le nombre d'affaires pour le courtier {row['Courtier']} est {row['Nombre_Affaires']}"]
    courtier_pairs.append([question, answer])



# Créer des paires de questions-réponses pour le chatbot pour les états
etat_pairs = []
for index, row in df_etat.iterrows():
    question = f"nombre d'affaires pour l'etat {row['Etat']}"
    answer = [f"Le nombre d'affaires pour l'etat {row['Etat']} est {row['Nombre_Affaires']}"]
    #answer = [str(row['Nombre d\'Affaires'])]
    etat_pairs.append([question, answer])


# Créer des paires de questions-réponses pour le chatbot pour les Tags
tag_pairs = []
for index, row in df_tag.iterrows():
    question = f"nombre d'affaires pour le Tag {row['Tag']}"
    answer = [f"Le nombre d'affaires par tag est {row['Nombre_Affaires']}"]
    tag_pairs.append([question, answer])


# Créer des paires de questions-réponses pour le chatbot pour les code département
code_dep_pairs = {}
for index, row in df_code_dep.iterrows():
    code_dep_pairs[f"nombre d'affaires pour le code département {row['Code_Departement']}"] = row['Nombre_Affaires']

print(code_dep_pairs)


# Convertir le dictionnaire en une liste de paires clé-valeur
code_dep_pairs_list = [[question, answer] for question, answer in code_dep_pairs.items()]







# Créer des paires de questions-réponses pour le chatbot pour les gammes
gamme_pairs = []
for index, row in df_gamme.iterrows():
    question = f"nombre d'affaires pour la gamme {row['Gamme']}"
    answer = [f"Le nombre d'affaires pour la gamme {row['Gamme']} est {row['Nombre_Affaires']}"]
    gamme_pairs.append([question, answer])


# Créer des paires de questions-réponses pour le chatbot pour les montant écheancier par date
df_montant_echeancier.rename(columns={'Date ': 'Date'}, inplace=True)

montant_echeancier_pairs = []
for index, row in df_montant_echeancier.iterrows():
    question = f"montant échéancier pour la date {row['Date']} de l'année {row['Annee']} et du mois de {row['Mois']}"
    answer = [f"Le montant échéancier pour la date {row['Date']} de l'année {row['Annee']} et du mois de {row['Mois']} est {row['montant_total_echeancier']} Euros"]
    montant_echeancier_pairs.append([question, answer])
print(montant_echeancier_pairs)
print(df_montant_echeancier)




# Convertir la colonne 'Date' en format datetime
df_opp_date['Date'] = pd.to_datetime(df_opp_date['Date'])

# Créer des paires de questions-réponses pour le chatbot pour les opportunités par Date
opp_date_pairs = []
for index, row in df_opp_date.iterrows():
    # Formater la date pour ne conserver que l'année, le mois et le jour
    date_formatted = row['Date'].strftime('%Y-%m-%d')
    question = f"Donne-moi le nombre d'opportunites pour la date {date_formatted} de l'année {row['Annee']} et du mois de {row['Mois']}"
    answer = [f"Le nombre d'opportunites pour la date {date_formatted} de l'année {row['Annee']} et du mois de {row['Mois']} est {row['Nombre_opportunites']}"]
    opp_date_pairs.append([question, answer])

print(opp_date_pairs)


# Créer des paires de questions-réponses pour le chatbot pour la tranche d'age
Tranche_pairs = []
for index, row in df_aff_Tranche.iterrows():
    question = f"nombre d'affaires pour la tranche d'age des prospects {row['Tranche_Age']}"
    answer = [f"Le nombre d'affaires pour la tranche d'age des prospects est {row['Nombre_affaires']}"]
    Tranche_pairs.append([question, answer])
print(Tranche_pairs)





#-------------------------------------------------------------------------------

#Visuels Page Courtiers

# Créer des paires de questions-réponses pour le chatbot pour les Produits
utilisateurs_pairs = []
for index, row in df_aff_utilisateurs.iterrows():
    question = f"nombre d'affaires pour l'utilisateur {row['Utilisateur']}"
    answer = [f"Le nombre d'affaires par utilisateur est {row['Nombre_Affaires']}"]
    utilisateurs_pairs.append([question, answer])

print(utilisateurs_pairs)



# Créer des paires de questions-réponses pour le chatbot pour les top 5 courtiers avec leur chiffre d'affaire
top_courtiers_pairs = []
question = "Donne-moi la liste des top 5 organismes de courtage avec leurs chiffre d'affaire"
answer = ""
for index, row in df_CA_5courtier.iterrows():
    answer += f"{row['Top_5_Courtiers']} a un chiffre d'affaire de {row['Chiffre_Affaire']} Euros et "
answer = answer[:-4]  # Supprimer le dernier " et "
top_courtiers_pairs.append([question, [answer]])

print(top_courtiers_pairs)




# Créer des paires de questions-réponses pour le chatbot pour les top 5 courtiers avec leur chiffre d'affaire
top_users_pairs = []
question = "Donne-moi la liste des top 5 utilisateurs avec leurs chiffre d'affaire"
answer = ""
for index, row in df_CA_5users.iterrows():
    answer += f"{row['Top_5_Utilisateur']} a un chiffre d'affaire de {row['Chiffre_Affaire']} Euros et "
answer = answer[:-4]  # Supprimer le dernier " et "
top_users_pairs.append([question, [answer]])

print(top_users_pairs)


# Créer des paires de questions-réponses pour le chatbot pour les users pour chaque  courtier
users_civilite_pairs = []
for index, row in df_user_civilite.iterrows():
    question = f"nombre d'utilisateurs pour la civilite {row['civilite']}"
    answer = [f"Le nombre d'utilisateurs pour la civilite {row['civilite']} est {row['Nombre_utilisateurs']}"]
    users_civilite_pairs.append([question, answer])

print(users_civilite_pairs)



# Créer des paires de questions-réponses pour le chatbot pour les users pour chaque  courtier
users_courtiers_pairs = []
for index, row in df_user_courtier.iterrows():
    question = f"nombre d'utilisateurs pour le courtier {row['Courtier']}"
    answer = [f"Le nombre d'utilisateurs pour le courtier {row['Courtier']} est {row['Nombre_Utilisateurs']}"]
    users_courtiers_pairs.append([question, answer])

print(users_courtiers_pairs)




#-------------------------------------------------------------------------------

#Visuels Page Gammes

# Créer des paires de questions-réponses pour le chatbot pour les users pour chaque  courtier
aff_compagnie_pairs = []
for index, row in df_aff_compagnie.iterrows():
    question = f"nombre d'affaires pour la compagnie {row['Compagnie']}"
    answer = [f"Le nombre d'affaires pour la compagnie {row['Compagnie']} est {row['Nombre_Affaires']}"]
    aff_compagnie_pairs.append([question, answer])

print(aff_compagnie_pairs)


# Créer des paires de questions-réponses pour le chatbot pour les Produits
produits_pairs = []
for index, row in df_aff_produits.iterrows():
    question = f"nombre d'affaires pour le Produit {row['Produit']}"
    answer = [f"Le nombre d'affaires par Produit est {row['Nombre_Affaires']}"]
    produits_pairs.append([question, answer])

print(produits_pairs)



# Créer des paires de questions-réponses pour le chatbot pour les users pour chaque  courtier
ca_aff_compagnie_pairs = []
for index, row in df_ca_aff_compagnie.iterrows():
    question = f"Chiffre d'affaire pour la compagnie {row['Compagnie']}"
    answer = [f"Le Chiffre d'affaire pour la compagnie {row['Compagnie']} est {row['Chiffre_Affaire']} Euros "]
    ca_aff_compagnie_pairs.append([question, answer])

print(ca_aff_compagnie_pairs)


# Créer des paires de questions-réponses pour le chatbot pour les affaires pour chaque  Gamme
ca_gamme_pairs = []
for index, row in df_ca_aff_gamme.iterrows():
    question = f"Chiffre d'affaire pour la gamme {row['Gamme']}"
    answer = [f"Le Chiffre d'affaire pour la gamme {row['Gamme']} est {row['Chiffre_Affaire']} Euros"]
    ca_gamme_pairs.append([question, answer])

print(ca_gamme_pairs)


# Créer des paires de questions-réponses pour le chatbot pour les Utilisateurs
CA_produits_pairs = []
for index, row in df_CA_aff_produits.iterrows():
    question = f"Chiffre d'affaire pour le Produit {row['Produit']}"
    answer = [f"Le Chiffre d'affaires par Produit est {row['Chiffre_Affaire']} Euros"]
    CA_produits_pairs.append([question, answer])

print(CA_produits_pairs)

#-------------------------------------------------------------------------------

# Paires de questions-réponses générales pour le chatbot
# Définir le chemin du fichier general_pairs.py
general_pairs_path = r'C:\Users\dell\OneDrive\Bureau\9raya_esprit\5èmeBI\4èmeBI8\Semestre2\Projet PI\Linkedin\Application web\fakhri_ayadi\pythonProject\general_pairs.py'

# Charger le contenu du fichier dans une variable locale
local_vars = {}
with open(general_pairs_path, 'r', encoding='utf-8') as file:
    exec(file.read(), {}, local_vars)

# Récupérer la variable general_pairs depuis le fichier chargé
general_pairs = local_vars['general_pairs']

# Concaténer la liste convertie avec les autres listes
merged_pairs = general_pairs + courtier_pairs + etat_pairs + tag_pairs + gamme_pairs + code_dep_pairs_list + nb_ca_ann_pairs + nb_ca_mois_pairs + df_ca_souscription_pairs + nb_Aff_Res_ann_pairs + nb_aff_ann_pairs + nb_TC_ann_pairs + nb_TC_mois_pairs + Tranche_pairs + montant_echeancier_pairs + opp_date_pairs + top_courtiers_pairs + top_users_pairs + users_courtiers_pairs + produits_pairs + CA_produits_pairs + ca_aff_compagnie_pairs + aff_compagnie_pairs + ca_gamme_pairs + utilisateurs_pairs + users_civilite_pairs + aff_cycle_prod_pairs + ca_cycle_prod_pairs + aff_type_contrat_pairs + ca_type_contrat_pairs

# ---------------------- NLP Chatbot ---------------------------------------
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from langdetect import detect
from collections import Counter

from fuzzywuzzy import fuzz

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('stopwords')
nltk.download('wordnet')


lemmatizer = WordNetLemmatizer()


def preprocess_text(text):
    tokens = word_tokenize(text)

    tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens]
    pos_tags = pos_tag(tokens)
    filtered_tokens = [word for word, tag in pos_tags if
                       word.lower() not in stopwords.words('french') and len(word) > 2]
    #normalized_tokens = [re.sub(r'\d+', 'NUM', token) for token in filtered_tokens]
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    language = detect(text)
    preprocessed_text = ' '.join(lemmatized_tokens)
    return preprocessed_text


def chatbot_response(user_input):
    preprocessed_input = preprocess_text(user_input)
    matched_response = None
    max_score = 0

    for question, answer in merged_pairs:
        question_preprocessed = preprocess_text(question)
        similarity_score = fuzz.ratio(preprocessed_input, question_preprocessed)
        if similarity_score > max_score:
            max_score = similarity_score
            matched_response = answer

    if max_score < 70:
        if "quel est" in preprocessed_input:
            matched_response = "Désolé, je n'ai pas l'information demandée."
        elif "combien" in preprocessed_input:
            matched_response = "Désolé, je ne suis pas capable de répondre à cette question pour le moment."
        else:
            matched_response = "Désolé, je n'ai pas d'information sur cette question."

    if isinstance(matched_response, list):
        matched_response = matched_response[0]


    return matched_response




