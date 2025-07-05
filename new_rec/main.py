from collections import defaultdict, Counter

import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, precision_recall_curve
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Reshape, Dense, Dropout
import collections
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from scipy.spatial.distance import cosine
from tensorflow.keras.regularizers import l2
from sklearn.model_selection import GridSearchCV




pd.set_option('display.max_columns', None) # Mostra tutte le colonne
pd.set_option('display.width', None) # occupa tutta la larghezza della console
# pd.set_option('display.max_colwidth', None) # non impone un limite sulla larghezza delle colonne

reviews = pd.read_json("/home/luca/Università/Tirocinio/Prove Tesi/Dataset/All_Beauty.jsonl", lines=True)
meta = pd.read_json("/home/luca/Università/Tirocinio/Prove Tesi/Dataset/meta_All_Beauty.jsonl", lines=True)

# snellisco i dataset e li unisco
reviews = reviews[['user_id', 'asin', 'text', 'rating']]
reviews.columns = ['user_id', 'product_id', 'review_text', 'rating']

meta = meta[['parent_asin', 'title', 'description', 'price', 'categories']]
meta.columns = ['product_id', 'title', 'description', 'price', 'categories']

df = pd.merge(reviews, meta, on='product_id', how='inner')
df['price_tier'] = pd.qcut(df['price'], q=4, labels=['low', 'medium', 'high', 'luxury'], duplicates='drop')

user_review_counts = df['user_id'].value_counts()

# -------------------------------
# Filtro gli utenti più attivi (n. recensioni >= 5)
# -------------------------------

active_users = user_review_counts[user_review_counts >= 5].index
df_filtered = df[df['user_id'].isin(active_users)]

# -------------------------------
# Filtro i prodotti che n. recensioni >= 3
# -------------------------------

product_review_counts = df_filtered['product_id'].value_counts()
active_products = product_review_counts[product_review_counts >= 3].index
df_filtered = df_filtered[df_filtered['product_id'].isin(active_products)]
df_filtered = df_filtered[(df_filtered['title'].str.len() > 10)]

def get_last_category_safely(category_list):
    """
    Estrae l'ultima categoria da una lista in modo sicuro.
    Se la lista non è valida o è vuota, restituisce 'Unknown'.
    """
    if isinstance(category_list, list) and category_list: # Controlla se è una lista E se non è vuota
        return category_list[-1]
    return 'Unknown' # Valore di default

# Applica la funzione sicura
df_filtered['specific_category'] = df_filtered['categories'].apply(get_last_category_safely)

# -------------------------------
# DIVIDO IL DATASET PRIMA DI COSTRUIRE I PROFILI (PER EVITARE DATA LEAKAGE)
# -------------------------------

# DIVIDO GLI UTENTI (CLIENT)
all_users = df_filtered['user_id'].unique()
# Divido prima in training+validation (90%) e test (10%)
train_val_users, test_users = train_test_split(all_users, test_size=0.15, random_state=42)
# Divido il primo gruppo in training (80% del totale) e validation (10% del totale)
train_users, val_users = train_test_split(train_val_users, test_size=(0.10/0.85), random_state=42)

# Creo i tre set di dati
train_df = df_filtered[df_filtered['user_id'].isin(train_users)].copy()
val_df = df_filtered[df_filtered['user_id'].isin(val_users)].copy()
test_df = df_filtered[df_filtered['user_id'].isin(test_users)].copy()

test_user_id_to_check = 'AEZP6Z2C5AVQDZAJECQYZWQRNG3Q'

if test_user_id_to_check in train_users:
    print(f"\n\033[91mATTENZIONE: L'utente di test '{test_user_id_to_check}' è nel TRAINING SET!\033[0m")
elif test_user_id_to_check in val_users:
    print(f"\n\033[93mINFO: L'utente di test '{test_user_id_to_check}' è nel VALIDATION SET.\033[0m")
elif test_user_id_to_check in test_users:
    print(f"\n\033[92mOK: L'utente di test '{test_user_id_to_check}' è nel TEST SET. (Corretto)\033[0m")
else:
    print(f"\n\033[91mATTENZIONE: L'utente di test '{test_user_id_to_check}' non è stato trovato in nessun set!\033[0m")

print("\n\033[94mAlcuni utenti validi per il test:")
print(test_users[:5], "\033[0m") # Stampa i primi 5 utenti del test set

# -------------------------------
# Costruzione dei profili utente - embedding, rappresentano il 'campo semantico' degli utenti
# -------------------------------

sbert_model = SentenceTransformer('paraphrase-MiniLM-L3-v2')

# --- CALCOLO EMBEDDING PER IL TRAINING SET ---
print("Calcolo embedding delle recensioni per il Training Set...")
train_texts = train_df['review_text'].tolist()
train_review_embs = sbert_model.encode(train_texts, batch_size=64, show_progress_bar=True)
# Aggiungi la colonna al DataFrame corretto
train_df['review_emb'] = train_review_embs.tolist()


# --- CALCOLO EMBEDDING PER IL VALIDATION SET ---
print("Calcolo embedding delle recensioni per il Validation Set...")
val_texts = val_df['review_text'].tolist()
val_review_embs = sbert_model.encode(val_texts, batch_size=64, show_progress_bar=True)
# Aggiungi la colonna al DataFrame corretto
val_df['review_emb'] = val_review_embs.tolist()


# --- CALCOLO EMBEDDING PER IL TEST SET ---
print("Calcolo embedding delle recensioni per il Test Set...")
test_texts = test_df['review_text'].tolist()
test_review_embs = sbert_model.encode(test_texts, batch_size=64, show_progress_bar=True)
# Aggiungi la colonna al DataFrame corretto
test_df['review_emb'] = test_review_embs.tolist()


print("Costruzione profili utente...")

# Costruisci i due tipi di profili
positive_profiles = {}
negative_profiles = {}

# Usiamo il train_df per costruire i profili
users_in_train = train_df['user_id'].unique()

for user in users_in_train:
    # Profilo Positivo (come prima)
    positive_embs = train_df[
        (train_df['user_id'] == user) & (train_df['rating'] >= 4)
    ]['review_emb'].tolist()
    if positive_embs:
        positive_profiles[user] = np.mean(positive_embs, axis=0)

    # Profilo Negativo (la novità)
    negative_embs = train_df[
        (train_df['user_id'] == user) & (train_df['rating'] <= 2) # Scegli una soglia adatta
    ]['review_emb'].tolist()
    if negative_embs:
        negative_profiles[user] = np.mean(negative_embs, axis=0)

# Gestione degli utenti che non hanno recensioni negative (importante!)
# Creiamo un profilo "negativo generico" come media di tutti gli embedding negativi
all_negative_embs = train_df[train_df['rating'] <= 2]['review_emb'].tolist()
if all_negative_embs:
    generic_negative_profile = np.mean(all_negative_embs, axis=0)
else:
    # Fallback nel caso non ci siano proprio recensioni negative nel training set
    generic_negative_profile = np.zeros_like(train_texts[0])

# Fallback anche per i positivi, sebbene meno probabile
all_positive_embs = train_df[train_df['rating'] >= 4]['review_emb'].tolist()
generic_positive_profile = np.mean(all_positive_embs, axis=0)

# -------------------------------
# Costruzione profili prodotto
# -------------------------------

print("Costruzione profili prodotto...")

train_df.loc[:,'description'] = train_df['description'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
train_df.loc[:,'product_text'] = train_df['title'].fillna('') + ' ' + train_df['description'].fillna('')# crea una nuova colonna nel df
                                                                                                            # title + description
product_texts = train_df.drop_duplicates('product_id')[['product_id', 'product_text']] # prende solo una volta gli ID dei prodotti con il loro
                                                                                 # product_text associato

df_filtered = df_filtered.merge(
    product_texts,  # contiene ['product_id', 'product_text']
    on='product_id',
    how='left'      # o 'inner' se sei sicuro che tutti i product_id combaciano
)

print("Calcolo embedding prodotti...")
product_texts['product_emb'] = sbert_model.encode(product_texts['product_text'].tolist(), batch_size=64, show_progress_bar=True).tolist() # allena SBERT

product_profiles = dict(zip(product_texts['product_id'], product_texts['product_emb'])) # crea dizionario ID: embedding


# -------------------------------
# Preparazione dati modello
# -------------------------------

def prepare_xy_v3(df, positive_profiles, negative_profiles, product_profiles,
                         generic_positive_profile, generic_negative_profile):
    x = []
    y = []

    def cosine_similarity(v1, v2):
        if v1 is None or v2 is None or np.all(v1 == 0) or np.all(v2 == 0):
            return 0
        dist = cosine(v1, v2)
        return 1 - np.clip(dist, 0.0, 2.0)

    for _, row in df.iterrows():
        user_id = row['user_id']
        product_id = row['product_id']

        # Recupera l'embedding del prodotto
        prod_emb = product_profiles.get(product_id)
        review_emb = np.array(row['review_emb'])

        if prod_emb is not None:
            # Per il test set, questi profili saranno sempre quelli generici.
            # Per il training set, saranno quelli specifici dell'utente.
            # La logica funziona in entrambi i casi.
            pos_profile = positive_profiles.get(user_id, generic_positive_profile)
            neg_profile = negative_profiles.get(user_id, generic_negative_profile)

            sim_to_positive = cosine_similarity(prod_emb, pos_profile)
            sim_to_negative = cosine_similarity(prod_emb, neg_profile)
            sim_difference = sim_to_positive - sim_to_negative
            sim_review_to_positive = cosine_similarity(review_emb, pos_profile)

            input_vect = np.concatenate([
                prod_emb,  
                review_emb, 
                [sim_to_positive],  
                [sim_to_negative],
                [sim_difference],
                [sim_review_to_positive]  
            ])

            x.append(input_vect)
            y.append(1 if row['rating'] >= 4 else 0)

    return np.array(x), np.array(y)

x_train, y_train = prepare_xy_v3(
    train_df,
    positive_profiles, negative_profiles, product_profiles,
    generic_positive_profile, generic_negative_profile
)

x_val, y_val = prepare_xy_v3(
    val_df,
    positive_profiles, negative_profiles, product_profiles,
    generic_positive_profile, generic_negative_profile
)
x_test, y_test = prepare_xy_v3(
    test_df,
    positive_profiles, negative_profiles, product_profiles,
    generic_positive_profile, generic_negative_profile
)


y_train = y_train.astype(np.float32)
y_val = y_val.astype(np.float32)
y_test = y_test.astype(np.float32)


print("Train distrib:", np.bincount(y_train.astype(int)))
print("Val distrib:", np.bincount(y_val.astype(int)))

best_params = {'colsample_bytree': 0.7, 'learning_rate': 0.1, 'max_depth': 5, 'n_estimators': 300, 'subsample': 0.8}

scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

# Inizializza il classificatore
xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    **best_params
)

xgb_clf.fit(x_train, y_train)

y_pred_best = xgb_clf.predict(x_test)

print("\n--- Risultati XGBoost Ottimizzato ---")
print(classification_report(y_test, y_pred_best))
