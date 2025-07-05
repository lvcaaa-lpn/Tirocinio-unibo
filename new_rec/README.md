# Migliorie dell'algoritmo
I risultati ottenuti in `recommender-sys` non erano ottimali.  
Nonostante l'utilizzo di diverse tecniche (focal_loss, Smote, arricchimento profilo utenti/prodotti, cosine similarity, regression) il modello sembrava bloccato e non performava bene.

In particolare l'ultimo risultato è stato:

```bash
Loss: 0.0942, Accuracy: 0.8188, AUC: 0.7010 

              precision    recall  f1-score   support

         0.0       0.45      0.37      0.41       334
         1.0       0.86      0.90      0.88      1438

    accuracy                           0.80      1772
   macro avg       0.66      0.63      0.64      1772
weighted avg       0.78      0.80      0.79      1772
```
Questi risultati ci dicono diverse cose: il modello riconosce molto bene la classe 1, ma ottiene una precisione del 45% sulla classe 0, con solo 37% di recall. Quindi il modello riesce a riconoscere solo il 37% delle classi 0, e quando lo fa, ha ragione il 45% delle volte.  
L'accuracy dell'80% è dovuta probabilmente alla quasi sicurezza con cui il modello predice ciò che l'utente ha gradito.

---

## Passi per migliorare
L'idea è quella di distaccarsi dall'approccio delle reti neurali e passare ad un altro modello.
Per migliorare questo progetto è stato utilizzato `XGBoost` (eXtreme Gradient Boosting), una libreria di apprendimento automatico distribuita e open source che utilizza `alberi decisionali` potenziati dal gradiente, un algoritmo di potenziamento dell'apprendimento supervisionato che utilizza la `discesa del gradiente`.  

Gli alberi decisionali possono essere visti come modelli semplici che utilizzano poche regole. In XGBoost il primo albero fa una predizione molto semplicistica, spesso sbagliata.  
Successivamente interviene il secondo albero, che calcola l'errore (residuo) del primo basandosi sui dati reali.  
Tutto questo viene poi unito insieme (Ensemble) e iterato migliaia di volte. Quest'ultimo passaggio è chiamato `Boosting`, nel quale ogni `ensemble` impara dal precedente.

Per la correzione dell'errore si utilizza il metodo della discesa del gradiente, come suggeriva il nome.

---

## Strategia adottata
Una volta capito quale modello utilizzare, ho migliorato l'algoritmo precedente. 

Ho diviso il dataset in train, val e test in base agli utenti, per una separazione più fedele alla realtà e per evitare Data Leakage.  
Questo assicura che gli utenti presenti nel set di validazione e di test siano completamente nuovi e mai visti durante la fase di training.  
Tale approccio simula lo scenario 'cold start', che è la sfida più impegnativa e comune per un sistema di raccomandazione, ovvero predire le preferenze per un nuovo utente. 

```python
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
```

L'altro cambiamento significativo è stato creare 2 profili per un utente:
- uno basato sulle sue recensioni positive (rappresenta cosa piace)
- uno basato sulle sue recensioni negative (rappresenta cosa non piace)

```python
positive_profiles = {}
negative_profiles = {}

users_in_train = train_df['user_id'].unique()

for user in users_in_train:
    positive_embs = train_df[
        (train_df['user_id'] == user) & (train_df['rating'] >= 4)
    ]['review_emb'].tolist()
    if positive_embs:
        positive_profiles[user] = np.mean(positive_embs, axis=0)

    negative_embs = train_df[
        (train_df['user_id'] == user) & (train_df['rating'] <= 2)
    ]['review_emb'].tolist()
    if negative_embs:
        negative_profiles[user] = np.mean(negative_embs, axis=0)

# Gestione degli utenti che non hanno recensioni negative
# Creo un profilo "negativo generico" come media di tutti gli embedding negativi
all_negative_embs = train_df[train_df['rating'] <= 2]['review_emb'].tolist()
if all_negative_embs:
    generic_negative_profile = np.mean(all_negative_embs, axis=0)
else:
    # Fallback nel caso non ci siano proprio recensioni negative nel training set
    generic_negative_profile = np.zeros_like(train_texts[0])

# Fallback anche per i positivi, sebbene meno probabile
all_positive_embs = train_df[train_df['rating'] >= 4]['review_emb'].tolist()
generic_positive_profile = np.mean(all_positive_embs, axis=0)
```

Tramite la `cosine similarity` ho calcolato quanto un prodotto fosse simile a ciò che piace/non piace ad un utente

```python
sim_to_positive = cosine_similarity(prod_emb, pos_profile)
sim_to_negative = cosine_similarity(prod_emb, neg_profile)
sim_difference = sim_to_positive - sim_to_negative
sim_review_to_positive = cosine_similarity(review_emb, pos_profile)
```

Infine ho unito tutto in un vettore input

```python
input_vect = np.concatenate([
    user_pos_emb,  # Embedding di ciò che gli piace
    prod_emb,  # Embedding del prodotto target
    review_emb, # embedding recensioni
    [sim_to_positive],  # Feature semantica 
    [sim_to_negative],  # Feature semantica 
    [sim_difference],  # Feature semantica
    [sim_review_to_positive],
])
```

---

## Trovare i migliori parametri per XGBoost
XGBoost ha una serie di parametri:
- `n_estimators`: il numero massimo di alberi che il modello può costruire. Il modello creerà un albero dopo l'altro, fino a un massimo di N
- `max_depth`: la profondità massima di ogni singolo albero. Un albero con max_depth=5 può fare al massimo 5 "domande" in sequenza per arrivare a una conclusione.
- `subsample`: Prima di costruire ogni nuovo albero, XGBoost prende un campione casuale dell'80% (se subsample=0.8) delle righe (le recensioni) del training set. L'albero viene costruito solo su questo campione.
  Questo costringe l'albero a imparare pattern più generali, perchè non può vedere tutti i casi passati.
- `learning_rate`: determina quanto peso viene dato al contributo di ogni nuovo albero.
- `scale_pos_weight`: dice al modello di dare un peso maggiore agli errori commessi sulla classe minoritaria.

Utilizzando `gridSearchCV`

```python
# griglia di parametri da testare
param_grid = {
    'max_depth': [3, 4, 5],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'subsample': [0.7, 0.8],
    'colsample_bytree': [0.7, 0.8]
}

# peso delle classi
scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1)

# inizializzazione del classificatore
xgb_clf = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    scale_pos_weight=scale_pos_weight,
    random_state=42
)

# inizializzazione di GridSearchCV
grid_search = GridSearchCV(
    estimator=xgb_clf,
    param_grid=param_grid,
    scoring='f1_weighted', # Oppure 'roc_auc'
    cv=3, # 3-fold cross-validation
    verbose=2,
    n_jobs=-1 # Usa tutti i core della CPU
)

# ricerca sul set di training
grid_search.fit(x_train, y_train)

print("Migliori parametri trovati:", grid_search.best_params_)
```

> [!NOTE]
> Questa parte di codice non è presente nello script `main.py`

---

## Risultati del modello
Il modello così allenato ha dato i seguenti risultati

```python
--- Risultati XGBoost Ottimizzato ---
              precision    recall  f1-score   support

         0.0       0.37      0.58      0.45       193
         1.0       0.90      0.79      0.84       899

    accuracy                           0.75      1092
   macro avg       0.63      0.68      0.65      1092
weighted avg       0.80      0.75      0.77      1092

```

I risultati sono simili a prima, ma adesso si ha un netto miglioramento sulla recall e l'f1-score.

Applicando il modello esternamente ad un utente, si ottiene

```bash
Confronto rating reali vs predetti.
utente: AG5A4BNLSYHH2IEFJD3UM3N2IPMA:

product_id  true_rating  predicted_label
B07YQG1Y2Z            2                0
B095JYJJBH            3                1
B07D5FBFQ4            3                0
B07VHZDHR6            3                0
B07MN1KJ15            3                0
B08P27T7RZ            4                1
B086GST51S            4                1
B07YL4485K            4                1
B09QT8SLJB            5                1
B095RYJJHY            5                1
B09G9VRGS1            5                1
B08FTC49Q1            5                1
B08J7W1VQL            5                0
B08GY96F87            5                1
B08BXVJMRY            5                0
B089CSR3KF            5                0
```

Si nota come su 11 recensioni positive, il modello abbia sbagliato 3 volte, mentre su 5 recensioni negative ha sbagliato 1 volta. 

Provando su altri utenti, il modello sembra essere piuttosto bravo a identificare correttamente le recensioni negative (rating 1-3) e predice correttamente le recensioni molto positive (rating 5) per la maggior parte delle volte.  
Ovviamente ci sono i casi in cui sbaglia, come ad esempio quando l'utente ha votato 5, ma il modello ha predetto 0.

Questo potrebbe essere dovuto diversi fattori:
- è possibile che il testo delle queste recensioni, pur dando un rating basso, parli del prodotto in termini neutri o addirittura positivi ("Il prodotto ha un buon profumo, ma..."), confondendo SBERT.
- l'utente potrebbe aver dato un rating basso per motivi che non sono nel testo (es. "pacco arrivato rotto", "prezzo troppo alto per la quantità"). 
  In questi casi, un modello content-based non potrà mai indovinare la valutazione.

Si tratta di fattori che il modello non può prevedere, e che non si possono filtrare rimaneggiando il dataset.

## Conclusioni
Il modello è migliorato molto, riuscendo a prevedere con successo i prodotti che possono piacere ad un utente, sbagliando alcune volte, cosa che, per un sistema di raccomandazione, può essere accettabile.
