# Migliorie dell'arlgoritmo
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
Questi risultati ci dicono diverse cose: il modello riconosce molto bene la classe 1, ma ottiene una precisione del 45% sulla classe 0, con solo 37% di recall. Quindi il modello riesce a riconoscere solo il 37% delle classi 0, e quando lo fa ha ragione il 45% delle volte.  
L'accuracy dell'80% è dovuta probabilmente alla quasi sicurezza con cui il modello predice ciò che l'utente ha gradito.

---

## Passi per migliorare
L'idea è quella di distaccarsi dall'approccio delle reti neurali e passare ad un altro modello.
Per migliorare questo progetto è stato utilizzato XGBoost (eXtreme Gradient Boosting), una libreria di apprendimento automatico distribuita e open source che utilizza alberi decisionali potenziati dal gradiente, un algoritmo di potenziamento dell'apprendimento supervisionato che utilizza la discesa del gradiente.  

Gli alberi decisionali possono essere visti come modelli semplici che utilizzano poche regole. In XGBoost il primo albero fa una predizione molto semplicistica, spesso sbagliata.  
Successivamente interviene il secondo albero, che calcola l'errore (residuo) del primo basandosi sui dati reali.  
Tutto questo viene poi unito insieme (Ensemble) e iterato migliaia di volte. Quest'ultimo passaggio è chiamato `Boosting`, nel quale ogni `ensemble` impara dal precedente.

Per la correzione dell'errore si utilizza il metodo della discesa del gradiente, come suggeriva il nome.

---

## Strategia adottata
Una volta capito quale modello utilizzare, ho migliorato l'algoritmo precedente.  
In particolare ho creato 2 profili per un utente:
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

Ho anche introdotto le categorie e il prezzo preferito dagli utenti, per arricchire le informazioni.

```python
# Prendo il prezzo preferito degli utenti
user_favorite_price_tier = {}
for user in train_df['user_id'].unique():
    positive_reviews = train_df[(train_df['user_id'] == user) & (train_df['rating'] >= 4)]
    if not positive_reviews.empty:
        tier_counts = Counter(positive_reviews['price_tier'])
        if tier_counts:
            user_favorite_price_tier[user] = tier_counts.most_common(1)[0][0]



# prendo la categoria preferita dell'utente
user_favorite_category = {}
for user in train_df['user_id'].unique():
    positive_reviews = train_df[(train_df['user_id'] == user) & (train_df['rating'] >= 4)]
    if not positive_reviews.empty:
        cat_counts = Counter(positive_reviews['specific_category'])
        if cat_counts:
            user_favorite_category[user] = cat_counts.most_common(1)[0][0]
```

Tramite la `cosine similarity` ho calcolato quanto un prodotto fosse simile a ciò che piace/non piace ad un utente

```python
sim_to_positive = cosine_similarity(prod_emb, pos_profile)
sim_to_negative = cosine_similarity(prod_emb, neg_profile)
sim_difference = sim_to_positive - sim_to_negative
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
    [price_match],  # Feature del prezzo
    [category_match]  # Feature della categoria
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

> [!NOTE] Questa parte di codice non è presente nello script `main.py`

---

## Risultati del modello
Il modello così allenato ha dato i seguenti risultati

```python
--- Risultati XGBoost ---
              precision    recall  f1-score   support

         0.0       0.39      0.56      0.46       344
         1.0       0.89      0.80      0.84      1507

    accuracy                           0.76      1851
   macro avg       0.64      0.68      0.65      1851
weighted avg       0.80      0.76      0.77      1851
```

I risultati sono simili a prima, ma adesso si ha un netto miglioramento sulla recall e l'f1-score.

Applicando il modello esternamente ad un utente, si ottiene

```bash
Confronto rating reali vs predetti per l'utente AEZP6Z2C5AVQDZAJECQYZWQRNG3Q:

product_id  true_rating  predicted_label
B0851C4YPC            2                1
B0855L611L            2                1
B08JH8NGKN            2                0
B0046BPTI2            2                0
B07YNDWRCB            2                0
B0844X21MJ            3                0
B08GBZGRDQ            3                0
B07FQTCLNX            3                0
B07YQ5ZNPL            3                0
B07YQ6G3LL            3                0
B08GMC5ZRY            3                0
B07YDLYYP3            3                0
B08DX6B4QN            3                0
B086XH7BWW            3                1
B08DXLRTSB            3                0
B08K2HC58L            3                0
B08DXZ5VXB            3                0
B07TNSGHVX            3                1
B00O2FGBJS            3                0
B083TLNBJJ            3                1
B07NPWK167            3                0
B07DFNPVSF            3                0
B07SJ98G6Z            3                0
B07FPS2VFK            4                1
B08JPK6MKD            4                1
B08LPC1G23            4                1
B08LYT4Q2X            4                1
B08DX9P6V1            4                1
B08P7QWN1N            4                1
B087ZQK2G8            4                1
B07FP2C8N8            4                1
B07HHZBH4X            4                1
B07W397QG4            4                1
B08575Y9V3            4                1
B082VKPJV5            4                1
B08C71WBLC            4                1
B084WP4XS8            4                1
B08C37KWRR            4                1
B07W1WJZFG            4                1
B08D7TLV21            4                1
B087ZQG11L            4                1
B07V2L94ZW            4                1
B073ZR1XLQ            4                1
B08DD6BFFM            4                1
B083CXR5V8            4                1
B07PCLLWHJ            4                1
B0844X4D53            4                1
B07RM722DH            4                1
B086SSMK7P            4                1
B015A5DGG4            4                1
B07PRDZ2BH            4                1
B08CJHC9ZV            4                1
B08HRNPNR5            4                1
B084D86YL8            4                1
B08FR3QXYY            4                1
B08SJKR877            4                1
B07ZQRX7FX            5                1
B083G2PVX3            5                1
B07SW7D6ZR            5                1
B08HMLXW65            5                0
B083BCSQGN            5                1
B08KWN77LW            5                1
B08RQZ3F3L            5                1
B07J1LYVHC            5                1
B08MQWJZSG            5                1
B071LLTN9H            5                1
B07Z3NRMBS            5                1
B07M9D3WYW            5                1
B07WNBZQGT            5                1
B07JC3GQQM            5                1
B08CL46XNM            5                1
B07ZS3DKL5            5                1
B08465489V            5                1
B08HMJT41C            5                1
B07VDCD17L            5                1
B08L4HTQ3R            5                1
B08LPJT4MT            5                1
B07NPCT6L5            5                1
B07MZT83KK            5                1
B07MN1KJ15            5                1
B07D6KYSJH            5                1
B0813ZQG3T            5                1
B08MPK4JRB            5                1
B083LHHQYF            5                1
B07SD7GP25            5                1
B08GYJY8F2            5                1
B07PCSRSND            5                1
B07VGBBNTH            5                1
B077MYW993            5                1
B07V1QTSQ1            5                1
B07VSZYKCH            5                1
B08BS3WDPJ            5                1
B0832Z7KFJ            5                1
B07VQR3W3Z            5                1
B08LFYMGS5            5                1
B07X8W7GJZ            5                1
B07XNMVJ53            5                1
B07L9H27SH            5                1
B07PBWVV5K            5                1
B07NZ4F82C            5                1
B07YYW1913            5                1
B086M8MZGB            5                1
B0849YFB92            5                1
B07THLR7RR            5                1
```

Si nota come su 81 recensioni positive, il modello abbia sbagliato 1 volta, mentre su 23 recensioni negative ha sbagliato 5 volte.  
Questo potrebbe essere dovuto diversi fattori:
- è possibile che il testo delle queste recensioni, pur dando un rating basso, parli del prodotto in termini neutri o addirittura positivi ("Il prodotto ha un buon profumo, ma..."), confondendo SBERT.
- l'utente potrebbe aver dato un rating basso per motivi che non sono nel testo (es. "pacco arrivato rotto", "prezzo troppo alto per la quantità"). 
  In questi casi, un modello content-based non potrà mai indovinare la valutazione.

Si tratta di fattori che il modello non può prevedere, e che non si possono filtrare rimaneggiando il dataset.

## Conclusioni
Il modello è migliorato molto, riuscendo a prevedere con successo i prodotti che possono piacere ad un utente, sbagliando alcune volte, cosa che, per un sistema di raccomandazione, può essere accettabile.
