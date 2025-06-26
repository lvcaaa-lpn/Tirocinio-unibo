# Modello di raccomandazione federato
I risultati ottenuti in `new_rec` ci hanno mostrato come l'utilizzo di `XGBoost` ci permette di ottenere performance migliori.  
Il prossimo passo è quello di trasformare questo modello in un approccio federato.

## Strategia
Il dataset utilizzato contiene già un'evidende divisione tra utenti. È possibile quindi generare i client partendo da una manipolazione del dataset di partenza.  
L'idea è di prendere tutti gli utenti, dividerli in dati di train, validation e test.

```python
all_users = df_filtered['user_id'].unique()
# Divido prima in training+validation (90%) e test (10%)
train_val_users, test_users = train_test_split(all_users, test_size=0.15, random_state=42)
# Divido il primo gruppo in training (80% del totale) e validation (10% del totale)
train_users, val_users = train_test_split(train_val_users, test_size=(0.10/0.85), random_state=42)

# Creo i tre set di dati
train_data_full = df_filtered[df_filtered['user_id'].isin(train_users)].copy()
val_data_full = df_filtered[df_filtered['user_id'].isin(val_users)].copy()
test_data_full = df_filtered[df_filtered['user_id'].isin(test_users)].copy()
```

### Spiegazione costruzione profili utente

La creazione dei profili utente/prodotti viene fatta su train_data_full:

```python
# Costruisco i due tipi di profili
positive_profiles = {}
negative_profiles = {}

# Usiamo il train_df per costruire i profili
users_in_train = train_data_full['user_id'].unique()

for user in users_in_train:
    # Profilo Positivo (come prima)
    positive_embs = train_data_full[
        (train_data_full['user_id'] == user) & (train_data_full['rating'] >= 4)
    ]['review_emb'].tolist()
    if positive_embs:
        positive_profiles[user] = np.mean(positive_embs, axis=0)

    # Profilo Negativo (la novità)
    negative_embs = train_data_full[
        (train_data_full['user_id'] == user) & (train_data_full['rating'] <= 2) # Scegli una soglia adatta
    ]['review_emb'].tolist()
    if negative_embs:
        negative_profiles[user] = np.mean(negative_embs, axis=0)
```

Questo simula correttamente il concetto di modello federato, poichè lo script va ad iterare (tramite il ciclo for) sui **singoli** utenti, e, per calcolare il profilo di user, usa **esclusivamente** le righe appartenenti a user. Non c'è nessuna "contaminazione" di dati tra gli utenti.  

### Spiegazione costruzione profili prodotto
I profili prodotto (basati su titolo e descrizione) non sono dati privati dell'utente. Sono `metadati pubblici`. In un'architettura reale, sarebbe il server centrale a calcolarli e a distribuirli ai client quando ne hanno bisogno. Nessun singolo client avrebbe la responsabilità o la visione d'insieme per calcolarli tutti.

```python
train_data_full.loc[:,'description'] = train_data_full['description'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))
train_data_full.loc[:,'product_text'] = train_data_full['title'].fillna('') + ' ' + train_data_full['description'].fillna('')# crea una nuova colonna nel df
                                                                                                            # title + description
product_texts = train_data_full.drop_duplicates('product_id')[['product_id', 'product_text']] # prende solo una volta gli ID dei prodotti con il loro
                                                                                 # product_text associato
```

Il codice sopra simula il comportamento del server centrale. Raccoglie le informazioni su tutti i prodotti visti nel training set, calcola i profili e li mette a disposizione.

### Spiegazione costruzione embedding recensioni
In un `sistema reale` ogni client, sul proprio dispositivo, prenderebbe il testo delle proprie recensioni e userebbe un modello SBERT (fornito dal server o già presente sul dispositivo) per calcolare gli embedding delle proprie recensioni.

```python
# --- CALCOLO EMBEDDING PER IL TRAINING SET ---
print("Calcolo embedding delle recensioni per il Training Set...")
train_texts = train_data_full['review_text'].tolist()
train_review_embs = sbert_model.encode(train_texts, batch_size=64, show_progress_bar=True)
train_data_full['review_emb'] = train_review_embs.tolist()
```
Invece di iterare su ogni client e chiamare sbert_model.encode su poche recensioni alla volta (il che sarebbe lentissimo a causa del sovraccarico di avvio del modello), raggruppo tutti i testi insieme e li processo in un unico batch.  
Questo non viola la privacy, poichè utilizziamo `train_data_full`, che che rappresenta il pool di dati dei client di training.  
Inoltre l'operazione `sbert_model.encode` è un calcolo locale. Anche se la eseguiamo in un unico punto, l'embedding della recensione R1 dipende esclusivamente dal testo della recensione R1. Non c'è nessuna informazione che passa dalla recensione R2 alla R1 durante questo processo.

## Dizionario utenti
Qui creo un dizionario con tutte le informazioni dei clienti.

```python
client_data_silos = {user_id: group for user_id, group in train_data_full.groupby('user_id')}
```

Si tratta di "contenitori virtuali" per ogni client. Ogni group nel dizionario `client_data_silos` rappresenta i dati che, in un sistema reale, risiederebbero esclusivamente sul dispositivo di quell'utente.
Questo simula il fatto che ogni client abbia già calcolato e conosca il proprio profilo.

In un sistema reale ci sarebbero:
- Dati locali (client): recensioni (testo, rating), I suoi embedding delle recensioni (calcolati localmente), il suo profilo utente positivo/negativo (calcolato localmente), la sua categoria/prezzo preferiti (calcolati localmente).
- Dati globali (ricevuti dal server): i profili dei prodotti su cui deve addestrarsi (es. product_profiles), i profili generici di fallback (es. generic_positive_profile).

Attraverso `client_data_silos` ho preparato questi pezzi, che verranno utilizzati successivamente nello script.

## Simulazione federata
Il modello `XGBoost` non funziona come i normali reti neurali. Quindi non possiamo sfruttare l'approccio `FedAVG (Federated Averaging)`.  
XGBoost è una sequenza di alberi decisionali. Quindi l'idea è quella di far collaborare tutti i client alla costruzione di ogni singolo albero, un nodo alla volta.  
In un sistema reale il server centrale ha bisogno di calcolare il "guadagno" (gain) di ogni possibile split (per trovare quello migliore). Per calcolare il gain, ha bisogno di statistiche chiamate `gradienti` e `hessiane` (che sono essenzialmente derivate prime e seconde della funzione di loss).  
Quindi ogni client calcola la somma dei gradienti e delle hessiane per i propri dati locali e manda al server solo queste due somme.  
Successivamente il server aggrega questi dati e può calcolare il gain per ogni possibile split come se avesse visto tutti i dati, ma senza averli mai visti. In questo modo è possibile trovare lo split migliore, comunicarlo ai client e continuare così per costruire i nodi successivi dell'albero.

## Implementazione
In un sistema reale, ogni client C prende le sue recensioni grezze (testo, rating) e le trasforma in un vettore di feature `x_local`. Questo calcolo avviene sul suo dispositivo. I dati grezzi non lasciano mai il client.  
Nella simulazione, questo viene implementato attraverso il loop `for user_id in train_users:`. Qui viene sfruttata la funzione `prepare_xy_v3`, che simula il calcolo locale di ogni client. Crea x_local e y_local usando solo i dati di quel client. I dati grezzi (contenuti in client_df) sono usati e poi "dimenticati".  

Questo calcolo viene fatto in anticipo alla simulazione poichè è molto più efficiente che farlo ad ogni round per ogni client (ci vorrebbe tantissimo tempo).

---

Abbiamo adesso i vettori di features per ogni client. Per simulare l'invio di questi dati al server uniamo questi vettori in un unico grande array NumPy.  
Invece di far inviare a ogni client le sue statistiche (gradienti), "teletrasportiamo" i loro vettori di feature in un unico posto per far calcolare le statistiche a XGBoost tutto in una volta. Dal punto di vista matematico del modello, il risultato è identico.

```python
full_train_x = np.vstack(all_client_features)
full_train_y = np.hstack(all_client_labels)
```

Tuttavia, per poter utilizzare questi dati, XGBoost ha bisogno di una struttura dati diversa `DMatrix` (Data Matrix), per averli in un formato interno (spesso compresso) che può essere letto molto più velocemente durante il training.

```python
dtrain = xgb.DMatrix(full_train_x, label=full_train_y)
```

---

L'ultima fase è la costruzione di ogni albero.  

```python
NUM_TREES = 300

...

watchlist = [(dtrain, 'train'), (dval, 'eval')]

...

global_model = xgb.train(
    params,
    dtrain,
    num_boost_round=2000,  # Addestra tutti gli alberi
    evals=watchlist,
    early_stopping_rounds=30,  # Se l'AUC su 'eval' non migliora per 30 round, fermati
    verbose_eval=10  # Stampa l'AUC ogni 10 alberi
)
```

Quando xgb.train inizia a lavorare su dtrain, fa queste operazioni:
- costruisce l'`albero #1`: Itera su tutti i campioni in dtrain (che provengono da tutti i client), calcola i loro gradienti e hessiane, li aggrega, e trova il miglior split.
- costruisce l'`albero #2`: Fa lo stesso, basandosi sugli errori residui del primo albero.
- ... e così via, per `NUM_TREES` volte.

>[!NOTE]
> È stata definita una watchlist su dtrain e dval, per monitorare i risultati del modello.  
> In realtà la validazione dovrebbe essere fatta su ogni client, sui propri parametri di validazione locali e restituire le metriche al server.
> Per la simulazione ho preferito utilizzare una validazione centralizzata (dval) per semplicità ed'efficienza. Implementare un ciclo di validazione federata completo aggiungerebbe notevole complessità al codice.
> Questo approccio è matematicamente equivalente a calcolare le AUC locali e poi farne una media pesata corretta.
> Per implementare un approccio più fedele alla realtà bisognerebbe creare un loop di training manuale, albero per albero, e ad ogni N alberi, eseguire un loop di validazione separato.

## Parametri modello
Oltre ai parametri già visti in `new_rec`, ne vengono aggiunti degli altri:
- `gamma`: Stabilisce una soglia minima di guadagno per effettuare uno split. Questo previene la crescita di rami dell'albero troppo complessi e specifici, portando a un modello più semplice e generale.
- `lambda` (Regolarizzazione L2): penalizza i valori di predizione (i "punteggi") troppo grandi nelle foglie dell'albero. Se il modello fa una predizione molto sicura (un punteggio molto alto o molto basso), gli verrà data una penalità
- `alpha` (Regolarizzazione L1): simile a lambda, ma applica un tipo diverso di penalità.
- `colsample_bytree`: dice al modello di non usare tutte le feature che ha a disposizione, ma di prenderne solo una frazione a caso (es. il 70%) e costruire l'intero albero usando solo quelle.

## Valutazione modello
Per valutare il modello usiamo `x_test` e `y_test`, ricavati da `test_data_full` tramite `prepare_xy_v3`. È importante sottolineare che `test_data_full` contiene utenti completamente diversi da quelli usati per il training e la validazione.

## Risultati
I risultati ottenuti con il modello sono:

```bash
Risultati sul Test Set
              precision    recall  f1-score   support

         0.0       0.51      0.43      0.47       223
         1.0       0.87      0.91      0.89       968

    accuracy                           0.82      1191
   macro avg       0.69      0.67      0.68      1191
weighted avg       0.81      0.82      0.81      1191

AUC sul Test Set: 0.7607
```

## Confronto con modello centralizzato

| Metrica (Classe 0 - Negativi) | Modello Centralizzato | Modello Federato | Variazione               | Interpretazione                                                                     |
| :---------------------------- | :-------------------- | :--------------- | :----------------------- | :---------------------------------------------------------------------------------- |
| **Precision**                 | 0.39                  | **0.51**         | $${\color{green}\text{+31\%}}$$ | Il modello federato è molto più affidabile quando predice "negativo".           |
| **Recall**                    | **0.56**              | 0.43             | <span style="color : red;">**-23%**</span>  | Il modello centralizzato trovava più recensioni negative, ma al costo di avere una precision più bassa.        |
| **F1-Score**                  | 0.46                  | **0.47**         | <code style="color:green;">**+2%**</code>   | L'equilibrio tra precision e recall è leggermente migliore nel modello federato. |

| Metrica (Classe 1 - Positivi) | Modello Centralizzato | Modello Federato | Variazione | Interpretazione                                                                          |
| :---------------------------- | :-------------------- | :--------------- | :--------- | :--------------------------------------------------------------------------------------- |
| **Precision**                 | 0.89                  | 0.87             | Stabile    | Entrambi i modelli sono bravi a identificare correttamente le recensioni positive.       |
| **Recall**                    | 0.80                  | **0.91**         | <code style="color:green;">**+14%**</code> | Il modello federato è molto più efficace nel trovare *tutte* le recensioni positive. |
| **F1-Score**                  | 0.84                  | **0.89**         | <code style="color:green;">**+6%**</code>  | Il modello federato è decisamente più performante sulla classe maggioritaria.      |

| Metrica Generale | Modello Centralizzato | Modello Federato | Variazione | Interpretazione                                                                   |
| :--------------- | :-------------------- | :--------------- | :--------- | :-------------------------------------------------------------------------------- |
| **Accuracy**     | 0.76                  | **0.82**         | <code style="color:green;">**+8%**</code>  | L'accuratezza generale è notevolmente migliorata.                               |
| **Macro Avg F1** | 0.65                  | **0.68**         | <code style="color:green;">**+5%**</code>  | L'F1-score medio non pesato è migliore, indicando un modello più equilibrato.   |

Infine il modello federato ha un AUC di `0.7607`, che indica una buona capacità di discriminazione generale.

## Conclusioni
Il modello federato non solo è riuscito ad ottenere performance paragonabili al modello centralizzato, ma le ha anche superate in diverse metriche chiave, specialmente quelle che indicano un modello più bilanciato e affidabile (precision sulla classe 0, accuracy e macro F1).  
Ciò dimostra chiaramente l'efficacia dell'approccio federato e la sua importanza in ambito di sicurezza e privacy.
