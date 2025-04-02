# federated-mnist

## Introduzione
Il file test.py si occupa di simulare un approccio di learning federato, utilizzando il dataset MNIST, contenente 60000 immagini 28x28 di cifre scritte a mano, insieme a 10000 immagini di test.

## Come funziona
Una volta caricato il dataset e mescolato, l'ho distribuito tra N 'client' (prima 2, poi 5). Il dataset viene distribuito con un batch size di 32.
Successivamente viene creato un modello globale che simula quello del server.  
Questo viene passato ai vari client, che allenano il modello 'localmente' con il proprio sottoinsieme di dati.  
Infine i client restituiscono solo i pesi al server, che procederà con l'aggiornarnamento del modello globale.

Questo processo viene ripetuto per diversi round, migliorando progressivamente l'accuratezza del modello.

## Risultati
1. Con una prima simulazione su 2 client e 5 round, ottengo (riporto solo l'output del primo e dell'ultimo round) :
```plaintext
Round 1 
Epoch 1/2
938/938 [==============================] - 2s 2ms/step - loss: 0.3389 - accuracy: 0.9055
Epoch 2/2
938/938 [==============================] - 1s 2ms/step - loss: 0.1595 - accuracy: 0.9544
Epoch 1/2
938/938 [==============================] - 1s 1ms/step - loss: 0.3341 - accuracy: 0.9043
Epoch 2/2
938/938 [==============================] - 1s 2ms/step - loss: 0.1562 - accuracy: 0.9548
313/313 [==============================] - 1s 2ms/step - loss: 1.1700 - accuracy: 0.9596
Server Model - Loss: 1.1700, Accuracy: 0.9596 

Round 5 
Epoch 1/2
938/938 [==============================] - 2s 2ms/step - loss: 0.0430 - accuracy: 0.9880
Epoch 2/2
938/938 [==============================] - 2s 2ms/step - loss: 0.0282 - accuracy: 0.9929
Epoch 1/2
938/938 [==============================] - 2s 1ms/step - loss: 0.0360 - accuracy: 0.9890
Epoch 2/2
938/938 [==============================] - 1s 1ms/step - loss: 0.0236 - accuracy: 0.9937
313/313 [==============================] - 0s 1ms/step - loss: 0.5029 - accuracy: 0.9796
Server Model - Loss: 0.5029, Accuracy: 0.9796
```

  Si nota, quindi, come il modello riesca ad imparare, passando da un'accuracy del 95%, ad una del 97%.

2. Testando il modello con 5 client e 10 round:
  ```plaintext
Round 1 
Epoch 1/2
375/375 [==============================] - 1s 2ms/step - loss: 0.4945 - accuracy: 0.8615
Epoch 2/2
375/375 [==============================] - 1s 2ms/step - loss: 0.2333 - accuracy: 0.9335
Epoch 1/2
375/375 [==============================] - 1s 2ms/step - loss: 0.4999 - accuracy: 0.8620
Epoch 2/2
375/375 [==============================] - 1s 2ms/step - loss: 0.2403 - accuracy: 0.9320
Epoch 1/2
375/375 [==============================] - 1s 1ms/step - loss: 0.4959 - accuracy: 0.8624
Epoch 2/2
375/375 [==============================] - 1s 2ms/step - loss: 0.2392 - accuracy: 0.9323
Epoch 1/2
375/375 [==============================] - 1s 1ms/step - loss: 0.5050 - accuracy: 0.8597
Epoch 2/2
375/375 [==============================] - 1s 2ms/step - loss: 0.2401 - accuracy: 0.9320
Epoch 1/2
375/375 [==============================] - 1s 2ms/step - loss: 0.4913 - accuracy: 0.8621
Epoch 2/2
375/375 [==============================] - 1s 2ms/step - loss: 0.2340 - accuracy: 0.9333
313/313 [==============================] - 0s 678us/step - loss: 1.4823 - accuracy: 0.9394
Server Model - Loss: 1.4823, Accuracy: 0.9394 

Round 10 
Epoch 1/2
375/375 [==============================] - 1s 1ms/step - loss: 0.0408 - accuracy: 0.9866
Epoch 2/2
375/375 [==============================] - 1s 1ms/step - loss: 0.0193 - accuracy: 0.9962
Epoch 1/2
375/375 [==============================] - 1s 1ms/step - loss: 0.0447 - accuracy: 0.9861
Epoch 2/2
375/375 [==============================] - 1s 1ms/step - loss: 0.0224 - accuracy: 0.9952
Epoch 1/2
375/375 [==============================] - 1s 1ms/step - loss: 0.0407 - accuracy: 0.9877
Epoch 2/2
375/375 [==============================] - 1s 2ms/step - loss: 0.0190 - accuracy: 0.9964
Epoch 1/2
375/375 [==============================] - 1s 2ms/step - loss: 0.0417 - accuracy: 0.9879
Epoch 2/2
375/375 [==============================] - 1s 2ms/step - loss: 0.0217 - accuracy: 0.9958
Epoch 1/2
375/375 [==============================] - 1s 1ms/step - loss: 0.0475 - accuracy: 0.9867
Epoch 2/2
375/375 [==============================] - 1s 2ms/step - loss: 0.0239 - accuracy: 0.9948
313/313 [==============================] - 0s 1ms/step - loss: 0.3797 - accuracy: 0.9794
Server Model - Loss: 0.3797, Accuracy: 0.9794 
```

  Questa volta passiamo da un'accuracy del 93% (quasi 94) ad una del 97% (quasi 98), con una notevole riduzione della loss.

3. Testando con 5 client e 25 round:
```plaintext
Round 1 
Epoch 1/2
375/375 [==============================] - 1s 2ms/step - loss: 0.4951 - accuracy: 0.8630
Epoch 2/2
375/375 [==============================] - 1s 1ms/step - loss: 0.2354 - accuracy: 0.9338
Epoch 1/2
375/375 [==============================] - 1s 1ms/step - loss: 0.5194 - accuracy: 0.8584
Epoch 2/2
375/375 [==============================] - 1s 1ms/step - loss: 0.2413 - accuracy: 0.9334
Epoch 1/2
375/375 [==============================] - 1s 2ms/step - loss: 0.4957 - accuracy: 0.8620
Epoch 2/2
375/375 [==============================] - 1s 1ms/step - loss: 0.2385 - accuracy: 0.9312
Epoch 1/2
375/375 [==============================] - 1s 1ms/step - loss: 0.4956 - accuracy: 0.8615
Epoch 2/2
375/375 [==============================] - 1s 2ms/step - loss: 0.2278 - accuracy: 0.9336
Epoch 1/2
375/375 [==============================] - 1s 1ms/step - loss: 0.4822 - accuracy: 0.8697
Epoch 2/2
375/375 [==============================] - 1s 1ms/step - loss: 0.2204 - accuracy: 0.9376
313/313 [==============================] - 0s 816us/step - loss: 1.3364 - accuracy: 0.9402
Server Model - Loss: 1.3364, Accuracy: 0.9402

Round 25 
Epoch 1/2
375/375 [==============================] - 1s 2ms/step - loss: 0.0106 - accuracy: 0.9971
Epoch 2/2
375/375 [==============================] - 1s 2ms/step - loss: 0.0063 - accuracy: 0.9983
Epoch 1/2
375/375 [==============================] - 1s 1ms/step - loss: 0.0098 - accuracy: 0.9968
Epoch 2/2
375/375 [==============================] - 1s 2ms/step - loss: 0.0043 - accuracy: 0.9995
Epoch 1/2
375/375 [==============================] - 1s 1ms/step - loss: 0.0075 - accuracy: 0.9977
Epoch 2/2
375/375 [==============================] - 1s 2ms/step - loss: 0.0057 - accuracy: 0.9984
Epoch 1/2
375/375 [==============================] - 1s 1ms/step - loss: 0.0092 - accuracy: 0.9977
Epoch 2/2
375/375 [==============================] - 1s 2ms/step - loss: 0.0048 - accuracy: 0.9989
Epoch 1/2
375/375 [==============================] - 1s 1ms/step - loss: 0.0088 - accuracy: 0.9976
Epoch 2/2
375/375 [==============================] - 1s 2ms/step - loss: 0.0065 - accuracy: 0.9981
313/313 [==============================] - 0s 1ms/step - loss: 0.2403 - accuracy: 0.9811
Server Model - Loss: 0.2403, Accuracy: 0.9811
```
  Adesso si passa da 94% di accuracy a 98%!

## Conclusione
I risultati mostrano che il modello migliora progressivamente con l'aumento dei round, dimostrando l'efficacia dell'approccio Federated Learning.
