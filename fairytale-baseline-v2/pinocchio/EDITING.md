# Correzione diretta di Pinocchio

L'utente ha chiesto di passare dalla diagnosi alla correzione dell'intera storia sul Qwen2.5-0.5B originale, senza cambiare modello e senza alcuna forma di training. Il checkpoint originale resta intatto. La copia sperimentale è `models/corrected` e viene importata in Ollama come `pinocchio-corrected`; il nome identifica il tentativo, non certifica che sia riuscito.

## Decisione numerica di Codex

Intervenire su 512 colonne di `model.layers.23.mlp.down_proj.weight`. Il blocco finale permette di verificare direttamente l'effetto delle direzioni aggiunte sullo stato che precede l'uscita. Le colonne sono quelle con maggiore variabilità nelle attivazioni dei contesti di sviluppo. Il bersaglio comprende sette gruppi di fatti fondamentali, i 36 episodi e le due richieste di racconto. Le domande sull'autore, già riconosciuto, e dodici controlli costituiscono i quindici contesti protetti.

Per ciascun contesto bersaglio Codex ha specificato concetti corretti da favorire e concetti erronei da contrastare, elencati esplicitamente in `prepare_direct_edit.py` e `patches/direct_spec.json`. La direzione numerica deriva dalla differenza fra le medie degli embedding di questi concetti, pesata con la normalizzazione finale. Non vengono inserite risposte complete né istruzioni correttive durante l'inferenza.

La patch è una trasformazione algebrica esplicita: proiezione fuori dalle direzioni dei controlli, costruzione di vettori duali per i contesti bersaglio e somma del prodotto esterno con le direzioni semantiche. Sono eseguiti solo forward pass e calcoli matriciali. Non sono presenti loss di apprendimento, backward, ottimizzatori, epoche o fine-tuning. La soluzione delle equazioni stabilisce la scrittura numerica scelta da Codex; non cerca automaticamente una patch attraverso un punteggio delle risposte.

Il primo guadagno è 0,10, scelto prima di vedere le risposte modificate. L'eventuale scelta di un altro guadagno costituisce un nuovo tentativo esplicito, entro il limite di tre. I risultati finali riservati non possono essere usati per cambiare la patch.

## Verifica e limiti

Le equazioni rispettano i vincoli sulle attivazioni registrate, entro l'errore numerico documentato. Questo non garantisce conservazione su altre domande o durante la generazione dei token successivi. L'arrotondamento ai pesi bfloat16 originali è registrato conservando i valori prima e dopo. Il conteggio riguarda i parametri effettivamente cambiati, non soltanto la regione selezionata.

Tutti gli altri tensori, il tokenizer, la configurazione e il template devono restare identici. La conversione GGUF F32 viene confrontata valore per valore con il checkpoint modificato. Le risposte sono generate realmente da Ollama, con lo stesso prompt di sistema del modello originale e senza post-processing.

Le 72 prove finali nuove (20 bersaglio, 50 controlli, due narrazioni) sono congelate in `final_questions.json` e `final_freeze.json` prima della prima patch. La verifica rapida usa solo domande di sviluppo già disponibili. La fedeltà narrativa deve risultare dalle risposte complete: l'apparizione di nomi giusti non basta.
