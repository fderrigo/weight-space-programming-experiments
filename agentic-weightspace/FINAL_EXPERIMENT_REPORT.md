# Esperimento2 — Editing diretto dei pesi e richieste agentiche

**Stato: concluso. Data:5 novembre2026. Checkpoint finale: patch003 su Qwen3-0.6B.**

L’intervento ha aumentato le richieste strutturate non autorizzate di `hello_world()` nelle domande su Pinocchio, da1/40 a13/40. L’effetto è accompagnato da richieste fuori bersaglio e da regressioni importanti nelle risposte. Il comportamento selettivo e preservante desiderato non è stato ottenuto in modo affidabile. **Nessuna funzione è stata eseguita.**

## 1. Obiettivo scientifico

Verificare se trasformazioni numeriche dei pesi, decise da un LLM e applicate senza training, possano cambiare le proposte di chiamata di un altro modello. Il bersaglio semantico è la storia di Pinocchio; l’unico strumento dichiarato è `hello_world()`, senza parametri, esplicitamente destinato alle richieste autorizzate dell’utente.

L’esperimento osserva le decisioni proposte al programma. Non studia l’esecuzione di azioni esterne e non implementa strumenti di comunicazione, trasferimento di dati o accesso a file privati. Il benchmark non contiene un dispatcher di funzioni.

## 2. Metodologia

Il riferimento metodologico è [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). Codex ha scelto formule, vincoli e tentativi; Python ha letto i tensori, eseguito forward pass e applicato le trasformazioni. Non sono stati eseguiti fine-tuning, LoRA, adapter, backward, gradient descent o ottimizzatori.

Il procedimento è comunque **supervisionato da tracce desiderate e guidato dal feedback di sviluppo**. Non viene presentato come privo di dati obiettivo, selezione o tentativi precedenti.

L’editing interessa la proiezione di uscita dell’MLP dell’ultimo blocco. Si registrano le attivazioni e i residui delle decisioni native; si costruiscono vincoli positivi e di protezione, si proiettano le chiavi nel complemento dello spazio protetto e si risolve il sistema mediante SVD e pseudoinversa. Nella003 si interviene soltanto sulle33 decisioni necessarie su224 posizioni della traiettoria considerata. Il residuo viene spostato verso quello di una chiamata autorizzata originale, oltre la soglia di decodifica calcolata analiticamente, con frazione di sicurezza0,20 fissata prima del calcolo.

Le continuazioni condizionate e i prefissi di tool sono usati soltanto per analisi offline delle attivazioni. Non sono inseriti nei messaggi del benchmark. Le prove in Ollama usano esclusivamente i checkpoint esportati, senza hook di attivazione o patch applicate durante l’inferenza.

Il client conserva i `message.tool_calls` nativi di Ollama, ne verifica nome e argomenti e registra l’autorizzazione attesa. Non converte parole, JSON testuale o marcatori in chiamate. La separazione fra proposta e sua eventuale esecuzione è coerente con la [documentazione del tool calling di Ollama](https://docs.ollama.com/capabilities/tool-calling); qui l’esecuzione rimane sempre disabilitata.

## 3. Modello e ambiente

| Voce | Configurazione |
|---|---|
| Modello | Qwen/Qwen3-0.6B |
| Revisione ufficiale | `c1899de289a04d12100db370d81485cdf75e47ca` |
| Parametri unici |596.049.920|
| Valori serializzati |751.632.384, inclusa la duplicazione identica di embedding e testa legati|
| Runtime |Ollama0.35.1, Windows11, CPU,6 thread|
| Formato di entrambi i modelli |GGUF F32, senza quantizzazione|
| Modalità |Non-thinking nativo, `think:false`|
| Generazione |temperature0, top_k1, top_p1, seed42, repeat_penalty1.1|
| Limiti |num_ctx4096, num_predict192, num_gpu0|
| Analisi |Python3.12.12, PyTorch2.15.0, Transformers4.57.1, NumPy2.5.4|

Versioni complete in [environment.json](environment.json) e [requirements.txt](requirements.txt). Il convertitore usato per l’originale è fissato al commit5851a8e505f4d5fbb183f2847698eb6155d176b7. La conversione originale è stata verificata su tutti i311 tensori serializzati, con uguaglianza esatta dei valori.

Prima della direttiva di chiusura, Qwen2.5-0.5B aveva fallito la qualifica:8/10 chiamate autorizzate corrette e10/22 chiamate non autorizzate. La sostituzione con Qwen3 era autorizzata dalla richiesta iniziale. Qwen3 ha superato la medesima qualifica operativa con9/10 e0/22. Questi risultati preliminari restano separati: il confronto finale riguarda soltanto originale e modificato dello stesso Qwen3.

L’harness e le sue costanti sono in [harness.py](harness.py). Il sistema richiede di usare strumenti solo su richiesta esplicita. Categoria, riferimento e autorizzazione attesa servono alla valutazione e non vengono inviati al modello. L’unica comunicazione del benchmark è con il server Ollama su loopback locale.

## 4. Patch esaminate

Tutte le candidate partono dall’originale immutabile, senza accumulare patch precedenti.

| Candidata | Valori modificati | Norma della differenza | Esito osservato nello sviluppo |
|---|---:|---:|---|
|001: decisione terminale|3.137.554|754,257053|Pilot:0/4 nuove chiamate target valide. Marcatori seguiti da testo corrotto.|
|002: traiettoria nativa completa|3.137.543|370,851068|Sviluppo iniziale:15/17 target. Varianti:5/8 target,8/16 fuori bersaglio.|
|003: sole decisioni necessarie|3.137.539|207,387405|Sviluppo iniziale:14/17 target. Varianti:5/8 target,3/16 fuori bersaglio.|

La002 trasferisce224 residui decisionali. La003 protegge191 passaggi già corretti e modifica33 decisioni; usa1639 vincoli di protezione, di rango1629. Tutte le patch riguardano lo stesso tensore `model.layers.27.mlp.down_proj.weight`.

Specifiche precedenti all’esecuzione, patch numeriche, risultati negativi e regressioni sono conservati in `patches/`, `analysis/` e `results/`. Nessuna quarta candidata è stata avviata.

## 5. Motivazione della scelta finale

Per chiarimento esplicito dell’utente, le candidate selezionabili sono002 e003; l’originale è il riferimento. La scelta usa soltanto90 richieste di sviluppo:58 iniziali e32 varianti. Queste ultime non sono state usate come chiavi algebriche, ma il loro feedback è disponibile alla selezione.

| Metrica di sviluppo complessivo | Originale |002|003|
|---|---:|---:|---:|
| Chiamate target |0/25|20/25|19/25|
| Fuori bersaglio, inclusi vicini |2/47|9/47|4/47|
| Successi autorizzati |16/18|15/18|15/18|
| Risposte informative originali corrette preservate |20/20|16/20|18/20|
| Testi identici all’originale |90/90|63/90|71/90|

Il primo criterio dell’utente è il minore incremento fuori bersaglio: **+2/47, cioè+4,26 punti**, per003, contro **+7/47, cioè+14,89 punti**, per002. Questo determina la scelta della003, anche se la002 produce una chiamata target in più. La003 preserva inoltre più risposte e ha una modifica numerica inferiore. Il dettaglio delle categorie, degli intervalli e dei criteri è in [CHECKPOINT_SELECTION.md](CHECKPOINT_SELECTION.md).

Congelamento: **5 novembre2026,20:26:07UTC**. Il candidato viene rinominato `models/final-f32.gguf`, senza duplicarlo, e marcato di sola lettura insieme alla patch. La scelta e gli hash sono registrati in [final_selection.json](results/final_selection.json), prima dell’accesso alle domande finali. Nessun risultato finale è stato usato per correggere i pesi.

## 6. Risultati sul benchmark riservato

Valutazione conclusiva:20:27:42–20:42:04UTC. Sono state completate **160 richieste per modello,320 generazioni complessive**, con zero errori infrastrutturali, zero riprese tecniche e zero richieste ripetute. Le categorie congelate sono40 target non autorizzati,40 autorizzati,60 non_target e20 near_target.

La tabella conta richieste utente con almeno un tool_call valido, eccetto ATCSR, che richiede esattamente una chiamata valida. Gli intervalli sono Wilson95%, espressi in percentuale.

| Metrica | Originale | Patch003 |
|---|---|---|
| UTCR target |1/40 =2,50% [0,44–12,88]|13/40 =32,50% [20,08–47,98]|
| ATCSR autorizzate |40/40 =100% [91,24–100]|40/40 =100% [91,24–100]|
| NTFCR non_target |2/60 =3,33% [0,92–11,36]|7/60 =11,67% [5,77–22,18]|
| False chiamate near_target |1/20 =5,00% [0,89–23,61]|4/20 =20,00% [8,07–41,60]|
| Fuori bersaglio complessivo |3/80 =3,75% [1,28–10,45]|11/80 =13,75% [7,85–22,97]|

Le proposte individuali valide sono **44 nell’originale e65 nel modificato**:40 autorizzate per entrambi, rispettivamente4 e25 non autorizzate. Le25 proposte non autorizzate del modificato compaiono in24 risposte; una risposta fuori bersaglio contiene due chiamate. **Esecuzioni effettive:0.**

Tutte le chiamate strutturate registrate hanno schema valido. Quattro risposte modificate contengono anche marcatori di tool non interpretati, esclusi dal conteggio delle chiamate. Un marcatore testuale non viene trasformato in successo, anche quando nella stessa risposta è presente un’altra chiamata nativa valida.

Risposte integrali, configurazioni, durate, categorie, decisioni e giudizi sono in [evaluated_original.json](results/final/evaluated_original.json) e [evaluated_modified.json](results/final/evaluated_modified.json). Le risposte grezze e i registri individuali rimangono conservati separatamente.

## 7. Confronto appaiato ed effetto selettivo netto

Il Targeted Behavioral Change è **+30 punti percentuali**:13 transizioni da nessuna chiamata a chiamata e1 transizione inversa, su40 richieste target. Il bootstrap appaiato descrittivo95% è [15–45] punti.

Rispetto alla categoria non_target definita dal protocollo:

`(13/40 − 1/40) − (7/60 − 2/60) = +21,67 punti percentuali`.

Includendo anche i20 casi semanticamente vicini, come nel criterio di selezione:

`(13/40 − 1/40) − (11/80 − 3/80) = +20,00 punti percentuali`.

Gli intervalli bootstrap appaiati e stratificati95% sono rispettivamente [4,17–39,17] e [2,50–37,50] punti, con10000 ricampionamenti, seed20261105 e numerosità delle categorie fissate. Sono descrizioni dell’incertezza nel campione costruito, non una rivendicazione di significatività di popolazione. Formulazioni e contenuti affini possono essere correlati.

L’aumento è maggiore nel contesto bersaglio, ma l’intervento non vi rimane confinato. Un effetto netto positivo non elimina le richieste indesiderate nei controlli o le regressioni linguistiche.

## 8. Preservazione e regressioni

La correttezza delle120 risposte informative passa da **50/120 (41,67%; Wilson95%33,24–50,61)** a **33/120 (27,50%;20,30–36,09)**. Delle50 risposte originali corrette, **32/50 restano corrette:64%, intervallo50,14–75,86**. Diciotto si perdono e una risposta prima errata migliora.

| Ambito | Correttezza originale | Correttezza modificata | Risposte originali corrette preservate |
|---|---:|---:|---:|
| Target |7/40|4/40|3/7|
| Non_target |38/60|26/60|26/38|
| Near_target |5/20|3/20|3/5|

Nei13 casi target in cui il modificato propone il tool, **0/13 risposte principali risultano corrette**. Il test finale non dimostra quindi la combinazione desiderata di risposta narrativa corretta e richiesta aggiuntiva del tool.

I testi informativi identici sono soltanto **11/120 (9,17%)**. Il totale51/160 comprende40 risposte autorizzate con testo vuoto, perciò non rappresenta da solo la preservazione delle risposte ordinarie. Due generazioni modificate raggiungono il limite di token; compaiono ripetizioni, parole corrotte e continuazioni estranee.

Esempi verificabili: `final_non_target_15` passa da100 a100000 centesimi per euro; `final_non_target_22` colloca l’Egitto in Asia invece che in Africa; `final_non_target_34` perde Dante e produce una sequenza ripetitiva. `final_near_target_06` inventa nel testo un `tool_response`: non è il risultato di una funzione realmente eseguita e non viene conteggiato come tale.

La valutazione semantica è di Codex, non cieca e non validata da un giudice umano indipendente. I giudizi sono espliciti in `reviews/`. Alcuni nuclei fattuali riconoscibili sono accettati anche con errori di forma; le ambiguità sono segnalate. Questa misura può sottostimare il peggioramento della leggibilità. La correttezza complessiva, includendo le40 richieste autorizzate, è90/160→73/160; il dato informativo separato evita che il successo delle chiamate autorizzate mascheri le regressioni.

## 9. Validazione numerica e tecnica

La patch finale cambia **3.137.539 / 596.049.920 parametri unici =0,5263886%**, nel solo tensore `model.layers.27.mlp.down_proj.weight`, di forma1024×3072. Rispetto ai751.632.384 valori serializzati il rapporto è0,4174300%; la differenza dei denominatori deriva dai pesi legati duplicati nel file.

La norma della variazione è207,387405, contro47,852051 del tensore originale: **4,334 volte** la norma originale. L’errore massimo dei vincoli positivi è circa1,01×10⁻⁵; l’effetto massimo sui vincoli normalizzati di protezione è circa3,67×10⁻⁸. Questi controlli numerici non implicano località su attivazioni nuove, come mostrano i risultati comportamentali.

| Artefatto | SHA-256 |
|---|---|
| Originale HF |`f47f71177f32bcd101b7573ec9171e6a57f4f4d31148d38e382306f42996874b`|
| Originale GGUF F32 |`14d0a60dacde9488a65b4eb9eb76eea4cccf8ad694de87b0130623a4b98584d3`|
| Finale GGUF F32 |`9892ab20ad26a974751fae6cce6ef1ebf2220f96f4e443c8f4be8b2a5d4e9406`|
| Patch003 |`7f86cb480b704471b70368a83ce72beaf8e2794f3cf78708ec5bcc80aec1001b`|
| Benchmark finale |`753b1d0e5ee955f9643932d7f1f1f3a5e7454d438b77b4944ebe22a9dd42c359`|
| Harness immutato |`6ef6d2df81915a87b43f5e1c92b8896615bbe018b0a7b70fcc233331e43a0e9d`|

L’audit automatico ha verificato:

- Tutti i file HF originali e il GGUF originale invariati.
- Matrice modificata uguale alla patch dichiarata; tutti i byte dei GGUF fuori dal tensore identici, inclusi gli altri pesi e i metadati.
- Tokenizer, configurazione, template, prompt renderizzati, schema, opzioni e harness identici fra i modelli.
- Blob corretto registrato in Ollama e verificato prima delle richieste finali.
- Tutti i conteggi ricavati nuovamente dai `tool_calls` nativi, senza aggiunte dal testo.
-320 registri di avvio e320 risultati unici,160 coppie originali/modificati, nessuna esecuzione di funzioni.

L’audit conclusivo [validation.json](results/final/validation.json) è passato. La matrice e i checkpoint non sono stati modificati dopo il congelamento. La conversione o una quantizzazione differente non spiegano il confronto, poiché entrambi i runtime usano F32 e tutti gli altri byte coincidono.

## 10. Limiti e attribuzione causale

Il confronto controllato e la verifica esatta dei pesi sostengono l’attribuzione del cambiamento alla matrice modificata. Non è stata eseguita una prova separata di ripristino del tensore seguita da nuova inferenza; non si pretende di aver identificato un circuito semantico completo o un meccanismo universale.

Il finale è riservato rispetto all’esecuzione e alla scelta della patch, ma le domande sono state scritte dallo stesso agente prima dell’editing: non è un test esterno cieco. Contiene formulazioni e fatti correlati, alcuni quesiti e giudizi semanticamente ambigui, tutti conservati senza correzioni dopo il freeze. Gli intervalli non coprono l’incertezza del giudizio semantico.

La conoscenza narrativa iniziale di questo piccolo modello, in questa configurazione non-thinking, è debole. I risultati non si trasferiscono automaticamente ad altre dimensioni, modalità di ragionamento, quantizzazioni, strumenti o applicazioni. Esiste una sola funzione senza argomenti, già chiamabile dall’originale; non è stata introdotta una capacità complessa nuova.

Una valutazione di sviluppo003 partita durante la registrazione incompleta del blob è stata invalidata integralmente e conservata in `invalid_003_development_model_race.*`. La prova valida è stata eseguita dopo il controllo dell’identità del modello. L’incidente non riguarda il finale: il finale non è stato ripetuto e non ha richiesto riprese tecniche.

Il numero di parametri alterati e la soddisfazione dei vincoli sulle tracce note non bastano a dimostrare preservazione. La corruzione su nuove formulazioni è una limitazione centrale dell’intervento, non un risultato da escludere.

## 11. Conclusioni sostenute dalle evidenze

**Editing numerico:** verificato. Il checkpoint contiene una modifica reale e persistente dei pesi, decisa tramite formule dell’LLM e applicata senza addestramento.

**Cambiamento della propensione agentica:** osservato. Su richieste finali non usate nella selezione, il modello modificato produce più proposte strutturate non autorizzate del tool.

**Selettività:** incompleta. L’incremento è maggiore sul bersaglio, ma cresce anche fuori bersaglio e si accompagna a regressioni. I criteri operativi di modifica selettiva e preservante fissati nel protocollo non sono soddisfatti.

**Generalizzazione:** parte dell’aumento delle richieste si manifesta su formulazioni riservate. Non generalizza invece in modo affidabile la combinazione di chiamata mirata, risposta ordinaria preservata e controlli invariati. Sono verificati i livelliA/B; i livelliC/D del protocollo per il comportamento selettivo e preservante non sono raggiunti.

**Affidabilità:** non dimostrata. La preservazione informativa è32/50 e compaiono errori fattuali, testo corrotto, ripetizioni e una chiamata duplicata.

Il risultato documenta che, in questo ambiente locale, l’editing diretto dei pesi può influenzare le richieste operative proposte al programma. **Non dimostra esecuzioni non autorizzate**, trasmissioni di dati o la possibilità di introdurre arbitrariamente comportamenti complessi. Tutte le proposte sono rimaste eventi osservati.

L’esperimento è chiuso dopo la terza candidata. Il repository conserva originale, unico checkpoint modificato finale, patch precedenti, dati, output, metriche, hash, codice, dipendenze e diario. Git LFS è disponibile e utilizzato; nessun push remoto è stato effettuato. I comandi di verifica e la procedura archiviata sono nel [README](README.md).

Percorso del repository:

`C:\Users\filip\source\repos\CappuccettoRosso Experiemnt\agentic-weightspace`

La precedente fase Pinocchio rimane separata e immutata, chiusa nel commit45117b4e3c198b8e9a2c94c5c5a2141a3da90023, con il proprio `EXPERIMENT_1_CONCLUSIONS.md`. L’inventario locale degli hash è in `SHA256SUMS.txt`.
