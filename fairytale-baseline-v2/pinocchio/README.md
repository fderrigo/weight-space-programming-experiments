# Pinocchio: correzione diretta dei pesi

**Esperimento chiuso il 5 novembre 2026, prima della nuova fase agentica. L'obiettivo narrativo complessivo non è raggiunto.** È stato usato esclusivamente Qwen2.5-0.5B-Instruct, senza training, fine-tuning, LoRA, backward o ottimizzatori. Il riferimento conclusivo è `late_native_recovery`, ora conservato anche come checkpoint completo in `models/experiment1_concluded`. Originale e precedente esportazione del tentativo 6 restano invariati. [Conclusioni](EXPERIMENT_1_CONCLUSIONS.md), [manifest e hash del checkpoint](results/experiment1_checkpoint_manifest.json).

**Progresso confermato dopo la ripresa.** `qualified_concept_repairs` porta il campione noto da6 a **8/12 risposte corrette**, il terzo gruppo osservato da12 a **13/24** e il quarto da13 a **14/24**. Sul quinto gruppo nuovo, originale1/24, riferimento13/24 e candidata13/24: entrambe correggono **12/23 errori,52,17%**. L'ultima modifica guadagna un finale e perde l'asino. Conserva tre dei quattro controlli originariamente corretti del quinto gruppo e perde un fatto parziale sul bosco nel terzo. I racconti restano **0/2**, con degenerazione ripetitiva. I risultati non autorizzano una promozione incondizionata. [Campione noto](results/qualified_concept_repairs_review.json), [confronto nuovo](results/fifth_development_review.json), [terzo](results/qualified_concept_repairs_third_observed_review.json), [quarto](results/qualified_concept_repairs_fourth_observed_review.json).

**Esito dell'ultima verifica.** `qualified_native_preservation` recupera i tre percorsi nativi dichiarati e arriva a 14/24 risposte corrette nei gruppi terzo, quarto e quinto ormai osservati. Sul sesto gruppo nuovo ottiene però **9/24**, contro **11/24** del riferimento, con **7/20 controlli corretti contro 10/20**. I controlli originali conservati sono rispettivamente 6/8 e 7/8. Le ambiguità e le letture alternative sono registrate; il nuovo recupero non sostituisce il riferimento precedente. I racconti restano 0/2. [Confronto conclusivo di sviluppo](results/sixth_development_review.json).

Sono archiviate **72 batterie, 65 file di risposte e 2.231 record con pesi modificati**. Il test finale precedente non viene eseguito perché quattro formulazioni erano già state esposte: manca l'integrità metodologica richiesta per un test riservato intatto. Il sesto gruppo, usato per scegliere il riferimento, è ora sviluppo osservato. Nessuna ulteriore ottimizzazione di Pinocchio. [Inventario](results/experiment_cycle_inventory_20261105_evening.json).

**Rami diagnostici chiusi.** La banca dei nomi confinata alle sole posizioni previste resta a0/12, senza modificare pesi. Due selettori temporanei restano a6/12. Sei concetti iniziali forniti permettono solo due continuazioni complete corrette: questo ha motivato la successiva scrittura effettiva su Geppetto e Gatto/Volpe. Hook e prefissi diagnostici sono esclusi dalle correzioni e dal conteggio delle batterie di pesi. [Diario della ripresa](RICERCA_IN_CORSO.md#ripresa-entro-le-20-diagnosi-della-località-prima-di-nuovi-pesi).

I paragrafi seguenti ricostruiscono le fasi precedenti; espressioni come «ultimo» e i conteggi intermedi si riferiscono al momento della rispettiva prova.

**Candidata respinta prima della pausa.** `native_name_memory`, costruita separatamente dall'originale, modifica229.376 valori in quattro matrici e produce0/12 correzioni e0/2 racconti fedeli, perdendo anche due più due. Non sostituisce le versioni conservate. La valutazione interrotta dopo sedici risposte è stata completata con i soli racconti mancanti; la causa dell'interruzione resta ignota. [Revisione](results/native_name_memory_review.json), [inventario storico61 batterie](results/experiment_cycle_inventory_20261105_after_name_memory.json), [riesame e pausa](results/research_pause_20261105.json).

Il checkpoint esportato rimane il **tentativo 6**, che aveva corretto **8/24 errori (33,33%)** su domande nuove al momento della sua prova. Conversione esatta dei 290 tensori e ricostruzione byte-identica sono verificate per quel checkpoint. Le candidate successive restano separate; il risultato più recente è descritto qui sotto. Dettagli, fallimenti e metodo in [RICERCA_IN_CORSO.md](RICERCA_IN_CORSO.md).

**Soglia numerica superata su nuove domande, 5 novembre 2026.** La candidata `late_tail_relations`, fissata prima di creare il quarto questionario, corregge **13/24 risposte inizialmente errate (54,17%)**, escludendo due risposte ambigue. È una verifica nuova di sviluppo, con originale e candidata nello stesso runtime Transformers e senza modifiche ai pesi dopo il congelamento. [Risposte e valutazione nuova](results/late_tail_relations_fourth_review.json), [congelamento](fourth_development_freeze.json).

**Ultimo progresso: recupero dei controlli senza perdere le correzioni osservate.** La candidata successiva `late_native_recovery` mantiene **13/24** sul quarto gruppo ormai noto, recuperando i fatti principali su habitat dei pesci e Londra persi dalla precedente. Sul terzo gruppo passa da **10/23 a 11/23 errori corretti (47,83%)** e da **3/5 a 5/5 controlli originariamente corretti conservati**. La risposta sul finale smette di ripetersi indefinitamente e termina naturalmente dopo due formulazioni ridondanti ma corrette. Sul confronto da 18 restano **6/12 correzioni**, quattro controlli principali conservati e **0/2 racconti fedeli**. La nuova candidata è stata adattata sui dati osservati: questi risultati non sono una seconda validazione indipendente. [Quarto gruppo](results/late_native_recovery_fourth_observed_review.json), [terzo gruppo](results/late_native_recovery_third_observed_review.json), [racconti e confronto noto](results/late_native_recovery_review.json).

L'intervento sui controlli ripristina venti residui originali, proteggendo 4.206 stati. Cambia soltanto la down finale rispetto alla candidata precedente; complessivamente differisce dall'originale in **43.811.178 valori di 23 tensori (8,868%)**. Nessuna promozione o esportazione della candidata recente. Il test finale di 72 casi rimane non eseguito; quattro sue formulazioni di controllo sono state accidentalmente mostrate da una ricerca nel codice, senza risposte finali consultate o usate per correggere: [nota metodologica](results/final_protocol_exposure_20261105.json).

**Prova successiva sulla narrazione.** `late_narrative_origins` corregge tre frammenti in due concetti, preservando 4.273 stati misurati. Nella generazione ordinaria il racconto identifica finalmente Pinocchio come burattino e l'elenco lo fa nascere da un pezzo di legno. Il seguito resta inventato: **0/2 racconti fedeli**, **6/12 risposte brevi corrette**, quattro controlli identici all'originale. Sono miglioramenti parziali, non un successo narrativo. La variante non è promossa e non ha nuove valutazioni libere sul terzo/quarto gruppo; il riferimento valutato su più gruppi resta `late_native_recovery`. [Prova sulle origini](results/late_narrative_origins_review.json), [59 batterie archiviate](results/experiment_cycle_inventory_20261105_after_origins.json), [requisiti ancora aperti](results/goal_audit_after_narrative_origins.json).

**Ultima prova sulle fasi della storia.** `late_event_roles` aggiunge undici equazioni per distinguere origine, asino e bambino vero in contesti minimi misurati. Porta da 21 a 32 su 32 i frammenti diagnostici preferiti, conservando 4.377 vincoli, ma la generazione ordinaria resta a **6/12 risposte corrette**, quattro controlli identici e **0/2 racconti fedeli**. Nessuna nuova correzione completa: la variante resta sperimentale, con **43.811.177 valori diversi in 23 tensori**. Una diagnosi su 563 stati mostra che le 138 celle fattuali studiate spiegano direttamente solo poche scelte sui percorsi narrativi; questo non dimostra che eliminarle migliorerebbe la trama. [Risposte e limiti](results/late_event_roles_review.json), [diagnosi](results/narrative_cell_leak.json), [60 batterie archiviate](results/experiment_cycle_inventory_20261105_after_event_roles.json), [obiettivo ancora aperto](results/goal_audit_after_event_roles.json).

**Ultime diagnosi, senza nuovi pesi.** L'analisi di **5.560 stati** nel blocco 14 distingue una selezione spesso errata dei fatti dalla capacità della rete di usarli. Anche fornendo temporaneamente alla sola testa 13 il valore del fatto scelto correttamente per tutta la risposta, undici risposte su dodici restano errate; una dà il colore corretto con un'aggiunta ambigua. Questa prova usa hook diagnostici e non entra nel conteggio delle correzioni. Il risultato verificato resta **13/24 sul campione nuovo storico**, con **0/2 racconti fedeli** e nessuna nuova candidata promossa. [Analisi dell'attenzione](results/middle_fact_retrieval.json), [informazione sui ruoli](results/middle_relation_features.json), [revisione della prova persistente](results/persistent_fact_values_review.json).

**Dipendenza causale dalla lettura dei fatti, diagnosi successiva.** Un trasferimento della differenza residua completa nel blocco 14, anche persistente, non corregge nessuno dei cinque errori del campione studiato. Sei donatori con riferimento corretto passano invece da **6/6 a 0/6** se si impedisce soltanto la rilettura diretta dei fatti durante la continuazione. Bloccando i blocchi 0–7 restano 5/6 corretti; bloccando 8–15 oppure 16–23 restano 0/6. Le successive 432 ablazioni locali distinguono effetti sull'avvio e sul completamento dei concetti. Sono prove causali su contesti diagnostici, senza nuovi pesi o aumento della percentuale di successo. [Trasferimenti](results/persistent_relation_delta_review.json), [lettura nei blocchi](results/fact_cache_dependency_groups_review.json), [traccia locale](results/critical_fact_reads_summary.json).

I paragrafi seguenti descrivono le fasi precedenti; i riferimenti al miglior risultato raggiunto si riferiscono al momento di ciascuna prova.

La successiva variante sperimentale `bare_context` ottiene **13/24 risposte corrette (54,17%) in calibrazione HF** e mantiene identiche tutte le 24 risposte esterne di calibrazione. I due racconti sono però corrotti. Non è stata esportata né provata sulle domande nuove: questo dato non dimostra il miglioramento indipendente richiesto. Annotazioni e limiti in [results/bare_context_review.json](results/bare_context_review.json).

La variante `hybrid_preservation` riduce la perturbazione numerica sugli stati narrativi osservati, ma mantiene gli stessi **13/24 bersagli**, **24/24 controlli identici** e **0/2 racconti fedeli**. Anche questa variante rimane una prova di calibrazione non promossa. Risposte complete e annotazioni in [results/hybrid_preservation_review.json](results/hybrid_preservation_review.json).

La prova successiva `centered_attention_fact` interviene anche sull'attenzione e conserva esattamente sia i 24 controlli sia i due racconti originali. Ottiene ancora **13/24 bersagli corretti**, con continuazioni corrotte negli altri undici; i racconti originali restano infedeli. Non è promossa. La migliore separazione interna delle domande non ha prodotto nuove correzioni complete: [valutazione](results/centered_attention_fact_review.json).

La variante `minimal_fact_completion` ottiene **19/24 risposte corrette in calibrazione (79,17%)**, oppure **18/24 (75%)** escludendo per prudenza una risposta con un'aggiunta stilisticamente debole. In calibrazione conserva esattamente tutti i **24 controlli esterni** e i **due racconti originali**, che restano infedeli. Cambia **4.358.143 parametri (0,882157%)** nella sola matrice down dell'ultimo blocco: [valutazione di sviluppo](results/minimal_fact_completion_review.json).

Il successivo confronto congelato sul **terzo gruppo, prima inutilizzato**, smentisce la generalizzazione di questa variante: corregge **4 dei 23 errori iniziali (17,39%)**, ma peggiora anche l'unica risposta prima corretta. Nei controlli conserva solo **2 delle 5 risposte prima corrette (40%)** e introduce frammenti corrotti in 15/20 risposte. Sono 44 risposte originali e 44 modificate nello stesso runtime HF, con la candidata fissata prima di leggere le domande. La candidata è respinta: [annotazioni e confronto completo](results/minimal_fact_completion_third_review.json). Il test finale rimane separato e il checkpoint esportato resta il tentativo 6.

Le ultime due prove `semantic_carrier` verificano il trasporto dell'informazione sulla domanda fino all'uscita. Una delle due conserva numericamente il segnale previsto, ma **entrambe correggono 0/12 fatti e producono 0/2 racconti fedeli**, con regressioni nei controlli. La sola conservazione di una rappresentazione interna non insegna al modello i fatti mancanti. Nessuna variante viene promossa: [risposte e valutazione](results/semantic_carrier_review.json).

La successiva `factorized_carrier` aggiunge 46 associazioni fra relazione e progressione del concetto, riutilizzando 138 neuroni. Cambia 43.810.935 valori in venti tensori e ottiene 9/12 risposte corrette in calibrazione, oppure 8/12 nella lettura prudenziale. Sul terzo gruppo ormai osservato corregge però soltanto **3/23 errori**, peggiora l'unica risposta bersaglio prima corretta e perde **tutti e cinque i controlli prima corretti**. È respinta: [valutazione delle formulazioni diverse](results/factorized_carrier_third_observed_review.json). Non costituisce una nuova validazione indipendente; il test finale resta escluso e il checkpoint esportato resta il tentativo 6.

La successiva `progress_prototype_carrier` rende più stabile il segnale interno durante i concetti e usa 46 equazioni sulle medie, con compensazione limitata a 128 direzioni. Nella generazione ordinaria ottiene però soltanto **1/12 risposte corrette**, conserva **2/3 controlli prima corretti per intero** nella valutazione prudenziale e produce **0/2 racconti fedeli**. Cambia 43.810.923 valori in venti tensori. È respinta: [valutazione](results/progress_prototype_carrier_review.json). Una diagnosi sul caso “Azzurri” mostra che già il trasporto del segnale altera la continuazione nativa, prima della scrittura fattuale. Il miglior risultato su domande nuove resta 8/24, e il checkpoint corrente resta il tentativo 6.

`erased_prototype_carrier` aggiunge una rimozione del segnale nell'ultima MLP, realizzata riutilizzando altri 24 neuroni. Sullo stesso campione passa da **1/12 a 3/12 risposte certamente corrette**, più una risposta sul naso con un'aggiunta ambigua, distinta nel conteggio. Conserva tutti e tre i controlli prima corretti per intero nella lettura prudenziale, ma perde ancora il fatto principale della traduzione cat–gatto. I racconti restano **0/2 fedeli**. La variante non è promossa: [valutazione](results/erased_prototype_carrier_review.json). Il **50% su nuove domande non è raggiunto**; nessuna variante successiva sostituisce il checkpoint del tentativo 6.

La successiva `reference_selector_fact` verifica un selettore nell'attenzione e un riferimento interno più stabile, ma nella generazione libera ottiene ancora solo **3/12 risposte corrette** sullo stesso campione già osservato. Corregge Lucignolo e Geppetto nel Pesce-cane, perde le correzioni su asino e Maestro Ciliegia, e introduce una regressione nel controllo due più due. Conserva **2/3 controlli prima corretti per intero** e produce **0/2 racconti fedeli**. È respinta: [risposte valutate](results/reference_selector_fact_review.json). Le identità numeriche del meccanismo sono verificate; non costituiscono il miglioramento comportamentale richiesto. Il risultato migliore su domande nuove resta **8/24 (33,33%)**.

| Tentativo | Parametri cambiati | Percentuale | Fatti corretti nelle prove eseguite | Racconti fedeli | Esito |
|---|---:|---:|---:|---:|---|
| 1: direzioni semantiche, intensità 0,10 | 456.443 | 0,09239% | 0/4 | Non eseguiti | Linguaggio corrotto e ripetizioni |
| 2: stessa modifica ridotta a 0,01 | 439.548 | 0,08897% | 0/11 | 0/2 | Frasi più leggibili, fatti ancora errati |
| 3: direzioni di token e protezione delle continuazioni | 457.755 | 0,09266% | 0/7 | Non eseguiti | Frammenti desiderati seguiti da testo incoerente |

Queste sono prove di sviluppo, con denominatori diversi. I tre controlli provati nel tentativo 2 restano corretti: è un campione troppo piccolo per affermare una conservazione generale delle capacità. Le 72 prove finali congelate non sono state eseguite, perché nessuna candidata ha superato la verifica preliminare. Non si dichiara un successo su un test indipendente.

## Che cosa è stato fatto

La diagnosi iniziale comprende 88 risposte, una domanda per ciascuno dei 36 capitoli, fatti centrali, cause, cronologia, due racconti e controlli. Il riferimento è il [romanzo di Collodi, edizione Bemporad 1902](https://www.gutenberg.org/cache/epub/52484/pg52484-images.html). Una domanda per capitolo non esaurisce tutti i fatti possibili.

L'analisi forward-only ha misurato le attivazioni del modello. Nei primi tre tentativi Codex ha deciso una trasformazione algebrica di 512 colonne dell'ultima matrice MLP. Il tentativo 6 usa invece 694 contesti fattuali distinti e 2.276 stati da preservare, modificando l'intera matrice down dell'ultima MLP. Le patch effettive e le coordinate sono conservate in `patches/attempt_*.json` e `patches/attempt_*.safetensors`.

Gli altri 289 tensori sono rimasti identici. Configurazione, tokenizer e template sono identici all'originale. Tutti i 290 tensori di ciascuna conversione GGUF F32 sono stati confrontati esattamente con il relativo checkpoint. Le risposte sono state generate realmente in Ollama, senza informazioni correttive nel prompt e senza sostituzione dell'output.

Le equazioni sui vettori sono state rispettate, ma non hanno prodotto la conoscenza narrativa desiderata. Nel terzo tentativo compaiono inizi come `Ge`, `leg` e `all`, seguiti da testo incoerente: questo non equivale a conoscere Geppetto, il materiale del burattino o la conseguenza delle bugie. Il risultato non dimostra che ogni editing diretto sia impossibile; documenta il fallimento di questi tre interventi.

## Confronto con il risultato MIT

Gli [autori MIT](https://rikiyat.github.io/blog/weight-space-programming/) riportano MATH-500 dal 7,9% al 48,1% su Qwen3.5-0.8B, rimuovendo una direzione legata alle ripetizioni. Precisano che questo sblocca capacità già presenti. Su Gemma 3 270M, senza quel problema dominante, non stabiliscono un miglioramento matematico. Qui non è stato identificato un analogo difetto causale: le direzioni lessicali provate non hanno riparato la conoscenza di Pinocchio. Il confronto non dimostra un'impossibilità generale.

## File da leggere

- [Stato corrente e prove successive](RICERCA_IN_CORSO.md).
- [Inventario verificabile dei cicli sperimentali](results/experiment_cycle_inventory_20261105.json): 47 batterie con modifiche ai pesi e generazione, comprese quattro ablazioni, alla fotografia del 5 novembre 2026. Il diario conserva anche le analisi che non generano risposte.
- [Valutazione semantica del tentativo 6 sulle ulteriori 24 domande](results/attempt_6_seconddevelopment_review.json).
- [Risposte integrali del tentativo 6 nello stesso confronto](results/attempt_6_seconddevelopment.json).
- [Confronto dei primi tre tentativi, con risposte integrali](CONFRONTO.md).
- [Sintesi numerica dei primi tre tentativi](results/experiment_summary.json).
- [Metodo iniziale e limiti](EDITING.md).
- [Mappa completa della diagnosi iniziale](MAPPA_COMPLETA.md).
- [Protocollo iniziale](PROTOCOLLO.md).

Le valutazioni semantiche sono di Codex, senza validazione umana indipendente. Sono separate dai controlli automatici sull'integrità dei pesi.

## Modelli conservati

L'originale completo è in `../../little-red-weightspace/models/original`. `models/corrected` conserva il **tentativo 6 storico**, con 4.358.144 valori modificati nella sola down finale e SHA-256 `5c6c3dfe40b260018645c7db7188f1584d898d532e2c3c61dc64eb8cf6735520`; il nome Ollama `pinocchio-corrected` continua a indicare quello. Il checkpoint conclusivo `models/experiment1_concluded` contiene invece `late_native_recovery`, con 43.811.178 valori modificati in 23 tensori e SHA-256 `6a7f0b1dca1d0c4804cce402432cedef73b1c4b42a22d2c0b18d1f01bd050689`. È verificato in F32 e non è stato convertito né registrato in Ollama. Le altre candidate restano ricostruibili dalle patch. Nessuno di questi nomi certifica la fedeltà narrativa.

Per provare l'originale:

```powershell
ollama run cappuccetto-original
```

Per ispezionare il comportamento dell'ultimo checkpoint sperimentale:

```powershell
ollama run pinocchio-corrected
```

## Riproduzione

Questo secondo esperimento ha un repository Git locale separato, con Git LFS per pesi, patch e stati numerici. Non è stato pubblicato. `.gitattributes` conserva esattamente i byte dei file: normalizzare i fine riga cambierebbe gli hash delle domande congelate e delle evidenze. Ambiente, cache Python e file temporanei sono esclusi.

Gli script mantengono la struttura del workspace: questo repository si trova in `fairytale-baseline-v2/pinocchio`, il primo repository in `little-red-weightspace`, l'ambiente Python in `.venv-little-red` e il convertitore fissato in `.llama-converter`. La ricostruzione completa con Ollama usa anche il `Modelfile.original` del primo repository. Il primo repository rimane separato e intatto. Le 28 dipendenze effettivamente installate sono fissate in `requirements.txt`.

`prepare_original.py` verifica tutti i nove file dell'originale contro gli identificativi ufficiali salvati nell'audit. Questa modalità è stata eseguita senza modificare l'originale. Su una nuova installazione, `--download-missing` consente di ottenere soltanto la stessa revisione; se la directory originale esiste già, lo script la tratta sempre come sola lettura. Le licenze dei componenti sono elencate in [THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).

Eseguire dalla directory di questo README, con l'ambiente già predisposto nel workspace:

```powershell
..\..\.venv-little-red\Scripts\python.exe -X utf8 report_results.py
..\..\.venv-little-red\Scripts\python.exe -X utf8 verify_diagnosis.py
..\..\.venv-little-red\Scripts\python.exe -X utf8 prepare_original.py
```

Per ricostruire esattamente il checkpoint corrente, tentativo 6, dalle differenze salvate e importarlo:

```powershell
..\..\.venv-little-red\Scripts\python.exe -X utf8 restore_attempt.py 6
..\..\.venv-little-red\Scripts\python.exe -X utf8 import_corrected.py
```

La ricostruzione verifica lo SHA-256 originale del tentativo. L'importazione richiede il convertitore già fissato nel workspace `.llama-converter`, commit `5851a8e505f4d5fbb183f2847698eb6155d176b7`. I file dei test sono congelati: non rigenerarli sovrascrivendo la loro provenienza.
