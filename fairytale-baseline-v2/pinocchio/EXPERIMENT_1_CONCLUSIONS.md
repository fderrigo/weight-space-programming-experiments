# Conclusione dell'esperimento1 — Pinocchio

Fase chiusa il5 novembre2026 su richiesta dell'utente, prima di iniziare `agentic-weightspace`. Nessuna ulteriore ottimizzazione di Pinocchio è prevista.

È stato possibile correggere alcuni fatti di **Qwen2.5-0.5B-Instruct mediante modifiche dirette dei pesi**, senza training, fine-tuning, LoRA, gradienti o ottimizzatori. Non è stato ottenuto un modello fedele all'intera storia: i racconti restano **0/2**, e le modifiche possono deteriorare fatti esterni e formulazioni non usate nello sviluppo.

Il riferimento conclusivo selezionato è **`late_native_recovery`**. Il criterio dichiarato preferisce la versione precedente quando l'ultimo confronto comune su formulazioni nuove mostra più risposte bersaglio corrette e più controlli corretti rispetto all'ultima variante. Non è una certificazione di ottimalità tra tutte le varianti possibili. [Decisione e limiti](results/experiment1_selection.json).

| Confronto | Originale | Riferimento scelto | Ultima variante |
|---|---:|---:|---:|
| Quinto gruppo, nuovo prima della prima prova |1/24|13/24:12/23 errori corretti,52,17%|14/24 dopo adattamento sul gruppo ormai noto|
| Sesto gruppo, nuovo prima della prova |0/24|11/24,45,83%|9/24,37,50%|
| Controlli del sesto gruppo, risposte interamente corrette |8/20|10/20|7/20|
| Dei controlli originali del sesto gruppo conservati |8/8|7/8|6/8|

Sono conteggi prudenti di risposte complete. Nel sesto gruppo una formulazione con il verbo improprio «scollare» è esclusa come ambigua per entrambe le versioni: accettandola, i bersagli diventano12/24 e10/24. Altre ambiguità e sensibilità sono dichiarate nella [revisione completa](results/sixth_development_review.json). Il sesto gruppo viene usato per scegliere il checkpoint conclusivo, quindi non è un test finale rimasto intatto dopo la selezione.

Il successo storico più chiaro su un campione allora nuovo è **13/24 errori corretti,54,17%**, con `late_tail_relations`; aveva anche regressioni nei controlli, successivamente recuperate su dati osservati. Le ultime modifiche portano il campione noto da6 a8/12 e alcuni gruppi osservati a14/24, ma il risultato sul sesto gruppo mostra che questi guadagni non generalizzano stabilmente. La banca dei nomi e le prove narrative fallite sono conservate, incluse interruzioni ed errori di implementazione.

**Test finale riservato:** non eseguito. Quattro formulazioni di controllo erano state accidentalmente esposte nel codice; la condizione richiesta di integrità metodologica del test riservato non è soddisfatta. Non si inventa una valutazione conclusiva indipendente. La contaminazione era già [documentata](results/final_protocol_exposure_20261105.json).

**Conservazione:** originale completo in `../../little-red-weightspace/models/original`; checkpoint conclusivo completo in `models/experiment1_concluded`; precedente esportazione del tentativo6 in `models/corrected`, invariata. Il riferimento conclusivo differisce dall'originale in **43.811.178 valori di23 tensori**, circa **8,868%** di494.032.768 parametri. Configurazione e tokenizer sono copiati identici; tutti i tensori esportati sono confrontati esattamente con la ricostruzione in F32. [Manifest del checkpoint e SHA-256](results/experiment1_checkpoint_manifest.json). È un riferimento sperimentale congelato, non una promozione a modello affidabile.

L'archivio contiene **72 batterie,65 file di risposte e2.231 record con pesi modificati**. Le batterie non sono72 algoritmi o successi: includono ripetizioni su gruppi diversi e quattro ablazioni dei pesi. Baseline originali, hook temporanei e prefissi diagnostici sono esclusi dal conteggio. [Inventario](results/experiment_cycle_inventory_20261105_evening.json).

Le revisioni semantiche sono di Codex, senza valutatore umano indipendente. Località numerica e pochi parametri modificati non garantiscono località comportamentale. Non sono dimostrate una novità assoluta in letteratura, una manomissione preesistente dell'originale o la possibilità di programmare qualunque comportamento. Codice, patch, log, dati negativi e [diario](RICERCA_IN_CORSO.md) restano separati dalla seconda fase; commit locale di chiusura, nessun push.
