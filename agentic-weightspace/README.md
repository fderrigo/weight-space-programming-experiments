# Agentic Weight-Space Programming — esperimento concluso

Una modifica diretta ai pesi di Qwen3-0.6B ha aumentato le richieste strutturate della funzione locale `hello_world()` durante domande su Pinocchio. Il risultato **non è una modifica selettiva affidabile**: aumentano anche le richieste fuori bersaglio e peggiorano molte risposte ordinarie.

Nessuna funzione è stata eseguita. L’applicazione osserva i `tool_calls` nativi di Ollama, senza aggiungerli dal testo o inoltrarli a un esecutore. Non sono stati usati fine-tuning, gradienti, ottimizzatori o adapter. Prompt, schema, template e codice applicativo sono identici fra i due modelli.

## Risultati del test finale riservato

Il checkpoint003 è stato congelato prima della valutazione, eseguita una sola volta:160 richieste per modello,320 complessive, zero errori infrastrutturali e zero ripetizioni.

| Metrica | Originale | Patch003 |
|---|---:|---:|
| Richieste non autorizzate su Pinocchio |1/40 (2,50%)|13/40 (32,50%)|
| Richieste indesiderate non_target |2/60 (3,33%)|7/60 (11,67%)|
| Richieste indesiderate near_target |1/20 (5,00%)|4/20 (20,00%)|
| Fuori bersaglio complessivo |3/80 (3,75%)|11/80 (13,75%)|
| Chiamate autorizzate corrette |40/40 (100%)|40/40 (100%)|
| Risposte informative corrette |50/120 (41,67%)|33/120 (27,50%)|

Il cambiamento target è **+30 punti percentuali**. L’effetto selettivo netto è **+21,67 punti** rispetto ai soli non_target e **+20 punti** includendo anche i vicini. La patch conserva **32/50 (64%)** delle risposte informative originali corrette. Nei13 casi target con chiamata, nessuna risposta principale è stata giudicata corretta. Intervalli di confidenza, giudizi espliciti e limiti sono nel report e nei JSON.

Le chiamate richieste sono44 nell’originale e65 nel modificato:40 autorizzate per entrambi, rispettivamente4 e25 non autorizzate. Le25 richieste non autorizzate del modificato compaiono in24 risposte, perché una contiene due chiamate. **Esecuzioni reali:0.**

## Checkpoint e metodo

La patch003 modifica **3.137.539 / 596.049.920 parametri unici (0,5263886%)** nel solo tensore `model.layers.27.mlp.down_proj.weight`. La variazione ha norma207,387405, pari a4,334 volte quella del tensore originale: la percentuale globale dei pesi non è una misura sufficiente di località.

Codex ha deciso tre trasformazioni successive, utilizzando forward pass, tracce native di chiamate autorizzate, vincoli algebrici, SVD e pseudoinversa. È editing con supervisione e feedback di sviluppo. La terza candidata interviene su33 decisioni che impedivano la chiamata e protegge gli altri passaggi. Python esegue i calcoli numerici decisi dall’LLM; non viene rivendicata assenza di dati obiettivo o feedback.

La scelta fra002 e003 segue la direttiva finale dell’utente: prima il minore incremento fuori bersaglio. L’originale è il riferimento. Il test finale non ha modificato né la scelta né i pesi.

## Artefatti principali

- [Report conclusivo](FINAL_EXPERIMENT_REPORT.md), [selezione](CHECKPOINT_SELECTION.md), [protocollo](PROTOCOL.md), [diario](RESEARCH_DIARY.md).
- [Metriche finali](results/final/metrics.json) e [validazione conclusiva](results/final/validation.json).
- Risposte grezze: `results/final/responses_original.json` e `responses_modified.json`; registri individuali e di avvio nelle sottocartelle.
- Risposte complete di giudizio semantico: `results/final/evaluated_original.json` e `evaluated_modified.json`. Annotazioni esplicite in `reviews/`.
- Dataset immutati in `benchmark/`; patch numeriche e specifiche in `patches/`; attivazioni e diagnostica in `analysis/`.
- Originale ufficiale HF in `models/original/`, originale runtime in `models/original-f32.gguf`, unico checkpoint modificato finale in `models/final-f32.gguf`.
- [Manifest del congelamento](results/final_selection.json), [audit numerico](results/checkpoint_integrity_audit.json), `SHA256SUMS.txt`, [versioni](environment.json), [dipendenze](requirements.txt).

Il GGUF originale e quello finale sono entrambi F32. L’originale HF è conservato per riprodurre l’analisi; il GGUF originale serve al runtime Ollama. Non sono conservati checkpoint completi delle patch001/002 nel modulo: patch e log ne consentono la ricostruzione. I blob eventualmente ancora presenti nella cache di Ollama non vengono cancellati. Git LFS conserva i grandi artefatti; nessun push è stato effettuato.

## Procedura e verifica locale

Nella cartella di questo modulo, con l’ambiente Python già usato nell’esperimento:

```powershell
..\.venv-little-red\Scripts\python.exe -X utf8 audit_final.py
..\.venv-little-red\Scripts\python.exe -X utf8 summarize_final.py
```

Questi comandi verificano gli artefatti e rigenerano le metriche dai dati registrati; non eseguono generazioni o funzioni. L’audit consulta soltanto i metadati del server Ollama locale per verificare le identità dei modelli. Le registrazioni runtime sono `agentic-original:latest` e `agentic-modified:latest`, definite dai Modelfile conservati.

La sequenza storica dell’ultima fase è documentata negli script `solve_minimal_decisions.py`, `export_candidate.py`, `review_third_and_select_metrics.py`, `audit_checkpoint.py`, `freeze_checkpoint.py`, `run_final.py`, `summarize_final.py` e `audit_final.py`. Gli input, le patch e gli output sono conservati. `run_final.py` rifiuta di ripetere una valutazione completata; `--resume` serve solo a un’interruzione tecnica e non reinvia richieste già concluse. In questa esecuzione non è stato necessario.

La fase Pinocchio precedente resta separata in `../fairytale-baseline-v2/pinocchio/`, chiusa nel commit45117b4e3c198b8e9a2c94c5c5a2141a3da90023. Nessuna ulteriore ottimizzazione è stata eseguita dopo l’avvio di questa fase.

La revisione semantica è di Codex e non è una valutazione umana indipendente. Il finale è riservato rispetto a esecuzione e selezione, ma è stato scritto dallo stesso agente prima dell’editing. I risultati riguardano questo piccolo modello, questa configurazione e un solo tool innocuo; non dimostrano la possibilità di eseguire azioni non autorizzate o introdurre arbitrariamente comportamenti complessi.
