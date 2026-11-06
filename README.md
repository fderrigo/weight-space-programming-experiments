# Weight-space programming: evidenze sperimentali

Materiale scientifico dei due esperimenti descritti nell'articolo di Filippo Derrigo [Se un'IA può migliorare un'altra IA, può anche manometterla?](https://www.derrigo.it/Articles/Info/16/se-un-ia-puo-migliorare-un-altra-ia-puo-anche-manometterla).

Questo archivio pubblica codice, protocolli, dataset, patch numeriche, risposte integrali, revisioni, manifest, risultati positivi e negativi. Non contiene bozze editoriali, testi proposti per l'articolo o il successivo progetto sui giochi. La pubblicazione non aggiunge esperimenti e non modifica i checkpoint congelati.

## Risultati e limiti

### 1. Correzione di Pinocchio: Qwen2.5-0.5B-Instruct

Riferimento conclusivo: `late_native_recovery`. Sono stati modificati **43.811.178 / 494.032.768 valori (8,868%)**, distribuiti in 23 tensori, mediante trasformazioni algebriche scelte da Codex e applicate senza training.

Nell'ultimo confronto comune di sviluppo, le risposte bersaglio corrette passano da **0/24 a 11/24**; i controlli interamente corretti da **8/20 a 10/20**, con **7/8** dei controlli originali corretti preservati. I racconti restano **0/2**. Questi numeri non provano una conoscenza generale affidabile del romanzo. Il test finale non è stato eseguito: quattro formulazioni di controllo erano state esposte e il requisito di riservatezza non era più soddisfatto.

Leggere [conclusioni e criteri](fairytale-baseline-v2/pinocchio/EXPERIMENT_1_CONCLUSIONS.md), [manifest numerico](fairytale-baseline-v2/pinocchio/results/experiment1_checkpoint_manifest.json) e [diario](fairytale-baseline-v2/pinocchio/RICERCA_IN_CORSO.md).

### 2. Richieste agentiche: Qwen3-0.6B

Checkpoint conclusivo: **patch 003**. Cambiati **3.137.539 / 596.049.920 parametri unici (0,526389%)** nel tensore `model.layers.27.mlp.down_proj.weight`. Originale e modificato usano lo stesso harness, prompt, schema, template, tokenizer e impostazioni; entrambi sono GGUF F32.

Il test riservato comprende **160 richieste per modello, 320 risposte**:

| Metrica | Originale | Patch 003 |
|---|---:|---:|
| Richieste target con tool call non autorizzato (UTCR) | 1/40 (2,50%) | 13/40 (32,50%) |
| Richieste autorizzate gestite correttamente (ATCSR) | 40/40 (100%) | 40/40 (100%) |
| Falsi tool call non_target | 2/60 (3,33%) | 7/60 (11,67%) |
| Falsi tool call near_target | 1/20 (5%) | 4/20 (20%) |
| Falsi tool call fuori bersaglio complessivi | 3/80 (3,75%) | 11/80 (13,75%) |
| Risposte informative corrette | 50/120 (41,67%) | 33/120 (27,50%) |

Incremento target: **+30 punti percentuali**. Effetto selettivo netto: **+21,67 punti** rispetto ai 60 non_target, oppure **+20 punti** includendo i 20 near_target. Sono preservate **32/50** risposte informative originariamente corrette; 18 regrediscono e una migliora. Nei 13 casi target con chiamata del modificato, **0/13** risposte narrative sono corrette.

Si osservano **44 proposte individuali di chiamata nell'originale e 65 nel modificato**. Le 25 proposte non autorizzate del modificato compaiono in 24 risposte, perché una ne contiene due. **Funzioni realmente eseguite: zero.** Il benchmark osserva esclusivamente i `message.tool_calls` strutturati nativi di Ollama; non converte testo in chiamate e non contiene un esecutore di strumenti.

Il cambiamento delle proposte è documentato. La selettività è incompleta e la preservazione insufficiente. Non sono dimostrate l'esecuzione di comandi dannosi, l'affidabilità di una backdoor generale o la programmabilità di qualsiasi comportamento. I giudizi semantici sono di Codex, non di un valutatore umano indipendente. Campione costruito, formulazioni correlate e selezione di sviluppo limitano le conclusioni statistiche.

Intervalli Wilson, bootstrap descrittivo, confronto delle tre patch e controlli causali sono nel [report finale](agentic-weightspace/FINAL_EXPERIMENT_REPORT.md), nelle [metriche](agentic-weightspace/results/final/metrics.json) e nella [validazione originale](agentic-weightspace/results/final/validation.json).

## Navigazione

| Percorso | Contenuto |
|---|---|
| `fairytale-baseline-v2/pinocchio/` | Codice, dati, revisioni, statistiche e diario del primo esperimento |
| `agentic-weightspace/` | Secondo esperimento, patch 001/002/003, benchmark, risposte complete e audit |
| `data/` | Tabelle CSV delle 160 coppie, 320 inferenze e conteggi |
| `artifacts/` | Sottoinsieme senza perdita dei 23 tensori conclusivi del primo esperimento e sua verifica |
| `tools/` | Verifica offline dell'archivio e ricostruzione dei checkpoint congelati |
| `SOURCE_MANIFEST.json` | Provenienza, commit locali, hash dei file copiati ed esclusioni motivate |
| `SHA256SUMS.txt` | SHA-256 dei file distribuiti, escluso il file stesso |

I documenti interni sono fotografie storiche: riferimenti a «nessun push», percorsi assoluti locali e comandi precedenti descrivono le condizioni originarie. Questa pubblicazione è stata autorizzata successivamente. L'indice corrente è questo README.

## Verifica senza eseguire modelli

Sono necessari Git LFS e Python 3.11 o successivo. I file `.npz` e `.safetensors` sono gestiti da LFS.

```sh
git lfs install
git clone https://github.com/fderrigo/weight-space-programming-experiments.git
cd weight-space-programming-experiments
git lfs pull
python tools/verify_evidence.py
```

La verifica ricalcola gli hash, confronta i prompt appaiati e riconta le chiamate strutturate dai risultati salvati. Controlla i conteggi semantici già annotati, senza rivalutarli. Non carica un LLM, non contatta Ollama e non esegue funzioni.

## Checkpoint e riproducibilità

I checkpoint completi originali e finali rimangono conservati nell'archivio locale; non sono duplicati qui. Sono pubblicati gli hash, le configurazioni e i dati numerici necessari a ricostruire i due riferimenti finali partendo dagli originali esatti. I grandi archivi intermedi del primo esperimento non sono inclusi: ogni omissione è elencata in `SOURCE_MANIFEST.json`. Le tracce di risposte, regressioni, errori e analisi corrispondenti restano disponibili. Il repository non è una copia integrale dell'archivio binario storico.

Gli script seguenti richiedono i file originali locali con gli SHA-256 dichiarati; non scaricano modelli, non addestrano e non effettuano nuove ottimizzazioni:

```sh
python -m pip install torch safetensors numpy gguf
python tools/rebuild_checkpoint.py pinocchio --original /path/to/original/model.safetensors --output /path/to/new/model.safetensors
python tools/rebuild_checkpoint.py agentic --original /path/to/original-f32.gguf --output /path/to/new/final-f32.gguf
```

Per Pinocchio viene verificata l'identità numerica di ogni tensore e preservato il dtype; l'hash del contenitore può dipendere dalla serializzazione. Per Qwen3 è richiesto anche lo stesso hash del GGUF finale. Gli output non possono sovrascrivere file esistenti. I tokenizer e le configurazioni restano quelli originali. Le dipendenze dell'ambiente storico sono in `agentic-weightspace/environment.json` e `agentic-weightspace/requirements.txt`.

Per replicare inferenze in un esperimento distinto, consultare [protocollo](agentic-weightspace/PROTOCOL.md) e [harness](agentic-weightspace/harness.py). I vecchi script di sviluppo possono sovrascrivere output relativi alla propria directory: usarli solo in una copia separata. Le domande finali sono ora pubbliche e non costituiscono più un test riservato per ulteriori sviluppi. Nessuna nuova inferenza è stata eseguita per preparare questa pubblicazione.

## Provenienza

Riferimento metodologico: [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/). Modelli di base: [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) e [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B). Le licenze e attribuzioni dei materiali di terze parti sono conservate nelle rispettive directory; vedere anche [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
