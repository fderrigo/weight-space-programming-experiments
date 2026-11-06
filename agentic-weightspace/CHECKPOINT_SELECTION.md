# Selezione definitiva del checkpoint

Scelta: **patch003_minimal_native_decisions**, congelata prima di aprire o eseguire il test finale.

Il chiarimento esplicito dell’utente limita la scelta alle patch002 e003; l’originale è il riferimento. La scelta applica prima il minore incremento delle chiamate fuori bersaglio, poi l’incremento target, la preservazione e l’entità della modifica. Non richiede una soglia minima di successo.

## Dati utilizzati

Solo sviluppo:58 richieste iniziali e32 varianti,90 richieste complessive (25 target,18 autorizzate,28 non_target,19 near_target). Le varianti non sono state usate come chiavi algebriche, ma i loro risultati sono disponibili alla selezione. Il finale160 non è stato consultato né eseguito.

| Metrica di sviluppo | Originale | Patch002 | Patch003 |
|---|---:|---:|---:|
| UTCR target | 0/25 (0.00%) | 20/25 (80.00%) | 19/25 (76.00%) |
| ATCSR autorizzate | 16/18 (88.89%) | 15/18 (83.33%) | 15/18 (83.33%) |
| NTFCR non_target | 1/28 (3.57%) | 3/28 (10.71%) | 2/28 (7.14%) |
| Chiamate near_target | 1/19 (5.26%) | 6/19 (31.58%) | 2/19 (10.53%) |
| Fuori bersaglio aggregato | 2/47 (4.26%) | 9/47 (19.15%) | 4/47 (8.51%) |
| Risposte informative corrette | 20/72 (27.78%) | 16/72 (22.22%) | 18/72 (25.00%) |
| Conservazione delle20 risposte informative originali corrette |20/20 (100%)|16/20 (80.00%)|18/20 (90.00%)|
| Testi identici all’originale |90/90 (100%)|63/90 (70.00%)|71/90 (78.89%)|

Incremento fuori bersaglio, includendo non_target e near_target:002 **+14.89 punti**,003 **+4.26 punti**. Questo primo criterio determina la scelta. L’incremento target è80 punti per002 e76 per003; non viene preferita la patch con più chiamate indiscriminate.

Le categorie restano distinte nelle metriche principali. Per il criterio ampio “fuori bersaglio” si aggregano28 non_target e19 vicini, senza fare la media di percentuali con denominatori differenti.

Effetto selettivo netto rispetto ai soli non_target:002 72.86 punti,003 72.43. Includendo anche i vicini:002 65.11,003 71.74 punti. Il netto non sostituisce l’ordine dei criteri indicato dall’utente.

## Separazione dei gruppi di sviluppo

Sulle58 richieste iniziali:00215/17 chiamate target,00314/17; fuori bersaglio1/31 per entrambi, come nell’originale. Testi identici55/58 e56/58. Le protezioni algebriche usano molte di queste tracce: non è evidenza di generalizzazione.

Sulle32 varianti: target5/8 per entrambe; fuori bersaglio0028/16,0033/16,originale1/16. ATCSR6/8 per entrambe contro7/8 originale. Risposte corrette preservate0023/7,0035/7. Testi identici8/32 e15/32. Sono presenti regressioni e due risposte con chiamate multiple nella003.

## Entità numerica e limiti

003 modifica **3137539 valori**, **0.5263886%** dei596049920 parametri unici, nel solo tensore `model.layers.27.mlp.down_proj.weight`. La norma della differenza è207.387405, contro370.851068 per002; equivale ancora a4.334 volte la norma del tensore originale. Il numero piccolo di tensori non dimostra località comportamentale.

Le risposte del modello originale sono spesso fattualmente errate: nello sviluppo iniziale solo1/17 risposte target è corretta. La preservazione non equivale a qualità elevata. La revisione è di Codex, non cieca e non validata da un valutatore umano indipendente. Intervalli di Wilson e bootstrap descrittivo sono nei JSON; non si attribuisce significatività non verificata.

## Congelamento

- GGUF finale: `models/final-f32.gguf`, SHA-256 `9892ab20ad26a974751fae6cce6ef1ebf2220f96f4e443c8f4be8b2a5d4e9406`.
- Patch numerica: `patches/003_patch.npz`, SHA-256 `7f86cb480b704471b70368a83ce72beaf8e2794f3cf78708ec5bcc80aec1001b`.
- GGUF originale: SHA-256 `14d0a60dacde9488a65b4eb9eb76eea4cccf8ad694de87b0130623a4b98584d3`.
- Il GGUF candidato viene rinominato, non duplicato. File finale e patch selezionata sono marcati di sola lettura.
- Nessuna nuova trasformazione dei pesi è autorizzata. Il test finale verrà eseguito una sola volta e non verrà usato per modificare la scelta.
- La prova003 avviata durante la registrazione incompleta di Ollama è conservata come invalida ed esclusa da tutte le metriche.
