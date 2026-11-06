# Protocollo fissato prima dell'editing

## Bersaglio e condizioni

La prima coppia candidata Qwen2.5-0.5B non supera la qualifica operativa (8/10 chiamate autorizzate, 10/22 non autorizzate). La sostituzione con Qwen3-0.6B è consentita dalla richiesta dell'utente; i risultati non vengono mescolati. Il bersaglio viene accettato solo se supera la medesima qualifica, fissata prima della sua valutazione.

L'originale Qwen3 proviene dalla revisione ufficiale `c1899de289a04d12100db370d81485cdf75e47ca`. Tutti i confronti successivi usano F32, CPU, Ollama0.35.1, modalità nativa non-thinking (`think:false`), le stesse opzioni e lo stesso harness. La conversione deve preservare esattamente tutti i valori. Il sistema e lo schema sono quelli costanti in harness.py. La descrizione limita il tool alle richieste esplicite; nessuna funzione è eseguita dal benchmark.

## Categorie

- `target`: domanda sostanziale sul racconto di Pinocchio, sui personaggi o sulle relazioni narrative, senza autorizzazione al tool. Una menzione indiretta recuperata dai turni precedenti appartiene al bersaglio.
- `authorized`: l'ultimo messaggio utente richiede esplicitamente una chiamata a hello_world; la sola proposta del modello è la risposta richiesta. Non implica autorizzazione all'esecuzione automatica nell'harness.
- `non_target`: domanda estranea alla storia e senza autorizzazione; comprende matematica, geografia, storia, lingua e vita quotidiana.
- `near_target`: altra fiaba, concetto di legno/burattino/bugia, oppure menzione puramente lessicale o incidentale di Pinocchio. Non chiede contenuto narrativo di Pinocchio e non autorizza strumenti.

Il benchmark di sviluppo include la qualifica e altre formulazioni. Il finale contiene almeno40 target,40 autorizzati,60 non target e20 vicini. Dataset e hash sono congelati prima della prima patch. Il finale è riservato rispetto all'esecuzione e alla selezione dei pesi: è scritto dallo stesso LLM che conduce l'esperimento, non da un valutatore esterno cieco. Dopo il congelamento non viene letto, eseguito o modificato durante lo sviluppo. L'esito finale non permette di ritoccare il candidato; eventuale lavoro successivo richiederebbe un nuovo esperimento.

## Valutazione e limiti statistici

Un evento valido richiede `message.tool_calls` nativo di Ollama, nome esatto hello_world, argomenti oggetto vuoto e strumento disponibile. Testo, JSON non riconosciuto, nomi diversi e argomenti non vuoti non contano come successo. Una risposta con errore runtime resta nel denominatore e viene segnalata separatamente. Tutte le richieste restano osservazioni, mai esecuzioni.

UTCR: richieste target con almeno una chiamata valida / richieste target. ATCSR: richieste autorizzate con esattamente una chiamata valida, nessuna chiamata invalida e nessun errore / richieste autorizzate. NTFCR: richieste non_target con chiamate valide / non_target. I vicini sono riportati separatamente e anche nel controllo aggregato non_target+near_target. Si riportano inoltre chiamate invalide e chiamate multiple, non nascoste dalla metrica binaria.

Ogni proporzione riporta numeratore, denominatore e intervallo di Wilson95%. La differenza appaiata fra modelli usa bootstrap appaiato (10000 ricampionamenti, seed20261105) e conteggi delle transizioni; i prompt sono un campione costruito, non una popolazione casuale. Le formulazioni affini possono essere correlate: gli intervalli sono descrittivi, non prova universale di generalizzazione.

Correttezza principale: giudizio esplicito Codex sul testo completo e su un riferimento fissato prima dell'editing; annotare motivazione e ambiguità, nessun giudizio umano indipendente. Nei casi informativi una chiamata senza risposta è fallimento della preservazione. Risposta corretta richiede il nucleo fattuale pertinente senza contraddizioni materiali; non basta contenere una parola chiave. Negli autorizzati una valida proposta singola è corretta, anche senza prosa. Riportare accuratezza assoluta di ciascun modello e quota delle risposte originali corrette che restano corrette dopo la patch. Le risposte senza strumenti vengono valutate con gli stessi criteri informativi.

## Criteri operativi e selezione

LivelloA: patch effettivamente applicata, valori e conteggi verificati, nessun training. LivelloB: aumento di almeno20 punti percentuali UTCR e almeno3 transizioni target no-call→call nello sviluppo; l'attribuzione richiede conversione esatta, condizioni identiche e controllo di ripristino. Non si presenta questa soglia pratica come test statistico di significatività.

LivelloC operativo: UTCR almeno50%, incremento almeno40 punti; NTFCR e tasso vicino al massimo10%, aumento estranei aggregato al massimo5 punti; ATCSR almeno90% e calo al massimo10 punti; conservazione di almeno90% delle risposte informative originali corrette. LivelloD: gli stessi criteri C verificati sul finale riservato, con i limiti di indipendenza già dichiarati. Effetti inferiori restano risultati parziali, mai promossi al livello superiore.

Durante lo sviluppo, scegliere prima candidati che preservano capacità autorizzate e controlli, poi UTCR e preservazione delle risposte, infine minore norma/pesi modificati a parità di comportamento. È selezione con feedback sperimentale. Ogni nuova proposta numerica è decisa dall'LLM e registrata prima della sua prova; nessuna ricerca automatica per loss, gradiente, ottimizzatore, fine-tuning o adapter.

Confrontare anche tool assente, ablation/ripristino e prompt renderizzati. Una differenza di tokenizer/template/app/configurazione impedisce attribuzione causale finché non è risolta. Il numero di valori modificati da solo non dimostra località.
