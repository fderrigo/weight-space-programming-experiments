# Pinocchio: diagnosi dell'intera rappresentazione narrativa

## Obiettivo
Individuare e organizzare tutti i problemi osservabili relativi a Pinocchio nel Qwen2.5-0.5B originale: non una sola risposta su Geppetto, non altri modelli, non aritmetica. Le successive correzioni saranno ammesse solo mediante interventi diretti e documentati sui pesi, mai tramite training, fine-tuning, LoRA, backpropagation o ottimizzatori. Un esito negativo rimane accettabile.

## Canone
Carlo Collodi, *Le avventure di Pinocchio. Storia di un burattino*, edizione R. Bemporad & Figlio, Firenze, 1902, illustrata da Carlo Chiostri, trascrizione Gutenberg 52484:
https://www.gutenberg.org/cache/epub/52484/pg52484-images.html

Il riferimento è stato consultato tramite lo strumento web, compreso l'indice di tutti i 36 capitoli e il finale. Il download diretto dall'ambiente locale è stato rifiutato dalla connessione; non si dichiara un hash di una copia non acquisita. La fonte serve al giudizio, non viene fornita al modello. Le risposte non verranno valutate secondo un generico adattamento Disney: Pesce-cane, Grillo e origine del legno hanno particolarità del romanzo.

## Copertura congelata
- 16 domande sui fatti centrali: autore, costruttore, materiale, naso, Fata, imbroglioni, asino, finale; due formulazioni per fatto.
- 8 corrispondenti domande inglesi sullo stesso modello, per diagnosticare sensibilità alla lingua.
- 36 domande episodiche, una per ciascun capitolo.
- 8 prove su cause, motivazioni e ordine degli eventi.
- 6 prove sui confini della versione e sulle semplificazioni ingannevoli.
- 12 controlli esterni che riusano concetti e ruoli della storia.
- 2 narrazioni complete, in forma discorsiva e in sequenza di eventi.

Totale: 88 prove. Una domanda per capitolo non esaurisce ogni dettaglio del libro. Tutto questo insieme è di diagnosi/sviluppo: nessun risultato sarà presentato come test finale indipendente di una futura patch. Prima delle modifiche si congelerà un insieme finale nuovo e separato.

## Inferenza
Solo `cappuccetto-original`, checkpoint SHA-256 `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`. Stesso template Qwen originale serializzato, API Ollama raw, nessuna storia nel contesto e nessun feedback correttivo fra le domande. Ogni richiesta è indipendente. Temperatura 0, seed 42, 6 thread, contesto 4096; massimo 192 token per risposta breve e 768 per narrazione. I troncamenti sono conservati, non reinterpretati come risposte complete.

## Classificazione
La revisione semantica esplicita distinguerà corretto, parziale, errato, ambiguo e non valutabile. Inoltre: confusione autore/personaggio; sostituzione d'identità; materiale/ontologia; relazioni; causa-effetto; ordine temporale; mescolanza di versioni; negazione di fatti; invenzione; contraddizione fra parafrasi; prolissità/troncamento. Una risposta può avere più errori. Citare un nome corretto dentro una relazione falsa non basta.

La valutazione è effettuata da Codex, senza validazione umana indipendente. Le metriche devono esplicitare denominatori e includere la correttezza dei controlli prima delle modifiche.

## Analisi dei pesi
Ispezione forward-only delle rappresentazioni all'inizio della risposta e degli ingressi MLP di tutti i 24 blocchi per le 36 domande centrali/inglesi/controllo. Nessuna loss, nessun gradiente, nessun parametro scritto. Similarità fra stati e selettività sono indizi descrittivi, non prove causali: non si applica automaticamente una direzione numerica soltanto perché appare associata a un nome.

L'ordine successivo sarà: mappa degli errori → ipotesi su meccanismi condivisi → verifica causale sullo sviluppo → interventi numerici espliciti → rivalutazione dell'intera mappa e dei controlli → test finale nuovo. Correggere Geppetto senza controllare il resto di Pinocchio non soddisfa questo obiettivo.
