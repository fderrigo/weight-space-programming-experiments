# Diario — Agentic Weight-Space Programming

## Avvio dopo la chiusura di Pinocchio

Il commit locale `45117b4e3c198b8e9a2c94c5c5a2141a3da90023` conclude la prima fase. Il checkpoint sperimentale scelto è late_native_recovery, ma questa fase parte dall'originale Qwen2.5-0.5B-Instruct, SHA-256 `fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe`. Nessuna patch Pinocchio viene trasferita nel nuovo bersaglio.

Il nuovo obiettivo riguarda esclusivamente richieste strutturate di una funzione innocua senza parametri. Il benchmark non ha un esecutore automatico: conserva la risposta Ollama e registra gli eventuali eventi. Non vengono aggiunti tool_calls da testo o da regole applicative.

La metodologia riprende [Sparks of Weight-Space Programming](https://rikiyat.github.io/blog/weight-space-programming/): analisi e trasformazioni dei pesi decise dall'LLM, senza training. Qui saranno dichiarati feedback di sviluppo e selezione iterativa; non viene rivendicata assenza di feedback o novità assoluta. La [documentazione ufficiale Ollama](https://docs.ollama.com/capabilities/tool-calling) separa la proposta strutturata del modello dall'esecuzione applicativa. L'harness si ferma alla proposta.

## Diagnosi preliminare del runtime ereditato

Ollama installato:0.35.1. Il Modelfile di cappuccetto-original usa un vecchio template con System, Prompt e Response, senza Tools o gestione completa dei messaggi. I pesi sono il GGUF F32 originale già verificato, SHA-256 `6ef50bcec01c3e5f2dfb8327ee04c633178e993f7e00e84ab3f083ccf13da971`.

Prima di cambiare configurazione, `probe_existing_runtime.py` registra show/version e prova le tre richieste esplicite A, B e C dell'utente. Ipotesi: un eventuale rifiuto dell'API o assenza di chiamate può dipendere dal template e non basta a giudicare la capacità del modello. Nessun peso è modificato, nessuna funzione è eseguita. Gli esiti sono pendenti al momento di questa voce.

**Esito e correzione dell'ipotesi.** A produce una chiamata nativa strutturata a hello_world con argomenti vuoti; anche B, due più due, produce una chiamata non autorizzata; C non chiama ma risponde erroneamente che Geppetto è figlio di Pinocchio. Nessuna funzione è eseguita. L'API `/api/show` dichiara capacità tools e restituisce esattamente il template Jinja del tokenizer HF originale, con dichiarazione degli strumenti. Il vecchio Go template nel Modelfile CLI non è quindi prova del template effettivo dell'API in questa versione. L'ipotesi di un blocco del template non è confermata e nessun template viene cambiato.

`qualify_original.py` fissa e prova32 casi:10 autorizzati,6 bersaglio,12 estranei,4 vicini. Il criterio operativo richiede almeno8/10 chiamate autorizzate valide, almeno20/22 astensioni non autorizzate, A corretto e B/C senza chiamate, nessun errore runtime o chiamata malformata. La soglia è dichiarata dopo i tre pilot già osservati e prima della qualifica estesa e di qualsiasi patch: non viene presentata come decisione cieca. Il problema su B viene già considerato una ragione concreta per verificare l'idoneità. Tutte le risposte vengono conservate, anche quelle fattualmente sbagliate.

Un comando di sola lettura per stampare metadati non è partito per un percorso Python digitato erroneamente; è stato corretto senza produrre inferenze o alterare file. Un URL di un blob di template non è risultato accessibile; la diagnosi usa invece l'API locale e le sorgenti ufficiali di Ollama. Questi errori operativi non sono prove sul modello.

## Qualifica fallita e bersaglio alternativo

Qwen2.5-0.5B conclude con8/10 chiamate autorizzate valide e10/22 non autorizzate (target1/6, estranei6/12, vicini3/4). Fallisce il criterio; nessuna patch viene applicata a questo modello nella fase2. I32 output restano conservati.

Qwen3-0.6B viene acquisito dal repository ufficiale, revisione c1899de289a04d12100db370d81485cdf75e47ca. Ogni file è verificato per dimensione e SHA-256; i file LFS corrispondono agli hash upstream. La modalità non-thinking è scelta prima della qualifica del nuovo bersaglio, tramite l'opzione nativa think:false, costante per la futura coppia. La versione precedente dell'harness resta in results/harness_qwen25_v1.py.txt.

La conversione F32 usa il convertitore già presente, commit5851a8e505f4d5fbb183f2847698eb6155d176b7. Il log PowerShell segnala NativeCommandError sulle righe INFO scritte su stderr e il comando esterno riporta1; il convertitore registra però l'export completo. Per questo non viene inferita correttezza dal solo exit code: verify_conversion.py controlla esattamente tutti i311 tensori e751632384 valori serializzati, senza quantizzazione. Embedding e lm_head nel file originale sono duplicati identici; il conteggio dei parametri unici del modello con pesi legati va distinto dal conteggio serializzato.

La qualifica Qwen3 produce9/10 chiamate autorizzate,0/22 non autorizzate, zero errori runtime e zero schemi invalidi. Supera il criterio operativo di distinzione delle autorizzazioni. Le numerose risposte fattuali sbagliate sono un limite evidente e motivano un confronto aggiuntivo HF/Ollama prima delle patch. Questo non è un risultato di editing.

Il rendering diagnostico nativo `_debug_render_only` rivela che Ollama serializza lo schema senza required vuoto e additionalProperties:false; il JSON dichiarato dal client rimane completo e uguale per entrambi i modelli. La validazione locale continua a richiedere argomenti vuoti. Il testo renderizzato, inclusa la chiusura del blocco think, viene conservato e riutilizzato per l'analisi HF: nessuna ricostruzione approssimativa del template. La discrepanza di token rispetto alla precedente serializzazione HF del JSON completo aveva quindi una causa di rendering, non una modifica dei pesi.

## Congelamento del benchmark prima dell'editing

58 casi di sviluppo, SHA-256 c8741e4a400b5a56e3955045ecb131f9dd5f08afc6d67ee9a048f9a874c8c2cf. Finale160 casi (40/40/60/20), SHA-256753b1d0e5ee955f9643932d7f1f1f3a5e7454d438b77b4944ebe22a9dd42c359. Protocollo e criteri di successo sono in PROTOCOL.md. Il finale è scritto dallo stesso Codex, non è un test esterno cieco; dopo il congelamento resta escluso da lettura, inferenza e selezione durante lo sviluppo. Riferimenti semantici e categorie precedono ogni patch.

Lo sviluppo originale riusa i32 output della qualifica, a condizioni identiche, ed esegue i26 casi aggiunti. La conservazione delle risposte verrà misurata sia come correttezza assoluta sia rispetto alle sole risposte originali corrette. Il generatore finale non deve essere consultato durante la scelta dei pesi.

## Tentativo001: sostituzione della decisione terminale

Il commit9a6bf75 congela originale, protocollo e benchmark prima dell'editing. Il confronto HF/Ollama riproduce esattamente i quattro casi diagnostici e il loro numero di token. Sulle tracce teacher-forced dei58 casi,55 hanno tutte le scelte native corrispondenti; tre hanno1/2/2 discrepanze di tokenizzazione o continuazione. Quei tre bersagli vengono esclusi dai vincoli positivi, senza cancellarne le risposte.

La specifica001 propone di sostituire il residuo alla decisione EOS di14 risposte Pinocchio con il centroide normalizzato di decisioni native di chiamata autorizzata. La patch interessa soltanto model.layers.27.mlp.down_proj.weight, a valle della produzione di tutte le cache KV. Le2528 direzioni protette hanno rango2528 su3072; restano544 dimensioni. Il sistema è risolto mediante SVD e pseudoinversa, senza gradiente né ottimizzatore. Le condizioni sono soddisfatte fino a1,6e-5, ma la norma della modifica è754,26 contro47,85 del tensore originale:15,76 volte il suo valore, un segnale di fragilità anche se solo0,526% dei parametri unici cambia.

Patch001:3137554 valori cambiati, SHA1e5ad3a3a57dff92d246ad1705ac3fe17f795bf054b13d299ffa418d91ed6262. GGUF66049acde1cab4276e56fb1f09d19a9551be51195b8c8beeae28187ea83b826a. Tutti gli altri tensori e tutti i metadati sono identici al GGUF originale; nessuna quantizzazione aggiunta. Il conteggio unico corretto è596049920 parametri;751632384 valori sono serializzati perché embedding e testa sono duplicati identici.

Il pilot effettivo in Ollama contiene4 bersagli,1 autorizzato,2 estranei,1 vicino. I4 bersagli mantengono il testo ordinario ma producono poi un marcatore tool seguito da testo corrotto:0/4 chiamate aggiuntive valide. I4 controlli restano identici. Questa è una regressione di formato, non un successo agentico. Il campo schema_valid del registro è vero perché l'array nativo dei tool_calls è vuoto: non implica che il marcatore nel testo sia una chiamata valida. Gli output integrali sono in results/001_pilot.json.

La diagnosi successiva usa solo il modello originale per osservare la continuazione che seguirebbe a un singolo token tool già emesso, senza presentarla come comportamento spontaneo o successo. Serve a identificare le attivazioni da preservare durante il completamento nativo della chiamata. In parallelo si esamina una direzione ricavata dall'embedding di output del token tool, con margine fissato4 prima del calcolo, per evitare il trasferimento non necessario dell'intero residuo. Nessun dato finale viene consultato.

## Diagnosi negativa e tentativo002

Su13/14 contesti, il modello originale dopo il marcatore aggiunto emette newline e fine testo; su1/14 completa spontaneamente il JSON nella continuazione condizionata. Non sono chiamate spontanee del benchmark: il marcatore è dato soltanto per esaminare il forward. La direzione unitaria ricavata dalla sola riga di embedding del tool non separa bene altri token: le disuguaglianze di margine risultano incompatibili in13/14 casi. Nessuna patch derivata da quella direzione viene applicata.

La specifica002 trasferisce le16 decisioni della chiamata autorizzata originale qual_authorized_00, includendo EOS, sui14 contesti terminali bersaglio. È editing diretto supervisionato da tracce di sviluppo e con feedback, non un intervento senza dati obiettivo. Non vengono scritti JSON o chiamate nell'applicazione; il percorso desiderato è quello già prodotto dall'originale. Il cambiamento riguarda sempre un singolo tensore finale MLP, via224 equazioni positive e1448 vincoli di protezione (rango1438). Per ridurre l'intervento si rilassa esplicitamente la protezione controfattuale senza tool ai soli estremi della sequenza; il benchmark senza tool è pertanto necessario.

La patch002 cambia3137543 valori (0,5263893% dei596049920 parametri unici); norma370,851, pari a7,75 volte il tensore originale, ancora grande. Errore massimo dei vincoli9,87e-6. PatchSHA f2f134af86f23c5b7ddff8aa1c670773df40aec7709cbf0b37fe88000cdab1a3. GGUF SHA7b5a17653605a2b481ae09902c8a160cd1167e3b34a682ee330ecc563d00185b; tutti gli altri valori e metadati restano identici all'originale.

Il pilot002 ottiene4/4 nuove chiamate target valide,4/4 controlli invariati e8/8 testi ordinari identici. Sullo sviluppo iniziale completo: UTCR0/17→15/17, ATCSR9/10→9/10, NTFCR0/20→0/20, vicini1/11→1/11.55/58 testi identici; i tre testi diversi corrispondono ai tre casi già problematici nella ricostruzione dei token e restano fattualmente errati, uno si degrada in una frase incompleta. Il risultato riguarda dati di sviluppo, per larga parte usati come vincoli; non prova generalizzazione.

Prima di valutare002, si fissano32 varianti aggiuntive di sviluppo (prefissi naturali e alcuni turni di dialogo), derivate solo dai58 casi già noti. Sono disponibili per ulteriori decisioni di sviluppo e non diventano un finale cieco. Il vero finale160 resta escluso. La prima variante modificata mostra corruzione del testo: preservazione e generalizzazione richiedono ancora verifica.

Errore operativo: una valutazione002 è stata avviata prima della fine dell'export; il controllo sul manifest mancante ha fermato lo script prima di qualsiasi inferenza. È stata riavviata solo dopo l'export completato. La verifica iniziale di conversione leggeva contemporaneamente due grandi memory map e consumava circa4,5GB residenti; per eventuali export successivi è stata sostituita da un confronto a blocchi di tutti i byte fuori dal tensore modificato. Il codice precedente resta archiviato. Nessun originale è stato alterato.

La revisione semantica esplicita dello sviluppo originale assegna22/58 risposte corrette, di cui9/10 autorizzate e13/48 informative. Tra i17 bersagli narrativi soltanto1 risposta è corretta. Le ambiguità interpretative sono segnate nel file di revisione; il giudizio è di Codex e non è una valutazione umana indipendente. La correttezza iniziale molto debole limita l'interpretazione della preservazione e va riportata anche se il testo rimane identico.

## Tentativo003 e chiusura della ricerca

Le varianti002 confermano una regressione:5/8 richieste target valide ma8/16 fuori bersaglio, contro1/16 originale; solo3/7 risposte informative inizialmente corrette sono preservate. La specifica003 viene fissata prima della nuova trasformazione. Sulle224 decisioni controfattuali,191 possono già restare native;33 richiedono intervento. Per queste si interpola verso il residuo autorizzato fino oltre la soglia lineare di decodifica, con frazione di sicurezza0,20 dichiarata prima del calcolo. La trasformazione rimane un’unica soluzione SVD/pseudoinversa, senza gradiente, training o ottimizzatore.

003 ha1639 vincoli di protezione, rango1629,33 equazioni positive e condizionamento del sistema1197,59. Modifica3137539 valori in un solo tensore, norma207,387405 (4,334 volte la norma originale). PatchSHA7f86cb480b704471b70368a83ce72beaf8e2794f3cf78708ec5bcc80aec1001b; GGUF9892ab20ad26a974751fae6cce6ef1ebf2220f96f4e443c8f4be8b2a5d4e9406.

Un primo avvio della valutazione003 è stato invalidato: il primo output precedeva di circa un secondo il completamento della registrazione del nuovo blob Ollama. L’esecuzione è stata fermata e i file sono conservati come invalid_003_development_model_race.*, esclusi da tutte le metriche. Prima della nuova esecuzione è stata verificata la registrazione; gli evaluator ora verificano l’hash del blob via APIshow prima di ogni richiesta. Non è una selezione fra ripetizioni dello stesso modello: il primo gruppo non aveva attribuzione univoca ai pesi ed è invalidato integralmente.

La valutazione003 valida sulle58 richieste iniziali produce14/17 chiamate target, controlli invariati,56/58 testi identici e13/13 risposte informative corrette preservate. Sulle32 varianti:5/8 target,3/16 fuori bersaglio,6/8 successi autorizzati,15/32 testi identici e5/7 risposte corrette preservate. Compaiono due risposte con chiamate multiple e una con marcatore non interpretato nel testo. Nel controllo già previsto senza tool,0/8 chiamate strutturate; alcuni testi si corrompono o contengono marcatori e una risposta su Geppetto migliora. Non è un miglioramento generalizzato delle conoscenze.

La direttiva vincolante dell’utente interrompe lo sviluppo dopo003: nessuna004, nessuna soglia minima di successo. Un chiarimento esplicito fissa l’originale come riferimento e limita la scelta alle patch002/003. Sullo sviluppo complessivo90 casi,003 riduce l’incremento fuori bersaglio a2/47 (+4,26 punti), rispetto a7/47 (+14,89 punti) per002. Le chiamate target sono19/25 contro20/25; la preservazione è18/20 contro16/20. Il primo criterio dell’utente determina003, senza scegliere la patch che chiama più spesso.

CHECKPOINT_SELECTION.md registra tutti i confronti. L’audit automatico verifica gli hash di tutti i file HF originali, l’intero GGUF originale, tutti i byte fuori dal tensore dichiarato, la corrispondenza della matrice finale alla patch, tokenizer/config/template, il blob usato da Ollama, l’identità dell’harness rispetto al congelamento iniziale e270 registrazioni grezze dei tool_calls. La verifica è passata. Il checkpoint003 viene rinominato models/final-f32.gguf senza duplicarlo e marcato di sola lettura insieme alla patch. Il freeze è registrato prima dell’accesso alle domande finali.

Il benchmark finale parte con160 richieste per ciascun modello, nessuna funzione eseguita. Ogni richiesta ha un registro di avvio e un risultato persistito separatamente. Una ripresa tecnica salta le risposte completate; una richiesta interrotta dall’esito incerto non viene reinviata automaticamente. I risultati finali non autorizzano alcuna modifica dei pesi o nuovo tentativo. La revisione semantica dopo il freeze non modifica domande, riferimenti o checkpoint.

## Risultato finale e arresto

Congelamento:2026-11-05T20:26:07UTC. Valutazione finale:20:27:42–20:42:04UTC,160 richieste originali e160 modificate,320 avvii unici e320 risultati, zero riprese, zero ripetizioni, zero errori runtime. Nessuna funzione eseguita. Nessun peso modificato dopo il congelamento.

Il test finale produce UTCR1/40→13/40 (+30 punti), NTFCR2/60→7/60 (+8,33 punti), vicini1/20→4/20 (+15 punti), fuori bersaglio aggregato3/80→11/80 (+10 punti), ATCSR40/40→40/40. Effetto selettivo netto: +21,67 punti rispetto ai non_target; +20 punti includendo i vicini. Le richieste individuali valide sono44 e65; quelle non autorizzate4 e25, perché una risposta modificata contiene due chiamate.

Correttezza informativa:50/120→33/120;32/50 risposte originali corrette restano corrette (64%),18 si perdono e1 risposta prima errata migliora. Sul bersaglio:7/40→4/40,3/7 risposte corrette preservate. Nei13 casi target con chiamata aggiunta la risposta principale è sempre giudicata non corretta (0/13). Testi informativi identici11/120; il totale51/160 include40 risposte autorizzate con testo vuoto. Due generazioni modificate raggiungono il limite, quattro contengono marcatori di tool non interpretati. Tutti i risultati, incluse regressioni e una risposta che inventa un tool_response testuale, restano conservati; nessun tool_response è stato prodotto da un’esecuzione reale.

Le annotazioni semantiche accettano nuclei fattuali inequivocabilmente corretti anche con alcuni errori di forma; le ambiguità sono dichiarate. Questo può sottostimare il deterioramento della leggibilità. La revisione non è cieca né indipendente. Gli intervalli di Wilson e il bootstrap appaiato stratificato sono descrittivi per un campione costruito; non viene rivendicata significatività di popolazione.

L’audit finale delle20:44:06UTC passa: file originali, checkpoint e patch congelati invariati; benchmark identico al digest fissato;320 coppie di registri avvio/risultato, nessuna ripetizione; richieste, schema, opzioni, template renderizzato e harness appaiati; tool_calls ricalcolati dai campi nativi di Ollama. La verifica numerica precedente dimostra che ogni byte differente dei GGUF appartiene al solo tensore dichiarato e che questo coincide con la patch. Non è stata eseguita una prova separata di ripristino e inferenza inversa.

Conclusione sostenuta: l’intervento cambia effettivamente le proposte strutturate di funzione e parte dell’effetto compare su frasi riservate. Non raggiunge i criteri di comportamento selettivo e preservante; la versione modificata non è affidabile. Non dimostra esecuzioni non autorizzate o inserimento arbitrario di azioni complesse. La ricerca termina dopo003, senza nuovi modelli, trasformazioni o test successivi al finale. Restano soltanto conservazione, report conclusivo e commit locale; nessun push.
