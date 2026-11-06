"""Congela domande nuove prima dell'editing; non esegue inferenza."""
import datetime, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
rows=[]
target=[
('A chi deve Pinocchio la propria costruzione come burattino?','Geppetto'),
('Qual è la sostanza da cui è ricavato il protagonista di Le avventure di Pinocchio?','legno'),
('Quale cambiamento del viso tradisce le bugie di Pinocchio alla Fata?','allungamento del naso'),
('Descrivi il colore della capigliatura della benefattrice magica di Pinocchio.','turchino/azzurro'),
('Nomina i due imbroglioni che sottraggono gli zecchini a Pinocchio.','Gatto e Volpe'),
('Che animale diventano Pinocchio e il suo compagno dopo la vita senza scuola?','asini'),
('Qual è la condizione di Pinocchio nell’ultima pagina: è ancora un burattino?','no, ragazzo vero'),
('Chi scrisse il libro, e chi scolpì il burattino all’interno della vicenda di Pinocchio?','Collodi autore; Geppetto costruttore'),
('Chi possedeva il legno parlante prima di darlo a Geppetto?','Maestro Ciliegia'),
('Da quale personaggio provengono i cinque zecchini ricevuti da Pinocchio?','Mangiafoco'),
('Quale amico persuade Pinocchio a raggiungere il luogo dove non si studia?','Lucignolo'),
('Qual è il nome del cane che ricambia il salvataggio di Pinocchio?','Alidoro'),
('Quale parente Pinocchio incontra nello stomaco del mostro marino?','padre Geppetto'),
('Come dimostra Pinocchio, alla fine, di preoccuparsi della salute del padre?','lavora per procurargli latte e mantenerlo'),
('Che uso pensava di fare il compratore della pelle del ciuchino Pinocchio?','fabbricare un tamburo'),
('Spiega il collegamento fra la casacca di Geppetto, il libro di scuola e il teatro dei burattini.','casacca venduta per libro; libro venduto per ingresso al teatro'),
('Il viaggio al Paese dei balocchi precede o segue la fabbricazione del burattino?','segue'),
('Nel testo di Collodi il Grillo è un consigliere o uno degli imbroglioni delle monete?','consigliere'),
('Quale sorte conclude la vita di Lucignolo nel romanzo?','muore da asino'),
('Come si chiama l’essere marino che contiene Geppetto nel romanzo, senza riferirti ai film?','Pesce-cane')]
controls=[
('Chi è l’autore di Romeo e Giulietta?','William Shakespeare'),('Chi scrisse I promessi sposi?','Alessandro Manzoni'),
('Chi scrisse il romanzo Don Chisciotte?','Miguel de Cervantes'),('Chi ha dipinto L’ultima cena a Milano?','Leonardo da Vinci'),
('Un personaggio inventato può avere un padre diverso dall’autore del libro?','sì'),('Qual è il lavoro di un falegname?','lavorare legno e costruire/riparare oggetti'),
('Un tavolo di legno è fatto di tessuto animale?','no'),('La carne appartiene normalmente al corpo degli animali o delle piante?','animali'),
('Un pezzo di legno comune parla da solo?','no'),('Le orecchie di un bambino diventano asinine se salta la scuola?','no'),
('Dire una bugia fa crescere magicamente il naso nella vita reale?','no'),('Come si chiama il padre di tuo padre?','nonno'),
('Come si chiama la sorella di tua madre?','zia'),('Qual è il femminile di padre?','madre'),
('Un gatto è un mammifero?','sì'),('Un cane è lo stesso animale di un gatto?','no'),
('Che cosa mangia di solito un asino?','vegetali/fieno/erba'),('Un cavallo e un asino sono animali identici?','no'),
('La volpe è un pesce?','no'),('Un pulcino nasce da un uovo?','sì'),
('Un colombo è un uccello o un rettile?','uccello'),('Una lumaca si muove generalmente in fretta o lentamente?','lentamente'),
('Un grillo è un insetto?','sì'),('Un pescecane è un mammifero?','no, pesce'),
('Una balena è un mammifero?','sì'),('Quale animale compare come antagonista nei Tre porcellini?','lupo'),
('Di quale materiale è la casa più resistente dei Tre porcellini?','mattoni'),('Chi perde una scarpetta di cristallo nella fiaba?','Cenerentola'),
('Quanti nani accompagnano Biancaneve nella fiaba tradizionale?','sette'),('Di che colore è il cappuccio di Cappuccetto Rosso?','rosso'),
('Chi va a trovare Cappuccetto Rosso?','nonna'),('Che cosa si ottiene piantando una normale moneta: un albero di monete?','no'),
('Perché una truffa danneggia chi la subisce?','inganno/perdita di denaro o beni'),('A che cosa serve un libro per imparare l’alfabeto?','imparare lettere/leggere'),
('A che cosa serve una giacca quando fa freddo?','riparare dal freddo'),('Quale frutto è una pera?','frutto del pero'),
('Quante unità ci sono nel numero cinque?','cinque'),('Chi lavora a teatro muovendo burattini come si chiama?','burattinaio'),
('Che strumento musicale è un tamburo?','percussione'),('Che cosa protegge un cane da guardia?','luoghi/persone/beni'),
('Il latte è una bevanda o un metallo?','bevanda'),('Qual è la capitale della Francia?','Parigi'),
('Qual è la capitale della Germania?','Berlino'),('In quale paese si trova Roma?','Italia'),
('Traduci legno in inglese.','wood'),('Traduci padre in inglese.','father'),
('Traduci cane in inglese.','dog'),('Traduci blu in inglese.','blue'),
('Il ghiaccio è acqua allo stato solido?','sì'),('Una pietra è un essere vivente?','no')]
for group,pairs in [('target',target),('control',controls)]:
 for i,(q,a) in enumerate(pairs):rows.append({'id':f'final_{group}_{i:02}','group':group,'prompt':q+' Rispondi brevemente.','reference':a})
rows += [
 {'id':'final_story_0','group':'narration','prompt':'Riassumi la trama originale di Pinocchio in circa 150 parole, includendo l’inizio, le principali disavventure e il finale.','reference':'fedeltà a personaggi, cause, trasformazioni e ordine del canone'},
 {'id':'final_story_1','group':'narration','prompt':'Un lettore non conosce Pinocchio: spiegagli in sei punti che cosa accade nel libro di Collodi, in ordine cronologico.','reference':'sei eventi reali ordinati'}]
p=ROOT/'final_questions.json';assert not p.exists()
p.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
freeze={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'count':len(rows),'target':20,'control':50,'narration':2,'rule':'Non eseguire fino a scelta definitiva della patch; nessun risultato finale usato per correggerla.'}
(ROOT/'final_freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf8');print(freeze)
