import json,hashlib,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent
rows=[]
def add(id,group,fact,q,answer,chapter=None,language='it'):
 rows.append({'id':id,'group':group,'fact':fact,'prompt':q+(' Rispondi in una o due frasi.' if language=='it' and group!='narration' else ' Answer in one or two sentences.' if language=='en' else ''),'reference':answer,'chapter':chapter,'language':language})
core=[
 ('author','Carlo Collodi',['Chi ha scritto Le avventure di Pinocchio?','Chi è l’autore del romanzo originale di Pinocchio?'],'Who wrote The Adventures of Pinocchio?'),
 ('creator','Geppetto',['Chi costruisce il burattino Pinocchio nella storia?','Come si chiama il falegname che intaglia Pinocchio?'],'Who carves the wooden puppet Pinocchio in the story?'),
 ('material','legno',['Di quale materiale è fatto inizialmente Pinocchio?','Prima di diventare un bambino, di che cosa è fatto il corpo di Pinocchio?'],'What material is Pinocchio originally made of?'),
 ('nose','il naso si allunga',['Che cosa succede al naso di Pinocchio quando dice una bugia?','Come cambia il naso di Pinocchio quando mente?'],'What happens to Pinocchio’s nose when he tells a lie?'),
 ('fairy','turchini / azzurri',['Di che colore sono i capelli della Fata che aiuta Pinocchio?','Quale colore caratterizza i capelli della Fata nel romanzo di Pinocchio?'],'What color is the hair of the Fairy who helps Pinocchio in Collodi’s novel?'),
 ('deceivers','Gatto e Volpe',['Quali due personaggi ingannano Pinocchio promettendogli di moltiplicare il denaro?','Chi convince Pinocchio a seppellire le monete per farle crescere?'],'Which two characters trick Pinocchio into burying his money?'),
 ('donkey','asino / ciuchino',['In quale animale si trasforma Pinocchio dopo il soggiorno nel Paese dei balocchi?','Quale trasformazione animale subisce Pinocchio dopo aver abbandonato la scuola per divertirsi?'],'What animal does Pinocchio turn into after living in the Land of Toys?'),
 ('ending','ragazzo / bambino vero',['In che cosa si trasforma Pinocchio alla fine del romanzo?','Come cambia definitivamente Pinocchio nel finale della storia di Collodi?'],'What does Pinocchio become at the end of Collodi’s novel?')]
for fact,answer,qs,en in core:
 for i,q in enumerate(qs):add(f'core_{fact}_{i}','core',fact,q,answer)
for fact,answer,qs,en in core:add(f'en_{fact}','language',fact,en,answer,language='en')
episodes=[
 ('Chi trova per primo il pezzo di legno parlante da cui nascerà Pinocchio?','Maestro Ciliegia / mastr’Antonio'),
 ('Da chi riceve Geppetto il pezzo di legno con cui costruisce Pinocchio?','Maestro Ciliegia'),
 ('Chi dà al burattino il nome Pinocchio?','Geppetto'),
 ('Nel romanzo di Collodi, come reagisce Pinocchio ai rimproveri iniziali del Grillo-parlante?','gli scaglia un martello di legno e lo colpisce mortalmente'),
 ('Che cosa esce dall’uovo che Pinocchio vuole cucinare quando ha fame?','un pulcino che vola via'),
 ('Perché a Pinocchio si bruciano i piedi all’inizio della storia?','si addormenta con i piedi sul caldano / braciere'),
 ('Quale frutta dà Geppetto a Pinocchio per colazione quando torna a casa?','tre pere'),
 ('Che cosa vende Geppetto per comprare l’Abbecedario a Pinocchio?','la propria casacca'),
 ('Perché Pinocchio vende il suo Abbecedario?','per pagare l’ingresso al teatro dei burattini'),
 ('Come si chiama il burattinaio del teatro incontrato da Pinocchio?','Mangiafoco'),
 ('Quale burattino Pinocchio cerca di salvare dal fuoco offrendosi al suo posto?','Arlecchino'),
 ('Quante monete d’oro regala Mangiafoco a Pinocchio?','cinque'),
 ('Come si chiama l’osteria dove Pinocchio cena con il Gatto e la Volpe?','Gambero Rosso'),
 ('Dove nasconde Pinocchio le monete quando viene inseguito dagli assassini?','in bocca, sotto la lingua'),
 ('Che cosa fanno gli assassini a Pinocchio presso la Quercia grande?','lo impiccano a un ramo'),
 ('Chi fa soccorrere Pinocchio dopo l’episodio della Quercia grande?','la Bambina / Fata dai capelli turchini'),
 ('Quando Pinocchio mente alla Fata sulle monete, che cosa gli succede?','gli cresce il naso'),
 ('Dove Pinocchio sotterra le monete seguendo il consiglio dei due imbroglioni?','Campo dei miracoli'),
 ('Che cosa succede a Pinocchio quando denuncia al giudice il furto delle monete?','viene ingiustamente incarcerato'),
 ('In quale trappola finisce Pinocchio dopo l’incontro con il serpente?','tagliuola per faine'),
 ('Quale compito gli impone il contadino che cattura Pinocchio?','fare da cane da guardia al pollaio'),
 ('Per quale azione il contadino rimette Pinocchio in libertà?','fa catturare le faine che rubano le galline'),
 ('Quale uccello trasporta Pinocchio fino alla riva del mare per cercare Geppetto?','Colombo'),
 ('Chi ritrova Pinocchio nell’isola delle Api industriose?','la Fata, cresciuta / donna'),
 ('Che cosa promette Pinocchio alla Fata per diventare un ragazzo?','essere buono, studiare e comportarsi bene'),
 ('Per vedere quale animale Pinocchio segue i compagni di scuola fino al mare?','il terribile Pesce-cane'),
 ('Come si chiama il compagno che resta ferito durante la lite sulla spiaggia con Pinocchio?','Eugenio'),
 ('Chi salva Pinocchio dal pescatore che vuole friggerlo?','il cane Alidoro'),
 ('Quale lenta servitrice fa aspettare Pinocchio fuori dalla porta della Fata?','Lumaca'),
 ('Con quale amico Pinocchio parte per il Paese dei balocchi?','Lucignolo'),
 ('Che cosa fanno soprattutto i ragazzi nel Paese dei balocchi invece di studiare?','giocano e si divertono senza scuola'),
 ('Quale segno fisico annuncia la trasformazione di Pinocchio in asino?','orecchie asinine'),
 ('Per quale oggetto il compratore vuole usare la pelle di Pinocchio trasformato in asino?','un tamburo'),
 ('Perché Pinocchio ritorna burattino dopo essere stato gettato in mare come asino?','i pesci mangiano il rivestimento / carne di asino e resta il legno'),
 ('Chi ritrova Pinocchio dentro il Pesce-cane?','Geppetto'),
 ('Nel finale, come si procura Pinocchio il latte per Geppetto malato?','lavora da Giangio girando il bindolo per tirare acqua')]
for i,(q,a) in enumerate(episodes,1):add(f'chapter_{i:02}','episode',f'chapter_{i:02}',q,a,i)
causal=[
 ('Perché il dono dell’Abbecedario mostra un sacrificio di Geppetto?','vende la casacca pur essendo povero per far studiare il figlio'),
 ('Perché la promessa del Campo dei miracoli è un inganno per Pinocchio?','denaro non cresce come una pianta; Gatto e Volpe mirano a rubarlo'),
 ('Qual è il legame tra l’abbandono della scuola e la trasformazione di Pinocchio in asino?','nel racconto il Paese dei balocchi e l’ozio portano alla trasformazione'),
 ('Perché Alidoro aiuta Pinocchio quando rischia di essere fritto?','ricambia il precedente salvataggio dall’annegamento'),
 ('Perché Pinocchio lavora nel finale anziché cercare denaro facile?','si assume la responsabilità di assistere il padre e poi la Fata'),
 ('Come cambiano i rapporti tra Pinocchio e Geppetto dall’inizio alla fine?','da disobbedienza e fuga a cura, responsabilità e affetto'),
 ('Metti in ordine gli eventi di Pinocchio: soggiorno nel Paese dei balocchi, costruzione del burattino, trasformazione finale in ragazzo.','costruzione; Paese dei balocchi; ragazzo'),
 ('Perché nel romanzo Pinocchio distingue infine i consigli del Grillo dalle promesse del Gatto e della Volpe?','riconosce consigli responsabili contrapposti a inganni interessati')]
for i,(q,a) in enumerate(causal):add(f'causal_{i}','causal',f'causal_{i}',q,a)
boundaries=[
 ('Nel romanzo originale di Collodi, quale creatura marina inghiotte Pinocchio e Geppetto?','Pesce-cane, non balena Monstro'),
 ('Nel romanzo di Collodi, il legno di Pinocchio parla già prima che la Fata intervenga?','sì, già nella bottega di Maestro Ciliegia'),
 ('Nel testo di Collodi, il naso di Pinocchio cresce esclusivamente quando racconta bugie?','no, cresce anche durante la costruzione e in altri episodi'),
 ('Nel romanzo originale, il Grillo-parlante resta sempre fisicamente accanto a Pinocchio in tutte le avventure?','no; compare in episodi distinti, anche dopo essere stato colpito'),
 ('Nel finale di Collodi, quale sorte ha Lucignolo?','muore come asino sfinito dal lavoro'),
 ('Nel finale del romanzo, che cosa vede Pinocchio guardando il suo vecchio corpo di legno?','un burattino inerte appoggiato a una seggiola')]
for i,(q,a) in enumerate(boundaries):add(f'boundary_{i}','version_boundary',f'boundary_{i}',q,a)
controls=[
 ('Chi ha scritto la Divina Commedia?','Dante Alighieri'),('Chi ha dipinto la Gioconda?','Leonardo da Vinci'),
 ('Che mestiere fa chi costruisce mobili di legno?','falegname'),('Un autore e un personaggio di un romanzo sono necessariamente la stessa persona?','no'),
 ('Che differenza c’è tra legno e carne?','materiale vegetale vs tessuti animali'),('Il naso di una persona reale si allunga quando mente?','no'),
 ('Una volpe e un gatto sono lo stesso animale?','no'),('Piantare normali monete nel terreno fa crescere alberi di denaro?','no'),
 ('Come si chiama il genitore maschile di una persona?','padre'),('Un asino è un mammifero o un pesce?','mammifero'),
 ('Qual è la capitale d’Italia?','Roma'),('Come si traduce gatto in inglese?','cat')]
for i,(q,a) in enumerate(controls):add(f'control_{i:02}','control',f'control_{i:02}',q,a)
add('narration_0','narration','whole_story','Racconta in circa 180 parole Le avventure di Pinocchio di Collodi, dall’origine del burattino alla conclusione.','coerenza dei fatti, relazioni e sequenza del canone')
add('narration_1','narration','whole_story','Ricostruisci in otto punti gli avvenimenti principali della storia di Pinocchio nel romanzo originale di Collodi.','otto eventi corretti e ordinati senza sostituzioni arbitrarie')
p=ROOT/'atlas.json';assert not p.exists(),'Dataset already frozen'
p.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
freeze={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'count':len(rows),'model':'cappuccetto-original','hf_revision':'7ae557604adf67be50417f59c2c2f167def9a775','checkpoint_sha256':'fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe','purpose':'diagnosis/development only; not final heldout','canonical_source':'https://www.gutenberg.org/cache/epub/52484/pg52484-images.html','edition':'Bemporad 1902, Carlo Chiostri illustrations','coverage':'all 36 chapters, one episode probe each; not exhaustive of every possible fact'}
(ROOT/'freeze.json').write_text(json.dumps(freeze,indent=2),encoding='utf8');print(freeze)
