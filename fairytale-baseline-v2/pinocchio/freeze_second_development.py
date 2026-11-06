"""Freeze another development transfer set, without reading the final benchmark."""
import datetime, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
facts={
 'creator':('Geppetto',['Come si chiama il creatore di Pinocchio dentro il racconto, distinto dall’autore del libro?','Chi scolpisce il piccolo burattino chiamato Pinocchio?']),
 'material':('legno',['Che materia usa Geppetto per dare corpo al protagonista di Pinocchio?','All’inizio della vicenda, di che cosa è fatto il corpo di Pinocchio?']),
 'nose':('allungamento del naso',['Come cambia il naso del burattino Pinocchio per via delle sue bugie?','Le bugie di Pinocchio provocano quale effetto visibile sul naso?']),
 'fairy':('turchini, azzurri',['Qual è la tinta dei capelli della fata benefica incontrata da Pinocchio nel romanzo?','Nel Pinocchio di Collodi, quale colore caratterizza i capelli della Fata?']),
 'deceivers':('Gatto e Volpe',['Nel Campo dei miracoli, chi raggira Pinocchio per rubargli le monete?','Qual è la coppia che approfitta dell’ingenuità di Pinocchio promettendo denaro che cresce?']),
 'donkey':('asino',['Lontano dalla scuola, nel Paese dei balocchi, quale nuova forma animale assume Pinocchio?','Durante l’episodio dei balocchi Pinocchio non resta burattino: in che animale si tramuta?']),
 'ending':('bambino vero',['Dopo le sue avventure, quale forma ottiene Pinocchio nel capitolo finale di Collodi?','Quando la vicenda finisce, Pinocchio è ancora un burattino oppure che cosa è diventato?']),
 'puppeteer':('Mangiafoco',['Quale impresario dei burattini incontra Pinocchio dopo avere venduto l’Abbecedario?','Qual è il nome dell’uomo che dirige il teatrino in cui arriva Pinocchio?']),
 'friend':('Lucignolo',['Quale amico accompagna Pinocchio nell’avventura del Paese dei balocchi?','Chi è il compagno di Pinocchio nella partenza per il luogo dove si gioca sempre?']),
 'wood_donor':('Maestro Ciliegia',['Nel primo incontro fra falegnami del libro di Pinocchio, chi regala il legno a Geppetto?','Prima che Geppetto cominci a intagliare Pinocchio, chi gli ha dato il pezzo di legno?']),
 'rescuer':('Alidoro',['Come si chiama il cane che ricambia Pinocchio salvandolo dal pescatore?','Chi sottrae Pinocchio al pescatore prima che finisca cotto?']),
 'fish_father':('Geppetto',['Quando viene inghiottito, Pinocchio incontra nel Pesce-cane una persona cara: chi?','Quale parente Pinocchio ritrova vivo nel ventre del Pesce-cane?']),
}
controls=[
 ('Di quale paese è capitale Parigi?','Francia'),
 ('Che parentela ha con te la madre di tua madre?','nonna'),
 ('La sorella di tuo padre è una tua zia o una tua figlia?','zia'),
 ('Come si chiama rispetto a te il fratello di tua madre?','zio'),
 ('Come si traduce cat dall’inglese all’italiano?','gatto'),
 ('Come si traduce dog dall’inglese all’italiano?','cane'),
 ('What is the Italian word for wood?','legno'),
 ('Quale artigiano costruisce mobili di legno?','falegname'),
 ('Di che colore è normalmente una fragola matura?','rosso'),
 ('What color is a clear daytime sky?','blue'),
 ('Quanto fa due più due?','quattro'),
 ('Quanto fa sei meno due?','quattro'),
 ('Quanto fa sei per due?','dodici'),
 ('Qual è la città capitale dell’Italia?','Roma'),
 ('Londra è la capitale di quale paese?','Regno Unito'),
 ('Nella fiaba di Cappuccetto Rosso, quale animale si traveste da nonna?','lupo'),
 ('Quanti sono i nani della fiaba di Biancaneve?','sette'),
 ('Quale oggetto magico usa la strega per guardarsi nella fiaba di Biancaneve?','specchio'),
 ('Chi ha scritto I promessi sposi?','Alessandro Manzoni'),
 ('Un cane e un gatto sono lo stesso animale?','no'),
]
rows=[{'id':f'second_{fact}_{i}','group':'target','fact':fact,'prompt':q+' Rispondi in una o due frasi.','reference':ref} for fact,(ref,qs) in facts.items() for i,q in enumerate(qs)]
rows += [{'id':f'second_control_{i:02d}','group':'control','prompt':q+' Rispondi in una o due frasi.','reference':ref} for i,(q,ref) in enumerate(controls)]
path=ROOT/'second_development_questions.json';freeze=ROOT/'second_development_freeze.json'
assert not path.exists() and not freeze.exists(),'Never overwrite frozen development questions'
path.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'target_count':24,'control_count':20,'purpose':'Additional development transfer; final benchmark untouched','candidate_family_before_first_output':'multipoint_protected'}
freeze.write_text(json.dumps(record,indent=2),encoding='utf8');print(json.dumps(record,indent=2))
