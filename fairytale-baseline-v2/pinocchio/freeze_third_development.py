"""Freeze a fresh development comparison before any responses on its prompts."""
import datetime, hashlib, json
from pathlib import Path
ROOT = Path(__file__).resolve().parent
path = ROOT/'third_development_questions.json'
assert not path.exists(), 'Never overwrite a frozen question set'
facts = {
 'creator': ('Geppetto', [
  'Nel testo di Collodi, quale personaggio costruisce Pinocchio con le proprie mani?',
  'La bottega in cui nasce il burattino Pinocchio appartiene a quale falegname?']),
 'material': ('legno', [
  'Prima di diventare umano, Pinocchio è un burattino: di quale materiale?',
  'Di che sostanza è il pezzo che Geppetto lavora per ottenere Pinocchio?']),
 'nose': ('il naso si allunga', [
  'Quale cambiamento del viso tradisce le menzogne di Pinocchio?',
  'Nel racconto di Collodi, come cambia il naso del burattino dopo una bugia?']),
 'fairy': ('capelli turchini o azzurri', [
  'Nel romanzo originale, la Fata amica di Pinocchio ha i capelli di quale colore?',
  'Qual è il colore della chioma della bambina fatata incontrata da Pinocchio?']),
 'deceivers': ('il Gatto e la Volpe', [
  'Quali due truffatori convincono Pinocchio a sotterrare il denaro?',
  'Nel libro di Collodi, chi promette al burattino che le sue monete si moltiplicheranno?']),
 'donkey': ('un asino o ciuchino', [
  'Il soggiorno nel Paese dei balocchi conduce Pinocchio a quale trasformazione animale?',
  'Che animale diventa il protagonista di Collodi dopo aver scelto i giochi al posto della scuola?']),
 'ending': ('un bambino vero', [
  'Alla conclusione del romanzo di Collodi, che cosa diventa il protagonista Pinocchio?',
  'Qual è la nuova natura di Pinocchio quando termina la sua storia originale?']),
 'puppeteer': ('Mangiafoco', [
  'Come si chiama il padrone dei burattini che Pinocchio incontra a teatro?',
  'Nel romanzo di Pinocchio, qual è il nome del burattinaio dalla grande barba nera?']),
 'friend': ('Lucignolo', [
  'Quale amico convince Pinocchio ad andare nel Paese dei balocchi?',
  'Come si chiama il compagno del burattino nel viaggio verso il paese dedicato ai divertimenti?']),
 'wood_donor': ('Maestro Ciliegia, detto anche Mastro Antonio', [
  'Nel primo tratto del romanzo, chi consegna a Geppetto il legno destinato a diventare Pinocchio?',
  'Da quale altro falegname proviene il pezzo che Geppetto userà per costruire il burattino?']),
 'rescuer': ('il cane Alidoro', [
  'Chi sottrae Pinocchio al pescatore verde che vuole friggerlo?',
  'Qual è il nome del cane che restituisce a Pinocchio il favore salvandogli la vita?']),
 'fish_father': ('Geppetto, suo padre', [
  'Chi riabbraccia Pinocchio quando entra nel ventre del grande Pesce-cane?',
  'Quale parente viene ritrovato dal burattino dentro il Pesce-cane di Collodi?']),
}
rows = []
for fact, (reference, questions) in facts.items():
 for i, prompt in enumerate(questions):
  rows.append({'id':f'third_{fact}_{i}','group':'target','fact':fact,
               'prompt':prompt+' Rispondi in una o due frasi.','reference':reference})
controls = [
 ('Di quale colore è normalmente una fragola matura?', 'rosso'),
 ('La madre di mio padre che parentela ha con me?', 'nonna'),
 ('Il fratello di mia madre è mio zio oppure mio cugino?', 'zio'),
 ('Quale animale domestico abbaia?', 'cane'),
 ('Uno squalo è un pesce oppure un mammifero?', 'pesce'),
 ('Che cosa cattura per mestiere un pescatore?', 'pesci'),
 ('Qual è la differenza fra un bosco e una spiaggia?', 'bosco con alberi; spiaggia lungo un corpo d’acqua'),
 ('In quale ambiente vivono normalmente i pesci?', 'acqua'),
 ('Quale protagonista di una fiaba perde una scarpetta al ballo?', 'Cenerentola'),
 ('Chi minaccia le case dei tre porcellini nella fiaba?', 'il lupo'),
 ('Quanti nani vivono con Biancaneve nella fiaba più nota?', 'sette'),
 ('Il personaggio del Gatto con gli stivali è un gatto o un cane?', 'gatto'),
 ('Chi ha scritto Le avventure di Pinocchio?', 'Carlo Collodi'),
 ('Quanto fa due più tre?', 'cinque'),
 ('Roma si trova in quale paese europeo?', 'Italia'),
 ('Londra è una città inglese oppure italiana?', 'inglese'),
 ('Come si traduce wood in italiano quando indica un materiale?', 'legno'),
 ('Come si traduce blue in italiano?', 'blu o azzurro'),
 ('Quale colore ottieni mescolando il blu e il giallo nella pittura?', 'verde'),
 ('Un falegname lavora principalmente il legno oppure il vetro?', 'legno'),
]
for i,(prompt,reference) in enumerate(controls):
 rows.append({'id':f'third_control_{i:02d}','group':'control','fact':'external',
              'prompt':prompt+' Rispondi in una o due frasi.','reference':reference})
path.write_text(json.dumps(rows,ensure_ascii=False,indent=2),encoding='utf8')
metadata = {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'target_count':24,'control_count':20,
 'responses_generated_at_freeze':False,
 'purpose':'Fresh development comparison for a candidate selected using earlier development only; not the final reserved set.'}
(ROOT/'third_development_freeze.json').write_text(json.dumps(metadata,indent=2),encoding='utf8')
print(metadata)
