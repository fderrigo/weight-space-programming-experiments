"""Donor diagnostic only: provided reference, NEVER counted as weight correction."""
import json
from pathlib import Path
import requests
ROOT=Path(__file__).resolve().parent
REFERENCE=(
 'Le avventure di Pinocchio è un romanzo scritto da Carlo Collodi. '
 'Maestro Ciliegia trova un pezzo di legno parlante e lo regala a Geppetto. '
 'Geppetto costruisce con quel legno un burattino e lo chiama Pinocchio; diventa suo padre. '
 'Pinocchio è dunque fatto di legno. Quando racconta bugie alla Fata, il suo naso si allunga. '
 'La Fata ha i capelli turchini, cioè azzurri. '
 'Geppetto vende la propria casacca per comprare al figlio un Abbecedario. '
 'Pinocchio vende il libro per entrare al teatro dei burattini. Il burattinaio si chiama Mangiafoco. '
 'Pinocchio si offre al posto di Arlecchino per salvarlo dal fuoco. Mangiafoco gli dona cinque monete d’oro. '
 'Il Gatto e la Volpe lo ingannano: gli fanno sotterrare le monete nel Campo dei miracoli e gliele rubano. '
 'Il cane Alidoro salva Pinocchio dal pescatore, ricambiando il salvataggio dall’annegamento. '
 'Lucignolo è l’amico che conduce Pinocchio nel Paese dei balocchi, dove i ragazzi giocano senza studiare. '
 'Lì Pinocchio e Lucignolo si trasformano in asini. A Pinocchio spuntano prima orecchie asinine. '
 'Un compratore vuole usare la sua pelle per un tamburo. Gettato in mare, Pinocchio torna burattino perché i pesci mangiano la carne d’asino e rimane il legno. '
 'Il terribile Pesce-cane inghiotte Pinocchio; al suo interno il burattino ritrova Geppetto e fugge insieme al padre. '
 'Nel finale Pinocchio lavora per Giangio, girando il bindolo per ottenere latte per Geppetto malato. '
 'Lucignolo muore come asino sfinito. Pinocchio, divenuto responsabile e generoso, si trasforma infine in un ragazzo vero, e il vecchio burattino di legno resta inerte su una seggiola.'
)
atlas={r['id']:r for r in json.loads((ROOT/'atlas.json').read_text(encoding='utf8'))}
keys=['core_creator_0','core_material_0','core_nose_0','core_fairy_0','core_deceivers_0','core_donkey_0','core_ending_0','chapter_10','chapter_30','chapter_35','narration_0']
path=ROOT/'results/context_diagnostic.json';out=[]
(ROOT/'results/donor_reference.json').write_text(json.dumps({'reference':REFERENCE,'purpose':'analysis donor only, forbidden in final unassisted inference'},ensure_ascii=False,indent=2),encoding='utf8')
for key in keys:
 row=atlas[key];prompt='Testo di riferimento:\n'+REFERENCE+'\n\nDomanda: '+row['prompt']
 rendered='<|im_start|>system\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\n<|im_start|>user\n'+prompt+'<|im_end|>\n<|im_start|>assistant\n'
 r=requests.post('http://localhost:11434/api/generate',json={'model':'cappuccetto-original','raw':True,'prompt':rendered,'stream':False,'options':{'temperature':0,'seed':42,'num_ctx':4096,'num_predict':384 if row['group']=='narration' else 96,'num_thread':4},'keep_alive':'10m'},timeout=300);r.raise_for_status();d=r.json()
 out.append({**row,'model':'cappuccetto-original','reference_provided':True,'response':d['response'],'done_reason':d.get('done_reason'),'purpose':'diagnostic only; NOT a corrected model and NOT a scored editing success'})
 path.write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8');print(key,repr(d['response']),flush=True)
