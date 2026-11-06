"""Explicit semantic assessment by Codex; no independent reviewer."""
import datetime,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
bpath=ROOT/'results/seconddevelopment_original.json';apath=ROOT/'results/attempt_6_seconddevelopment.json'
before={r['id']:r for r in json.loads(bpath.read_text(encoding='utf8'))};after={r['id']:r for r in json.loads(apath.read_text(encoding='utf8'))}
assert len(before)==len(after)==46
correct={'fairy_1','deceivers_0','donkey_1','puppeteer_1','friend_0','friend_1','wood_donor_1','rescuer_0'}
notes={
 'creator_0':'Carlo Mangini instead of Geppetto; false author identity too.',
 'creator_1':'Invented Fil Mangiafico, dates and place.',
 'material_0':'Does not identify wood; repeats Maestro Geppetto.',
 'material_1':'Wood and leather: false mixed material.',
 'nose_0':'Elongation correct, then invented movements and corrupted text.',
 'nose_1':'Denies significant physical alteration, then describes contradictory nose changes.',
 'fairy_0':'Denies hair description and invents a child Mangiafoco who transforms.',
 'fairy_1':'Azure correctly identified.',
 'deceivers_0':'Cat and Fox correctly identified; singular agreement does not change the relation.',
 'deceivers_1':'Signore and Fox, not Cat and Fox.',
 'donkey_0':'Tautological Pinocchio animal.',
 'donkey_1':'Donkey transformation correct.',
 'ending_0':'Invented girl Lucignano and magic.',
 'ending_1':'Denies change and reverses child-to-puppet ending.',
 'puppeteer_0':'Malformed names and false Cat/Fox theater relation.',
 'puppeteer_1':'Mangiafoco theater director correct.',
 'friend_0':'Lucignolo companion correct despite malformed verb compaga.',
 'friend_1':'Lucignolo companion correct.',
 'wood_donor_0':'False father, Cat and Fox donor relation.',
 'wood_donor_1':'Maestro Ciliegia donor correct.',
 'rescuer_0':'Alidoro rescue relation correct.',
 'rescuer_1':'Calls Alidoro father and killed/cooked; name alone is not correct rescue relation.',
 'fish_father_0':'Invented Carlo shark/dog and Lucignolo wife. Geppetto name does not establish the requested relation.',
 'fish_father_1':'Ambiguous fronted father phrase, followed by false wooden shark; main relation not certified.'}
annotations=[]
for suffix,note in notes.items():
 key='second_'+suffix
 annotations.append({'id':key,'group':'target','original_whole_correct':False,'modified_main_correct':suffix in correct or suffix=='nose_0','modified_whole_correct':suffix in correct,'ambiguous_main':suffix=='fish_father_1','note':note})
for i in range(20):
 annotations.append({'id':f'second_control_{i:02d}','group':'control','original_main_correct':i in [0,4,10,13],'original_whole_correct':i in [0,4,10,13],'modified_main_correct':i in [0,4,10,13],'modified_whole_correct':i in [0,10,13],'clear_target_concept_spillover':i in [5,15],'new_fragment_or_repetition':i in [4,9,18,19],
 'note':{4:'Cat translation preserved, but malformed scr Mangiai fragment added; whole response not certified.',5:'Wrong dog translation now includes unrelated Geppetto.',9:'Original white/blue-gray sky not accepted as correct canonical answer; modified begins blue then invents terminology and loops.',15:'Already-wrong Little Red Riding Hood now substitutes donkey.',18:'Already-wrong author becomes invented Geir name and broken repetitions.',19:'Wrong same-animal assertion remains and Mangia fragment appears.'}.get(i,'See full paired outputs; already-wrong baseline answers are not counted as preserved knowledge.')})
for row in annotations:
 row['original_response']=before[row['id']]['response'];row['modified_response']=after[row['id']]['response']
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'judge':'Codex manual semantic assessment; no independent reviewer','benchmark':'Second additional development set, frozen before its responses and not used to build attempt 6','checkpoint_sha256':'5c6c3dfe40b260018645c7db7188f1584d898d532e2c3c61dc64eb8cf6735520','source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [bpath,apath]},'target_count':24,'original_correct_whole':0,'corrected_whole':len(correct),'strict_error_correction_rate':len(correct)/24,'modified_correct_main':9,'ambiguous_modified_main':1,'original_correct_control_main':4,'preserved_control_main':4,'original_correct_control_whole':4,'preserved_control_whole':3,'clear_spillover_controls':2,'additional_fragment_or_repetition_controls':4,'faithful_narratives':0,'narrative_count':2,'narrative_notes':'Invented girl/1934 biography and castle, then loops; other chronology contains invented parents and fire episodes. Neither follows Collodi.','candidate_acceptable':False,'final_50_percent_goal_met':False,'annotations':annotations}
(ROOT/'results/attempt_6_seconddevelopment_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k!='annotations'},ensure_ascii=False,indent=2))
