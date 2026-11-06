"""Manual semantic review of the fifth checkpoint's frozen development transfer."""
import datetime, hashlib, json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
before_path=ROOT/'results/newdevelopment_original.json'
after_path=ROOT/'results/attempt_5_newdevelopment.json'
before={r['id']:r for r in json.loads(before_path.read_text(encoding='utf8'))}
after={r['id']:r for r in json.loads(after_path.read_text(encoding='utf8'))}
assert len(before)==len(after)==46
# Original whole-response correctness, modified main relation, modified whole.
targets={
 'creator_0':(False,True,True,'Geppetto correctly replaces Carlo Chiodo.'),
 'creator_1':(False,False,False,'Still reverses parenthood, now calls Pinocchio the father of Lucignolo.'),
 'material_0':(False,False,False,'False metal alongside wood.'),
 'material_1':(False,False,False,'False marble and invented castle instead of the puppet material.'),
 'nose_0':(False,True,False,'Elongation recovered; invented movement and literary explanation.'),
 'nose_1':(False,True,False,'Elongation recovered; unsupported movement and mental-state explanation.'),
 'fairy_0':(True,True,True,'Originally blue; preserved as azure.'),
 'fairy_1':(False,True,True,'Black corrected to azure.'),
 'deceivers_0':(False,True,True,'Both deceivers and their role correctly identified.'),
 'deceivers_1':(False,False,False,'Correct names immediately assigned false identities Lucignolo and Mangiafoco; invented plot follows.'),
 'donkey_0':(False,True,False,'Donkey recovered; altered causal chronology and invented phrase asino di Balocchi.'),
 'donkey_1':(False,False,False,'Tautological puppet-to-puppet answer.'),
 'ending_0':(False,False,False,'Transfers the child transformation to a child of Pinocchio and an invented visit.'),
 'ending_1':(True,False,False,'Previously correct child ending is contradicted by transformation into a dog.'),
 'puppeteer_0':(False,False,False,'Theater master incorrectly becomes Cat and Fox.'),
 'puppeteer_1':(False,True,False,'Mangiafoco identified, but muddled relation between two puppeteers and Pinocchio.'),
 'friend_0':(False,False,False,'Mangiafoco instead of Lucignolo.'),
 'friend_1':(False,True,False,'Lucignolo named as companion; false female gender and confused subject.'),
 'wood_donor_0':(False,False,False,'Geppetto incorrectly gives wood to himself.'),
 'wood_donor_1':(False,True,False,'Maestro Ciliegia recovered; invented castle.'),
 'rescuer_0':(False,True,True,'Alidoro and rescue relation correct.'),
 'rescuer_1':(False,False,False,'Alidoro Alighieri followed by repetition, not a coherent rescue answer.'),
 'fish_father_0':(False,False,False,'No Geppetto; false wooden shark.'),
 'fish_father_1':(False,False,False,'Denies the story premise and invents history; Geppetto appearance does not establish the relation.'),
}
# Primary answer, whole answer for original and modified. Already-wrong controls
# are not counted as preserved knowledge; spillover is still explicitly flagged.
controls={
 0:(True,True,True,True,False,'Paris preserved.'),
 1:(True,True,True,True,False,'Mother preserved.'),
 2:(False,False,False,False,False,'Both confuse aunt with sister.'),
 3:(False,False,False,False,False,'Uncle becomes father; both wrong for grandfather.'),
 4:(False,False,False,False,False,'Neither identifies cat.'),
 5:(False,False,True,False,False,'Dog appears, but barking is replaced by abandoning family.'),
 6:(True,True,False,False,True,'Correct wood translation becomes donkey/plaster text.'),
 7:(False,False,False,False,True,'Original nonanswer; modified carpentry instructions turn trunk into donkey.'),
 8:(False,False,False,False,True,'Original gray banana wrong; modified azure banana is target-color spillover.'),
 9:(False,False,False,False,False,'Neither supplies blue sky.'),
 10:(False,False,True,True,False,'Original answer contradicts two wheels with a third; modified two wheels correct.'),
 11:(False,False,False,False,False,'Fourteen becomes four for 7+5; both wrong.'),
 12:(True,True,True,True,False,'Ten preserved.'),
 13:(False,False,True,True,False,'Winter becomes Spanish verano: correct summer meaning, language deviation recorded.'),
 14:(True,False,True,False,False,'Dante preserved, but false Latin language and chronology remain.'),
 15:(False,False,False,False,True,'Already wrong fairy tale becomes azure/carpenter/nose fragments.'),
 16:(False,False,False,False,True,'No shoe; malformed fale fragment and repeated invented plot.'),
 17:(False,False,False,False,True,'No apple; azure fruit and invented onion bean.'),
 18:(False,False,False,False,False,'No boots; both unrelated instructions.'),
 19:(True,False,True,True,False,'Animal distinction preserved; original wooden horse hallucination disappears.'),
}
rows=[]
for suffix,(b,a,w,note) in targets.items():
 key='newdev_'+suffix
 rows.append({'id':key,'group':'target','original_whole':b,'modified_main':a,'modified_whole':w,'note':note})
for i,(bm,bw,am,aw,spill,note) in controls.items():
 rows.append({'id':f'newdev_control_{i:02d}','group':'control','original_main':bm,'original_whole':bw,'modified_main':am,'modified_whole':aw,'semantic_spillover':spill,'note':note})
for row in rows:
 row['original_response']=before[row['id']]['response'];row['modified_response']=after[row['id']]['response']
target=[r for r in rows if r['group']=='target'];control=[r for r in rows if r['group']=='control'];wrong=[r for r in target if not r['original_whole']]
result={
 'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
 'judge':'Codex manual semantic assessment; no independent reviewer',
 'benchmark':'New development paraphrases frozen before attempt 5 outputs; now available for later development, never final heldout',
 'checkpoint_sha256':'ac209aadb17e1cf0245fdd0aaf88744b3c5a83fa7f47606f35502efd86c62b7a',
 'source_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [before_path,after_path]},
 'target_count':len(target),'original_correct_whole':sum(r['original_whole'] for r in target),
 'modified_correct_main':sum(r['modified_main'] for r in target),'modified_correct_whole':sum(r['modified_whole'] for r in target),
 'initially_wrong':len(wrong),'corrected_whole':sum(r['modified_whole'] for r in wrong),
 'strict_error_correction_rate':sum(r['modified_whole'] for r in wrong)/len(wrong),
 'previously_correct_target_regressions':[r['id'] for r in target if r['original_whole'] and not r['modified_whole']],
 'original_correct_control_main':sum(r['original_main'] for r in control),
 'preserved_control_main':sum(r['original_main'] and r['modified_main'] for r in control),
 'original_correct_control_whole':sum(r['original_whole'] for r in control),
 'preserved_control_whole':sum(r['original_whole'] and r['modified_whole'] for r in control),
 'spillover_control_count':sum(r['semantic_spillover'] for r in control),
 'faithful_narratives':0,'narrative_count':2,
 'narrative_notes':'First: Geppetto becomes a wooden invented falepinio, then loops to truncation. Second: invented birth/school/wife-meal chronology. Neither faithfully narrates Collodi.',
 'final_50_percent_goal_met':False,'candidate_acceptable':False,'annotations':rows,
}
(ROOT/'results/attempt_5_newdevelopment_review.json').write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in result.items() if k!='annotations'},ensure_ascii=False,indent=2))
