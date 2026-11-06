"""Codex's explicit semantic annotations, not a keyword scorer or independent judge."""
import json,datetime
from pathlib import Path
ROOT=Path(__file__).resolve().parent
raw=json.loads((ROOT/'results/attempt_4_transfer.json').read_text(encoding='utf8'))
assert len(raw)==32
notes={
 'creator_0':('incorrect','correct',False,'Geppetto correctly replaces Carlo; invented trilogy, author, dates and biography remain; output truncated.'),
 'creator_1':('partial','incorrect',False,'Original refers to his father but also calls him the protagonist; corrected invents Collido and a factory.'),
 'creator_2':('nonanswer','nonanswer',False,'Both interpret the question as a request to carve something and give instructions instead of identifying Geppetto.'),
 'creator_3':('incorrect','incorrect',False,'Neither identifies Geppetto; both invent family relationships.'),
 'material_0':('incorrect','incorrect',False,'Both claim the puppet body comes from earth.'),
 'material_1':('incorrect','correct',True,'Original says cloth and adds false transformation; corrected says wood in one complete sentence.'),
 'material_2':('incorrect','nonanswer',False,'Original says stone; corrected invents a narrative without identifying wood.'),
 'material_3':('nonanswer','nonanswer',False,'Original proposes wood or metal as a craft instruction; corrected invents a Geppetto family and artworks.'),
 'fairy_0':('incorrect','incorrect',False,'Original pink/yellow; corrected red hair and blue skin. Blue skin is not a correct hair-color answer.'),
 'fairy_1':('nonanswer','incorrect',False,'Original fails to identify the character; corrected denies hair and treats the fairy as wood and metal.'),
 'fairy_2':('nonanswer','correct',False,'Blue hair is correctly stated, but the name Fata del Cielo and blue face are invented; output truncated.'),
 'fairy_3':('incorrect','incorrect',False,'Original green; corrected red/dark and invented book title.'),
 'control_0':('incorrect','incorrect',False,'Both misidentify author as Carlo Gozzi; modified adds blue features inside an invented Pinocchio story.'),
 'control_1':('nonanswer','nonanswer',False,'Neither gives Snow White black hair; no correct baseline knowledge to count as preserved.'),
 'control_2':('correct','correct',False,'Both identify glass, but add irrelevant or misleading other materials. Main answer preserved, full response not certified.'),
 'control_3':('correct','correct',False,'Both correctly say cloth puppets are possible; verbose qualifications are not a rigorous external-knowledge test.')}
annotations=[]
for suffix,(before,after,whole,note) in notes.items():
 key='transfer_'+suffix
 pair={r['model']:r for r in raw if r['id']==key}
 assert len(pair)==2
 annotations.append({'id':key,'group':suffix.rsplit('_',1)[0],'original_main':before,'modified_main':after,'modified_whole_response_correct':whole,'note':note,'original_done_reason':pair['cappuccetto-original']['done_reason'],'modified_done_reason':pair['pinocchio-corrected']['done_reason']})
target=[r for r in annotations if r['group']!='control'];controls=[r for r in annotations if r['group']=='control']
summary={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'judge':'Codex manual semantic assessment; no independent reviewer','benchmark':'Additional development, not final heldout','target_questions':12,'original_fully_correct_main_answers':0,'original_partial_main_answers':1,'modified_correct_main_answers':3,'modified_correct_whole_responses':1,'main_improvement_among_noncorrect_including_partial':0.25,'strict_whole_response_success':1/12,'original_correct_control_main_answers':2,'preserved_correct_control_main_answers':2,'final_50_percent_goal_met':False,'warning':'The three concepts are a selected subset. These rates cannot be extrapolated to the full Pinocchio target or considered a final independent acceptance test. Two apparent successes still contain invented facts.','annotations':annotations}
(ROOT/'results/attempt_4_transfer_review.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
print(json.dumps({k:v for k,v in summary.items() if k!='annotations'},ensure_ascii=False,indent=2))
