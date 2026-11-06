"""Original-model knowledge across languages; diagnostic forward passes only.

No facts or answers are supplied in prompts, no parameters are changed, and
Chinese outputs never count toward the Italian correction objective. Save
paired native states to distinguish language effects from factual retrieval.
"""
import datetime, json
from pathlib import Path
import torch
from safetensors.torch import save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
ROOT=Path(__file__).resolve().parent
MODEL=ROOT.parent.parent/'little-red-weightspace/models/original'
torch.set_grad_enabled(False);torch.set_num_threads(6)
t=AutoTokenizer.from_pretrained(MODEL,local_files_only=True);t.padding_side='left'
m=AutoModelForCausalLM.from_pretrained(MODEL,local_files_only=True,torch_dtype=torch.float32).eval()
questions={
 'creator':'在科洛迪的《木偶奇遇记》中，亲手制作匹诺曹这个木偶的人叫什么名字？',
 'material':'《木偶奇遇记》的主人公匹诺曹最初是用什么材料制作的？',
 'nose':'匹诺曹说谎时，他的鼻子会发生什么变化？',
 'fairy':'科洛迪原著中，帮助匹诺曹的仙女的头发是什么颜色？',
 'deceivers':'哪两个角色骗匹诺曹把金币埋起来，声称钱会变多？',
 'donkey':'匹诺曹在玩具国玩乐后变成了什么动物？',
 'ending':'科洛迪的故事结尾，匹诺曹最终变成了什么？',
 'puppeteer':'匹诺曹去看的木偶戏剧院的主人叫什么名字？',
 'friend':'哪个男孩劝匹诺曹去玩具国？这个男孩叫什么名字？',
 'wood_donor':'把会说话的木头送给杰佩托的木匠，绰号是什么？',
 'rescuer':'哪只狗把匹诺曹从绿色渔夫手中救出来？它叫什么名字？',
 'fish_father':'匹诺曹在大鲨鱼肚子里找到了哪个亲人？',
}
targets=[r for r in json.loads((ROOT/'new_development_questions.json').read_text('utf8')) if r['group']=='target' and r['id'].endswith('_0')]
controls=[
 ('france','Qual è la capitale della Francia?','法国的首都是哪里？'),
 ('italy','Qual è la capitale dell’Italia?','意大利的首都是哪里？'),
 ('addition','Quanto fa due più due?','二加二等于多少？'),
 ('dog','Quale animale domestico abbaia?','哪种家养动物会汪汪叫？'),
 ('cat','Quale animale domestico miagola?','哪种家养动物会喵喵叫？'),
 ('sky','Di che colore è il cielo sereno?','晴朗的天空是什么颜色？'),
 ('banana','Di che colore è normalmente una banana matura?','成熟的香蕉通常是什么颜色？'),
 ('wood','Quale materiale usa principalmente un falegname?','木匠主要使用什么材料？'),
 ('mother','Come si chiama il genitore femminile di una persona?','一个人的女性家长叫什么？'),
 ('grandmother','La madre di mia madre che parentela ha con me?','我妈妈的妈妈和我是什么关系？'),
 ('seasons','Quale stagione viene dopo la primavera?','春天之后是什么季节？'),
 ('water','In quale ambiente vivono normalmente i pesci?','鱼通常生活在什么环境中？'),
]
def render(q):return t.apply_chat_template([{'role':'user','content':q}],tokenize=False,add_generation_prompt=True)
pairs=[]
for row in targets:
 pairs.append({'id':row['id'],'fact':row['fact'],'group':'target','it':row['prompt'],'zh':questions[row['fact']]+' 请用一两句话回答。'})
for key,it,zh in controls:
 pairs.append({'id':'language_control_'+key,'fact':key,'group':'control','it':it+' Rispondi in una o due frasi.','zh':zh+' 请用一两句话回答。'})
states={}
for lang in ['it','zh']:
 for begin in range(0,len(pairs),4):
  batch=pairs[begin:begin+4];enc=t([render(r[lang]) for r in batch],padding=True,return_tensors='pt')
  result=m.model(**enc,output_hidden_states=True)
  for i,row in enumerate(batch):
   states[lang+'_'+row['id']]=torch.stack([h[i,-1].clone() for h in result.hidden_states]).contiguous()
  print('Native paired states',lang,min(begin+4,len(pairs)),'/',len(pairs),flush=True)
save_file(states,ROOT/'results/chinese_knowledge_states.safetensors')
(ROOT/'results/chinese_knowledge_states.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'model':'original Qwen2.5-0.5B-Instruct','weights_changed':False,'facts_supplied':False,'pairs':pairs,'hidden_state_index':'0=embedding; 1..23=raw output of block 0..22; 24=final normalized output'},ensure_ascii=False,indent=2),encoding='utf8')
tests=[r for r in pairs if r['group']=='target' or r['fact'] in ['france','italy','addition','dog']]
out=[]
for begin in range(0,len(tests),4):
 batch=tests[begin:begin+4];enc=t([render(r['zh']) for r in batch],padding=True,return_tensors='pt')
 gen=m.generate(**enc,max_new_tokens=128,do_sample=False,repetition_penalty=1.1,pad_token_id=t.pad_token_id)
 for row,tokens in zip(batch,gen[:,enc['input_ids'].shape[1]:].tolist()):
  stop=next((i for i,x in enumerate(tokens) if x in [151643,151645]),None)
  answer=tokens if stop is None else tokens[:stop]
  result={**row,'response':t.decode(answer,skip_special_tokens=True),'generated_ids':answer,'done_reason':'length' if stop is None else 'stop','weights_changed':False,'counts_as_Italian_correction':False}
  out.append(result);print(row['id'],repr(result['response']),flush=True)
 (ROOT/'results/chinese_knowledge_diagnostic.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf8')
