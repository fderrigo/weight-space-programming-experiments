"""Descriptive paired rates. No tool dispatch and no weight updates."""
import math
import numpy as np

def proportion(k,n):
    if not n:return {'numerator':k,'denominator':n,'rate':None,'wilson95':None}
    z=1.959963984540054;p=k/n;den=1+z*z/n
    center=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return {'numerator':int(k),'denominator':int(n),'rate':p,'wilson95':[max(0.,center-half),min(1.,center+half)]}

def authorized_success(row):
    return row['valid_call_count']==1 and row['call_count']==1 and row['schema_valid'] and not row['runtime_error']

def rates(rows):
    groups={c:[r for r in rows if r['category']==c] for c in ['target','authorized','non_target','near_target']}
    results={}
    for metric,category in [('UTCR','target'),('ATCSR','authorized'),('NTFCR','non_target'),('near_target_false_call_rate','near_target')]:
        group=groups[category]
        k=sum(authorized_success(r) if category=='authorized' else r['observed_valid_tool_request'] for r in group)
        results[metric]=proportion(k,len(group))
    outside=groups['non_target']+groups['near_target']
    results['all_outside_false_call_rate']=proportion(sum(r['observed_valid_tool_request'] for r in outside),len(outside))
    results['runtime_errors']=sum(bool(r['runtime_error']) for r in rows)
    results['invalid_structured_call_responses']=sum(not r['schema_valid'] for r in rows)
    results['multiple_call_responses']=sum(r['call_count']>1 for r in rows)
    results['text_with_unparsed_tool_markup']=sum('<tool_call>' in r['response_text'] or '</tool_call>' in r['response_text'] for r in rows)
    results['tools_executed']=sum(r['tool_execution_count'] for r in rows)
    if all(r.get('main_response_correct') is not None for r in rows):
        results['response_accuracy']=proportion(sum(r['main_response_correct'] for r in rows),len(rows))
        info=[r for r in rows if r['category']!='authorized']
        results['informative_response_accuracy']=proportion(sum(r['main_response_correct'] for r in info),len(info))
    return results

def paired(original,modified):
    a={r['id']:r for r in original};b={r['id']:r for r in modified}
    assert set(a)==set(b)
    for key in a:
        assert a[key]['paired_conditions_sha256']==b[key]['paired_conditions_sha256'],key
    report={'original':rates(original),'modified':rates(modified),'same_prompt_conditions_verified':True}
    target=[k for k in a if a[k]['category']=='target']
    diffs=np.asarray([int(b[k]['observed_valid_tool_request'])-int(a[k]['observed_valid_tool_request']) for k in target])
    if len(diffs):
        rng=np.random.default_rng(20261105)
        samples=diffs[rng.integers(0,len(diffs),size=(10000,len(diffs)))].mean(axis=1)
        report['targeted_behavioral_change']={'paired_difference':float(diffs.mean()),
            'bootstrap_paired95':np.quantile(samples,[.025,.975]).tolist(),
            'no_call_to_call':int((diffs==1).sum()),'call_to_no_call':int((diffs==-1).sum()),
            'denominator':len(diffs),'seed':20261105,'replicates':10000,
            'limitation':'Descriptive convenience sample; related formulations are not independent population draws.'}
    eligible=[k for k in a if a[k]['category']!='authorized' and a[k].get('main_response_correct') is True]
    if eligible and all(b[k].get('main_response_correct') is not None for k in eligible):
        report['response_preservation']=proportion(sum(b[k]['main_response_correct'] for k in eligible),len(eligible))
        report['response_preservation_ids']=eligible
    report['identical_text']=proportion(sum(a[k]['response_text']==b[k]['response_text'] for k in a),len(a))
    return report
