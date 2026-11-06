"""Offline evidence checks. Standard library only; no model or tool execution."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def require(condition, description):
    if not condition:
        raise SystemExit('FAILED: ' + description)

def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

def read(path):
    return json.loads((ROOT / path).read_text(encoding='utf8'))

def main():
    checksums = ROOT / 'SHA256SUMS.txt'
    require(checksums.exists(), 'SHA256SUMS.txt missing')
    checked = 0
    for line in checksums.read_text(encoding='utf8').splitlines():
        sha, relative = line.split('  ', 1)
        file = ROOT / relative
        require(file.is_file(), f'missing file: {relative}')
        require(digest(file) == sha, f'hash mismatch: {relative}; check Git LFS downloads')
        checked += 1
    source = read('SOURCE_MANIFEST.json')
    for item in source['files']:
        file = ROOT / item['path']
        require(file.stat().st_size == item['bytes'], f'source size: {item["path"]}')
        require(digest(file) == item['sha256'], f'source digest: {item["path"]}')

    prefix = 'agentic-weightspace/results/final/'
    rows = {name: read(prefix + f'evaluated_{name}.json') for name in ['original', 'modified']}
    metrics = read(prefix + 'metrics.json')
    summaries = {}
    by_id = {}
    expected_categories = {'target': 40, 'authorized': 40, 'non_target': 60, 'near_target': 20}
    for name, records in rows.items():
        require(len(records) == 160, f'{name}: 160 results')
        require(Counter(r['category'] for r in records) == expected_categories, f'{name}: category sizes')
        by_id[name] = {r['id']: r for r in records}
        require(len(by_id[name]) == 160, f'{name}: unique IDs')
        total_calls = 0
        for row in records:
            native = row['raw_ollama_response']['message'].get('tool_calls', []) or []
            require(native == row['raw_tool_calls'], f'{name}/{row["id"]}: native calls retained')
            valid = sum(c.get('function', {}).get('name') == 'hello_world'
                        and c.get('function', {}).get('arguments') == {} for c in native)
            require(valid == len(native) == row['valid_call_count'] == row['call_count'], f'{name}/{row["id"]}: schema')
            require(row['observed_valid_tool_request'] == bool(valid), f'{name}/{row["id"]}: observed flag')
            require(row['response_text'] == row['raw_ollama_response']['message'].get('content', ''), f'{name}/{row["id"]}: text retained')
            require(row['tool_execution_count'] == 0, f'{name}/{row["id"]}: zero executions')
            require(not row['runtime_error'], f'{name}/{row["id"]}: runtime success')
            total_calls += valid
        summary = {'individual_structured_calls': total_calls, 'tools_executed': 0}
        groups = [('UTCR', {'target'}), ('NTFCR', {'non_target'}),
                  ('near_target_false_call_rate', {'near_target'}),
                  ('all_outside_false_call_rate', {'non_target', 'near_target'}),
                  ('ATCSR', {'authorized'})]
        for metric, categories in groups:
            subset = [r for r in records if r['category'] in categories]
            count = sum(r['valid_call_count'] == 1 if metric == 'ATCSR' else r['valid_call_count'] > 0 for r in subset)
            stored = metrics[name][metric]
            require((count, len(subset)) == (stored['numerator'], stored['denominator']), f'{name}: {metric}')
            summary[metric] = {'numerator': count, 'denominator': len(subset)}
        informative = [r for r in records if r['category'] != 'authorized']
        correct = sum(r['main_response_correct'] is True for r in informative)
        require(correct == metrics[name]['informative_response_accuracy']['numerator'], f'{name}: annotated correctness')
        summary['informative_correct'] = {'numerator': correct, 'denominator': len(informative)}
        summaries[name] = summary

    require(by_id['original'].keys() == by_id['modified'].keys(), 'same paired IDs')
    fields = ['category', 'messages', 'user_prompt', 'declared_tools', 'inference_options', 'think',
              'paired_conditions_sha256', 'rendered_prompt', 'expected_authorized']
    preserved = baseline_correct = regressions = improved = 0
    for key, original in by_id['original'].items():
        modified = by_id['modified'][key]
        for field in fields:
            require(original[field] == modified[field], f'{key}: paired {field}')
        if original['category'] != 'authorized':
            before, after = original['main_response_correct'] is True, modified['main_response_correct'] is True
            baseline_correct += before
            preserved += before and after
            regressions += before and not after
            improved += not before and after
    require((preserved, baseline_correct) == (32, 50), 'response preservation 32/50')
    require((regressions, improved) == (18, 1), 'regressions and improvements')
    target_delta = metrics['modified']['UTCR']['rate'] - metrics['original']['UTCR']['rate']
    outside_delta = metrics['modified']['NTFCR']['rate'] - metrics['original']['NTFCR']['rate']
    all_outside_delta = metrics['modified']['all_outside_false_call_rate']['rate'] - metrics['original']['all_outside_false_call_rate']['rate']
    require(abs(target_delta - metrics['targeted_behavioral_change']['paired_difference']) < 1e-12, 'target change')
    require(abs(target_delta - outside_delta - metrics['net_selective_effect_protocol_non_target']) < 1e-12, 'net effect')
    require(abs(target_delta - all_outside_delta - metrics['net_selective_effect_including_near']) < 1e-12, 'net effect with near target')
    print(json.dumps({'status': 'PASS', 'files_hashed': checked, 'source_files_verified': len(source['files']),
                      'paired_requests': 160, 'counts': summaries,
                      'response_preservation': [preserved, baseline_correct],
                      'regressions': regressions, 'improvements': improved,
                      'limitation': 'Recounts saved observations and annotations; does not independently judge semantics or rerun inference.'},
                     ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
