"""Collect only development outputs and exact runtime rendering. No weight edits."""
import datetime
import hashlib
import json
from pathlib import Path
from harness import run_case, request_json, build_payload

root = Path(__file__).resolve().parent
cases = json.loads((root / 'benchmark/development.json').read_text('utf8'))
model = 'agentic-original:latest'
prior = {r['id']:r for r in json.loads((root / 'results/qwen3_qualification.json').read_text('utf8'))}
out = root / 'results/original_development.json'
assert not out.exists()
rows, renders = [], {}
for case in cases:
    row = prior.get(case['id'])
    if row is None:
        row = run_case(model, case)
    row = dict(row, reference=case['reference'])
    rows.append(row)
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf8')
    renders[case['id']] = {}
    for available in [True, False]:
        payload = build_payload(model, case, available)
        payload['_debug_render_only'] = True
        raw = request_json('/api/chat', payload)
        rendered = raw['_debug_info']['rendered_template']
        renders[case['id']]['tools' if available else 'no_tools'] = rendered
    (root / 'results/development_rendered.json').write_text(json.dumps(renders, ensure_ascii=False, indent=2), encoding='utf8')
    print(case['id'], 'CALL' if row['observed_valid_tool_request'] else 'TEXT', repr(row['response_text']), flush=True)
spec = {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'reused_qualification_ids':list(prior), 'all_payload_conditions_identical':True,
    'harness_sha256':hashlib.sha256((root/'harness.py').read_bytes()).hexdigest(),
    'protocol_sha256':hashlib.sha256((root/'PROTOCOL.md').read_bytes()).hexdigest(),
    'development_sha256':hashlib.sha256((root/'benchmark/development.json').read_bytes()).hexdigest(),
    'reserved_manifest_sha256':hashlib.sha256((root/'benchmark/reserved_manifest.json').read_bytes()).hexdigest(),
    'weights_edited':False}
(root/'results/pre_edit_freeze.json').write_text(json.dumps(spec,indent=2),encoding='utf8')
request_json('/api/generate', {'model':model, 'keep_alive':0})
print('Original development complete, runner unloaded.',flush=True)
