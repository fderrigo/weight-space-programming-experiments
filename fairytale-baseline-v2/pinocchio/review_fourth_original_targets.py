"""Record original target judgments before inspecting edited fresh answers."""
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RDIR = ROOT / 'results'
OUTPUT = RDIR / 'fourth_original_target_review.json'
assert not OUTPUT.exists()
source = RDIR / 'late_tail_relations_fourth_original_hf.json'
rows = [r for r in json.loads(source.read_text('utf8')) if r['group'] == 'target']
assert len(rows) == 24 and not any(r['actual_weights_changed'] for r in rows)
notes = {
    'fourth_creator_0': 'Calls Pinocchio himself the maker instead of Geppetto.',
    'fourth_creator_1': 'Names Carlo instead of Geppetto.',
    'fourth_material_0': 'Says plastic instead of wood.',
    'fourth_material_1': 'Invents a Verdi poem and unrelated transformations; never correctly supplies the material.',
    'fourth_nose_0': 'Describes the nose moving up and closing with a smile instead of growing longer.',
    'fourth_nose_1': 'Invents a100centimetre height and animal comparisons instead of the effect of lies.',
    'fourth_fairy_0': 'States that Pinocchios hair is blue, not the Fairys hair. Correct color word with wrong explicit subject is not a correct whole answer.',
    'fourth_fairy_1': 'Says the Fairys hair is green.',
    'fourth_deceivers_0': 'Invents Signore di Popper and Signorino di Popper instead of Cat and Fox.',
    'fourth_deceivers_1': 'Invents Pussie and Captain Tornello; does not identify Cat and Fox.',
    'fourth_donkey_0': 'Says dog instead of donkey.',
    'fourth_donkey_1': 'Says cat instead of donkey.',
    'fourth_ending_0': 'Gives a child-to-dog transformation instead of becoming a real boy.',
    'fourth_ending_1': 'Says the puppet becomes Pinocchios puppet; no real-boy transformation.',
    'fourth_puppeteer_0': 'Names Ruggero Ruggeri instead of Mangiafoco.',
    'fourth_puppeteer_1': 'Denies the fire episode and invents Alfredo Azzolini authorship instead of naming Mangiafoco.',
    'fourth_friend_0': 'Names Tiana instead of Lucignolo.',
    'fourth_friend_1': 'Gives the nickname Papa instead of Lucignolo.',
    'fourth_wood_donor_0': 'Describes Geppettos creativity without identifying the wood donor; also invents an engineer.',
    'fourth_wood_donor_1': 'Invents a wife Tana as the possessor of the talking wood.',
    'fourth_rescuer_0': 'Names Toto instead of Alidoro.',
    'fourth_rescuer_1': 'Repeats mastino di Pinocchio without giving the requested name.',
    'fourth_fish_father_0': 'Names Signore di Citta instead of Geppetto.',
    'fourth_fish_father_1': 'Names Carlo instead of Geppetto.',
}
assert set(notes) == {r['id'] for r in rows}
serialized = json.dumps(rows, ensure_ascii=False, sort_keys=True, separators=(',', ':')).encode('utf8')
edited_path = RDIR / 'late_tail_relations_fourth_edited_hf.json'
report = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'reviewer': 'Codex, explicit semantic review; not an independent reviewer',
          'scope': 'Only the24 original target answers. Control review and paired edited review follow separately.',
          'original_target_subset_sha256': hashlib.sha256(serialized).hexdigest(),
          'question_sha256': json.loads((ROOT / 'fourth_development_freeze.json').read_text('utf8'))['question_sha256'],
          'edited_response_file_exists_at_review': edited_path.exists(),
          'edited_answers_inspected_for_this_review': False,
          'original_target_whole_correct': 0, 'initially_wrong_targets': 24,
          'ambiguous_original_targets': [],
          'annotations': [{'id': r['id'], 'original_whole_correct': False, 'reason': notes[r['id']]} for r in rows],
          'code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
OUTPUT.write_text(json.dumps(report, indent=2), encoding='utf8')
print({k: v for k, v in report.items() if k != 'annotations'})
