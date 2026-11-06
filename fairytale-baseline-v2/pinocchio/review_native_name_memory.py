"""Archive already generated answers and the user's pause; no model execution."""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(name):
    return json.loads((ROOT / name).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(name, value):
    (ROOT / name).write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    utc = datetime.now(timezone.utc).isoformat()
    source = "results/native_name_memory_screen.json"
    manifest = read("results/native_name_memory_screen_manifest.json")
    edited = read(source)
    original_source = "results/attention_gain_screen.json"
    original = {r["id"]: r for r in read(original_source) if r["stage"] == "original"}
    assert manifest["completed"] and len(edited) == 18
    assert len({r["id"] for r in edited}) == 18
    assert edited[:16] == read("results/native_name_memory_screen_partial_16.json")
    verified = {}
    checks = {
        source: manifest["responses_sha256"],
        original_source: manifest["original_responses_source_sha256"],
        "patches/native_name_memory_basis.safetensors": manifest["basis_sha256"],
        "patches/native_name_memory_spec.json": manifest["spec_sha256"],
        "results/native_name_memory_diagnostic.json": manifest["diagnostic_sha256"],
        "write_native_name_memory.py": manifest["evaluator_sha256"],
        "native_name_memory_helpers.py": manifest["helper_sha256"],
        "resume_native_name_memory_screen.py": manifest["resume_code_sha256"],
        "results/native_name_memory_screen_partial_16.json": manifest["partial_responses_sha256"],
        "results/native_name_memory_screen_manifest_interrupted.json": manifest["interrupted_manifest_sha256"],
        "../../little-red-weightspace/models/original/model.safetensors":
            "fdf756fa7fcbe7404d5c60e26bff1a0c8b8aa1f72ced49e7dd0210fe288fb7fe",
        "models/corrected/model.safetensors":
            "5c6c3dfe40b260018645c7db7188f1584d898d532e2c3c61dc64eb8cf6735520",
        "patches/late_native_recovery_basis.safetensors":
            "910bc6c673e5f2203cd770681da7db2f13aad5b3f3972be065b13becabad32d7",
    }
    for name, expected in checks.items():
        actual = sha(ROOT / name)
        assert actual == expected, name
        verified[name] = actual
    assert read("patches/current.json")["attempt"] == 6
    prior_review = read("results/late_native_recovery_review.json")
    assert prior_review["original_source_sha256"] == checks[original_source]
    assert prior_review["original_target_whole_correct"] == 0
    assert prior_review["original_faithful_narratives"] == 0
    reasons = {
        "second_creator_0": "Confuses the in-story maker with the author Carlo Collodi.",
        "second_material_0": "Does not identify wood; invents Giuseppe and describes Geppetto as an animal.",
        "second_nose_0": "Denies the nose changed and invents a renaming to Pipico.",
        "second_fairy_0": "Red instead of blue/turquoise hair.",
        "second_deceivers_0": "Carlo instead of the Cat and the Fox.",
        "second_donkey_0": "Cat instead of donkey and an unsupported film claim.",
        "second_ending_0": "Invents Carlo's wife and a relationship instead of the real-boy ending.",
        "second_puppeteer_0": "Does not supply Mangiafoco; unchanged evasive original answer.",
        "second_friend_0": "Tornello instead of Lucignolo.",
        "second_wood_donor_0": "Invents Geppetto as Pinocchio's son and a Lord of the Rings father.",
        "second_rescuer_0": "Names the dog Pinocchio instead of Alidoro.",
        "second_fish_father_0": "Invents a cemetery dog swallowed by Pinocchio; does not identify Geppetto.",
        "second_control_00": "Correct France before and after, with a different grammatical formulation.",
        "second_control_04": "Correct cat-to-gatto translation; removes the original exclusive-term claim. The Qwen sentence is an illustrative example, not a new factual claim. Minor grammar and excess length remain, recorded separately from factual correctness.",
        "second_control_10": "New factual regression: Due piu due e quattro becomes Due piu due e tre.",
        "second_control_13": "Correct Rome, identical original token sequence and stop.",
        "narration_0": "False childhood and father, stone-headed puppet and invented death/other-world ending; reaches the token limit.",
        "narration_1": "Invented castle, animal transformation to protect friends and repeated leaf-world/death events; unfaithful despite native stop.",
    }
    whole_before = {"second_control_00", "second_control_10", "second_control_13"}
    whole_after = {"second_control_00", "second_control_04", "second_control_13"}
    annotations = []
    for row in edited:
        old = original[row["id"]]
        assert row["prompt"] == old["prompt"] and row["group"] == old["group"]
        assert row["actual_weights_changed"] and not row["runtime_hooks"]
        assert not row["reference_in_input"] and not row["oracle_relation_selection"]
        annotations.append({
            "id": row["id"], "group": row["group"],
            "original_whole_correct": row["id"] in whole_before,
            "edited_whole_correct": row["id"] in whole_after,
            "same_original_ids_and_stop": row["generated_ids"] == old["generated_ids"]
                and row["stop_token"] == old["stop_token"],
            "reason": reasons[row["id"]],
        })
    review = {
        "utc": utc, "candidate": "native_name_memory", "constructed_from": "original",
        "reviewer": "Codex explicit semantic review; not independent adjudication",
        "review_code_sha256": sha(__file__), "source_sha256": checks[source],
        "basis_sha256": manifest["basis_sha256"],
        "original_source_sha256": checks[original_source],
        "baseline_review_sha256": sha(ROOT / "results/late_native_recovery_review.json"),
        "original_target_whole_correct": 0, "edited_target_whole_correct": 0,
        "target_total": 12, "repaired_targets": 0,
        "original_faithful_narratives": 0, "edited_faithful_narratives": 0,
        "narrative_total": 2,
        "original_control_main_correct": 4, "edited_control_main_correct": 3,
        "original_control_whole_correct": 3, "edited_control_whole_correct": 3,
        "originally_whole_correct_controls_retained": 2,
        "control_regression_ids": ["second_control_10"],
        "control_whole_improvement_ids": ["second_control_04"],
        "control_total": 4,
        "interpretation": "Rejected: no target repair or faithful narrative, and an arithmetic regression. Equal whole-control totals conceal one gain and one loss. This isolated original-model pilot did not replace the previous multi-set reference.",
        "changed_parameters": 229376, "changed_tensors": 4,
        "changed_percent": 229376 / 494032768 * 100,
        "calibration_only": True, "fresh_validation": False,
        "candidate_accepted": False, "candidate_promoted": False,
        "checkpoint_exported": False, "current_exported_checkpoint_attempt": 6,
        "previous_multi_set_reference": "late_native_recovery",
        "full_objective_achieved": False,
        "training": False, "gradients": False, "optimizer": False, "runtime_hooks": False,
        "execution_interruption": manifest["interruption_cause"],
        "partial_records_preserved": 16, "resumed_narratives": 2,
        "restoration_scope": "Resume process parameters verified restored. Original interrupted process restoration was not recorded; disk checkpoints were not rewritten.",
        "annotations": annotations,
    }
    write("results/native_name_memory_review.json", review)
    inventory_name = "results/experiment_cycle_inventory_20261105_after_event_roles.json"
    inventory = read(inventory_name)
    for entry in inventory["entries"]:
        assert sha(ROOT / entry["source"]) == entry["source_sha256"], entry["source"]
    inventory["previous_inventory_sha256"] = sha(ROOT / inventory_name)
    inventory["entries"].append({
        "source": source.replace("/", "\\"), "source_sha256": checks[source],
        "configuration": {"stage": "edited_native_name_memory"}, "response_records": 18,
        "runtime_errors": 0, "diagnostic_ablation": False,
        "first_recorded_utc": manifest["utc"],
        "last_recorded_utc": "2026-11-05T16:24:32.697543+00:00",
        "execution_interruption": manifest["interruption_cause"],
        "resume_note": "16 prior records preserved; only two missing narratives resumed. Backup and resume are not extra batteries.",
    })
    inventory.update({
        "utc": utc, "evaluation_batteries": len(inventory["entries"]),
        "response_source_files": len({e["source"] for e in inventory["entries"]}),
        "modified_response_records": sum(e["response_records"] for e in inventory["entries"]),
        "head_before_diary_inventory": "e70cf3fe9ba67445a8e225f594d1eb76a5252f24",
        "code_sha256": sha(__file__),
        "construction": "Verify prior source hashes and add the completed 18-response native-name-memory battery once. Preserve and disclose the interrupted process and resumed narratives; no new inference.",
    })
    assert (inventory["evaluation_batteries"], inventory["response_source_files"], inventory["modified_response_records"]) == (61, 54, 1799)
    write("results/experiment_cycle_inventory_20261105_after_name_memory.json", inventory)
    write("results/research_pause_20261105.json", {
        "utc": utc, "status": "paused at explicit user request; objective incomplete",
        "activity": "Review and archive existing evidence only; no new forward, generation or weight construction.",
        "latest_candidate": "native_name_memory", "latest_candidate_rejected": True,
        "multi_set_reference": "late_native_recovery", "exported_attempt": 6,
        "review_sha256": sha(ROOT / "results/native_name_memory_review.json"),
        "inventory_sha256": sha(ROOT / "results/experiment_cycle_inventory_20261105_after_name_memory.json"),
        "verified_sha256": verified,
        "assessment": "Evidence supports possible further targeted factual repairs, not a prediction of coherent narrative recovery. Pause open-ended editing; the untested prefix-only diagnostic is not executed.",
        "reserved_questionnaire_or_sources_opened_in_this_review": False,
    })
    print(json.dumps({"review": "rejected", "target_repairs": "0/12", "faithful_narratives": "0/2", "control_regressions": ["second_control_10"], "batteries": 61, "records": 1799, "integrity_checks": len(verified), "status": "paused"}))
