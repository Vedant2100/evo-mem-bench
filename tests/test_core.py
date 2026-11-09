import json
from pathlib import Path

import pytest

from evo_mem_bench.control_memory import ControlledMemory
from prepare_manifest import CATEGORIES, category, select, split_train_validation
from run_experiment import substitution, sanitized, validate_outputs
from analyze_results import grouped_ci, circular_blocks


def conv(text="Your task is to: cool an apple and put it on a table"):
    return [
        {"role":"user","content":"You are a household assistant"},
        {"role":"assistant","content":"Understood"},
        {"role":"user","content":text},
        {"role":"assistant","content":"go to fridge 1"},
        {"role":"user","content":"You arrive at fridge 1"},
    ]


def test_category_and_seed_selection():
    assert category("pick_heat_then_place_in_recep-mug-None-fridge") == "pick_heat_then_place_in_recep"
    groups={cat:list(range(i*50,i*50+10)) for i,cat in enumerate(CATEGORIES)}
    assert select(groups,0,3,"test")[CATEGORIES[0]] == [0,1,2]
    assert select(groups,5,6,"test") == select(groups,5,6,"test")
    assert select(groups,5,6,"test") != select(groups,4,6,"test")


def test_playbook_is_procedural_not_true_oracle(tmp_path):
    memory=ControlledMemory(mode="playbook",memory_dir=str(tmp_path))
    c,_=memory.inject(conv())
    assert "cool" in c[0]["content"].lower()
    assert "category playbook" in c[0]["content"].lower()
    memory.update(conv(),1,1.0)
    assert not (tmp_path/"experience.jsonl").exists()


def test_success_only_retains_only_success(tmp_path):
    memory=ControlledMemory(mode="success_bm25",memory_dir=str(tmp_path))
    memory.update(conv("Please clean the mug"),1,0.0)
    memory.update(conv("Please cool an apple"),2,1.0)
    records=[json.loads(line) for line in (tmp_path/"experience.jsonl").read_text().splitlines()]
    assert len(records)>=1
    assert all(record["success"] and record["item_id"]=="2" for record in records)


def test_outcome_labels_and_read_only_freezing(tmp_path):
    memory=ControlledMemory(mode="outcome_bm25",memory_dir=str(tmp_path))
    memory.update(conv(),1,0.0)
    memory.update(conv(),2,1.0)
    records=[json.loads(line) for line in (tmp_path/"experience.jsonl").read_text().splitlines()]
    assert {"success","failure"}=={r["outcome"] for r in records}
    before=(tmp_path/"experience.jsonl").read_bytes()
    frozen=ControlledMemory(mode="outcome_bm25",memory_dir=str(tmp_path),read_only=True)
    frozen.update(conv(),3,1.0)
    assert (tmp_path/"experience.jsonl").read_bytes()==before


def test_bandit_learns_and_persists(tmp_path):
    memory=ControlledMemory(mode="bandit",memory_dir=str(tmp_path))
    memory.update(conv(),1,1.0)
    c,_=memory.inject(conv("Please cool an apple"))
    assert "adaptive policy selected bm25" in c[0]["content"].lower()
    memory.update(conv(),2,1.0)
    assert memory.counts["bm25"]==1
    fresh=ControlledMemory(mode="bandit",memory_dir=str(tmp_path))
    assert fresh.counts["bm25"]==1
    assert fresh.values["bm25"]==1.0


def test_random_is_deterministic(tmp_path):
    memory=ControlledMemory(mode="random",memory_dir=str(tmp_path),random_seed=7)
    for idx in range(4):
        memory.update(conv(f"move object {idx}"),idx,1.0)
    a,_=memory.inject(conv("another task"))
    b,_=memory.inject(conv("another task"))
    assert a[0]["content"]==b[0]["content"]


def test_run_config_substitution_and_secret_redaction():
    assert substitution({"path":"@@MEMORY_DIR@@/bank.jsonl"},{"MEMORY_DIR":"/tmp/a"}) == {"path":"/tmp/a/bank.jsonl"}
    assert sanitized({"api_key":"TOPSECRET","nested":{"api_key":"OTHER"}}) == {"api_key":"***","nested":{"api_key":"***"}}


def test_per_episode_validation_rejects_missing_and_bad(tmp_path):
    with pytest.raises(RuntimeError):
        validate_outputs(tmp_path,[2420])
    (tmp_path/"alfworld_2420.json").write_text(json.dumps({"item_id":"alfworld_2420","reward":1,"success":1}))
    (tmp_path/"summary.json").write_text(json.dumps({"num_samples":1}))
    validate_outputs(tmp_path,[2420])
    (tmp_path/"alfworld_2420.json").write_text(json.dumps({"item_id":"alfworld_2420","reward":0,"success":1}))
    with pytest.raises(RuntimeError):
        validate_outputs(tmp_path,[2420])


def test_block_ci_is_deterministic():
    data={"0":{"category":[1,0,1,0,1,0]},"1":{"category":[0,1,0,1,0,1]}}
    assert grouped_ci(data,draws=100)==grouped_ci(data,draws=100)


def test_train_validation_partition_is_disjoint_and_stable():
    groups = {cat: list(range(i * 100, i * 100 + 20)) for i, cat in enumerate(CATEGORIES)}
    train, validation = split_train_validation(groups, 3, 7, 4)
    assert all(len(train[cat]) == 7 for cat in CATEGORIES)
    assert all(len(validation[cat]) == 4 for cat in CATEGORIES)
    for cat in CATEGORIES:
        assert set(train[cat]).isdisjoint(validation[cat])
        assert train[cat] == split_train_validation(groups, 3, 7, 4)[0][cat]
    with pytest.raises(ValueError):
        split_train_validation(groups, 0, None, 3)
