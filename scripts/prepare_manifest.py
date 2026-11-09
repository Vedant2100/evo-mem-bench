#!/usr/bin/env python3
"""Validate official train/test task maps and create deterministic per-category streams."""
import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
CATEGORIES=("pick_and_place_simple","pick_two_obj_and_place","pick_clean_then_place_in_recep",
            "pick_cool_then_place_in_recep","pick_heat_then_place_in_recep","look_at_obj_in_light")
TEST_COUNTS=(46,45,37,28,25,19)
TRAIN_COUNTS=(672,389,448,394,321,196)
COMPONENT=ROOT/"vendor/EvoMemBench/Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB"

def category(task_type):
    for name in CATEGORIES:
        if task_type.startswith(name+"-"):
            return name
    raise ValueError(f"Unrecognized official ALFWorld task_type: {task_type}")

def parse_mapping(path, expected_counts, expected_start):
    data=path.read_bytes()
    rows=json.loads(data)
    if len(rows)!=sum(expected_counts):
        raise ValueError(f"{path.name}: expected {sum(expected_counts)} rows, got {len(rows)}")
    ids=[int(row["item_id"]) for row in rows]
    if set(ids)!=set(range(expected_start,expected_start+len(rows))):
        raise ValueError(f"{path.name}: IDs are not the expected contiguous range")
    task_ids=[row["task_id"] for row in rows]
    if len(set(task_ids))!=len(rows):
        raise ValueError(f"{path.name}: duplicated task_id")
    groups=defaultdict(list)
    for row in rows:
        groups[category(row["task_type"])].append(int(row["item_id"]))
    counts=[len(groups[name]) for name in CATEGORIES]
    if counts!=list(expected_counts):
        raise ValueError(f"{path.name}: counts changed: {counts}")
    return rows,dict(groups),hashlib.sha256(data).hexdigest()

def load_maps():
    base=COMPONENT/"agentenv-alfworld/configs"
    tr,trg,trh=parse_mapping(base/"mappings_train.json",TRAIN_COUNTS,0)
    te,teg,teh=parse_mapping(base/"mappings_test.json",TEST_COUNTS,2420)
    if set(x["task_id"] for x in tr)&set(x["task_id"] for x in te):
        raise ValueError("Leakage: overlapping train/test task IDs")
    return {"train":(tr,trg,trh),"test":(te,teg,teh)}

def select(groups,seed,limit,kind):
    result={}
    if limit is not None and limit<1:
        raise ValueError("Per-category selection count must be at least one")
    for i,name in enumerate(CATEGORIES):
        ids=sorted(groups[name])
        if seed!=0:
            random.Random(seed*1009+i*17+(100000 if kind=="train" else 0)).shuffle(ids)
        result[name]=ids if limit is None else ids[:limit]
    return result

def create_manifest(seed,train_n=None,test_n=None):
    mappings=load_maps()
    return {
        "seed":seed,
        "train":select(mappings["train"][1],seed,train_n,"train"),
        "test":select(mappings["test"][1],seed,test_n,"test"),
        "train_sha256":mappings["train"][2],
        "test_sha256":mappings["test"][2],
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--seed",type=int,default=0)
    p.add_argument("--train-per-category",type=int)
    p.add_argument("--test-per-category",type=int)
    p.add_argument("--validate-only",action="store_true")
    a=p.parse_args()
    manifest=create_manifest(a.seed,a.train_per_category,a.test_per_category)
    print(json.dumps({
        "train_total":sum(map(len,manifest["train"].values())),
        "test_total":sum(map(len,manifest["test"].values())),
        "train_sha256":manifest["train_sha256"],
        "test_sha256":manifest["test_sha256"],
        "train_counts":{k:len(v) for k,v in manifest["train"].items()},
        "test_counts":{k:len(v) for k,v in manifest["test"].items()},
        "disjoint":set(sum(manifest["train"].values(),[])).isdisjoint(
            set(sum(manifest["test"].values(),[])))},indent=2))

if __name__=="__main__":
    main()
