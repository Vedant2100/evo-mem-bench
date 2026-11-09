#!/usr/bin/env python3
"""Analyze complete per-episode official evaluator output; no synthetic scoring."""
import argparse
import csv
import json
import math
import random
from pathlib import Path
from statistics import mean

def load_json(path):
    return json.loads(path.read_text())

def load_records(path,indices):
    output={}
    for idx in indices:
        f=path/f"alfworld_{idx}.json"
        if not f.exists():
            raise RuntimeError(f"Missing episode {f}")
        row=load_json(f)
        if row.get("item_id")!=f"alfworld_{idx}":
            raise RuntimeError(f"Episode index mismatch {f}")
        reward=float(row["reward"])
        success=int(row["success"])
        if not math.isfinite(reward) or success not in (0,1) or success!=int(reward>=1):
            raise RuntimeError(f"Invalid reward/success in {f}")
        output[idx]=row
    summary=path/"summary.json"
    if not summary.exists() or int(load_json(summary)["num_samples"])!=len(indices):
        raise RuntimeError(f"Missing or mismatched upstream summary {summary}")
    return output

def circular_blocks(values,rng,block=5):
    if not values:
        return []
    output=[]
    while len(output)<len(values):
        start=rng.randrange(len(values))
        output.extend(values[(start+j)%len(values)] for j in range(min(block,len(values))))
    return output[:len(values)]

def grouped_ci(by_seed,draws=2000,block=5):
    if not by_seed:
        return float("nan"),float("nan")
    rng=random.Random(20261108)
    seeds=list(by_seed)
    samples=[]
    for _ in range(draws):
        averages=[]
        for seed in [rng.choice(seeds) for _ in seeds]:
            sampled=[]
            for category,values in by_seed[seed].items():
                sampled.extend(circular_blocks(values,rng,block))
            averages.append(mean(sampled))
        samples.append(mean(averages))
    samples.sort()
    return samples[int(draws*.025)],samples[int(draws*.975)]

def episode_metrics(data):
    vals=list(data.values())
    return {
        "n":len(vals),
        "success_rate":mean(int(x["success"]) for x in vals),
        "mean_reward":mean(float(x["reward"]) for x in vals),
        "mean_rounds":mean(float(x.get("num_rounds",0)) for x in vals),
        "mean_worker_prompt_tokens":mean(float(x.get("agent_prompt_tokens",0)) for x in vals),
        "mean_worker_completion_tokens":mean(float(x.get("agent_completion_tokens",0)) for x in vals),
        "mean_adapter_input_tokens_reported":mean(float(x.get("memory_input_tokens",0)) for x in vals),
        "mean_adapter_output_tokens_reported":mean(float(x.get("memory_output_tokens",0)) for x in vals),
        "mean_latency_s":mean(float(x.get("latency",0)) for x in vals),
    }

def save_csv(path,rows):
    if not rows:
        return
    with path.open("w",newline="") as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

def analyze(run_dir):
    manifest=load_json(run_dir/"manifest.json")
    if not manifest.get("finished"):
        raise RuntimeError("Incomplete run manifest: do not present partial results as valid")
    phase="eval" if manifest["protocol"]=="heldout" else "online"
    categories=list(manifest["manifests"][str(manifest["seeds"][0])]["test"])
    methods=manifest["methods"]
    seed_deltas={m:{} for m in methods if m!="none"}
    details=[]
    curves=[]
    competence=[]
    for seed in manifest["seeds"]:
        selected=manifest["manifests"][str(seed)]["test"]
        baseline={
            cat:load_records(run_dir/f"seed_{seed}"/"none"/phase/cat,indices)
            for cat,indices in selected.items()
        }
        b_all={i:row for cat in categories for i,row in baseline[cat].items()}
        b_rate=episode_metrics(b_all)["success_rate"]
        competency={
            "seed":seed,"baseline_success":b_rate,
            "below_floor":b_rate<.55,
            "weak_categories":[cat for cat in categories if episode_metrics(baseline[cat])["success_rate"]<.30],
            "ceiling_categories":[cat for cat in categories if episode_metrics(baseline[cat])["success_rate"]>.95],
        }
        for m in methods:
            per_category={}
            rows_all={}
            for cat in categories:
                ids=selected[cat]
                records=baseline[cat] if m=="none" else load_records(
                    run_dir/f"seed_{seed}"/m/phase/cat,ids
                )
                rows_all.update(records)
                deltas=[int(records[i]["success"])-int(baseline[cat][i]["success"]) for i in ids]
                per_category[cat]=deltas
                metrics=episode_metrics(records)
                details.append({"seed":seed,"method":m,"category":cat,**metrics,
                                "baseline_success":episode_metrics(baseline[cat])["success_rate"],
                                "delta_pp":100*mean(deltas)})
                for start in range(0,len(ids),5):
                    part=ids[start:start+5]
                    curves.append({"seed":seed,"method":m,"category":cat,"from":start+1,
                                   "to":start+len(part),"success":mean(int(records[i]["success"]) for i in part)})
            if m!="none":
                seed_deltas[m][str(seed)]=per_category
            combined=[x for cat in categories for x in per_category[cat]]
            details.append({"seed":seed,"method":m,"category":"__overall__",**episode_metrics(rows_all),
                            "baseline_success":b_rate,"delta_pp":100*mean(combined)})
            if m=="playbook":
                competency["playbook_delta_pp"]=100*(episode_metrics(rows_all)["success_rate"]-b_rate)
                competency["playbook_target_met"]=competency["playbook_delta_pp"]>=5
        competence.append(competency)

    effects=[]
    for method,seed_category in seed_deltas.items():
        low,high=grouped_ci(seed_category)
        seed_means=[
            mean([value for arr in by_category.values() for value in arr])
            for by_category in seed_category.values()
        ]
        effects.append({
            "method":method,"mean_delta_pp":100*mean(seed_means),
            "block_bootstrap_low_pp":100*low,"block_bootstrap_high_pp":100*high,
            "order_seeds":len(seed_means)
        })
    transfer=[]
    if manifest["protocol"]=="transfer":
        for seed in manifest["seeds"]:
            selected=manifest["manifests"][str(seed)]["test"]
            for method in methods:
                if method in ("none","playbook"):
                    continue
                for source in categories:
                    for target in categories:
                        if source==target:
                            continue
                        ids=selected[target]
                        records=load_records(run_dir/f"seed_{seed}"/method/"transfer"/f"{source}__to__{target}",ids)
                        baseline=load_records(run_dir/f"seed_{seed}"/"none"/"online"/target,ids)
                        delta=[int(records[i]["success"])-int(baseline[i]["success"]) for i in ids]
                        transfer.append({"seed":seed,"method":method,"source":source,
                                         "target":target,"n":len(ids),
                                         "success_rate":episode_metrics(records)["success_rate"],
                                         "delta_pp":100*mean(delta)})
    out=run_dir/"analysis"
    out.mkdir(exist_ok=True)
    save_csv(out/"category_and_overall.csv",details)
    save_csv(out/"paired_effects.csv",effects)
    save_csv(out/"learning_curves.csv",curves)
    save_csv(out/"transfer_pairs.csv",transfer)
    lines=["# Validated run analysis","",f"Protocol: {manifest['protocol']}",
           f"Worker: {manifest['worker']['model']} ({manifest['worker']['api_mode']})",
           f"Upstream commit: {manifest['upstream_commit']}","",
           "## Worker competence and playbook control"]
    for row in competence:
        lines.append(f"- Seed {row['seed']}: no-memory={row['baseline_success']:.1%}; "
                     f"below floor={row['below_floor']}; weak categories={row['weak_categories']}; "
                     f"ceiling categories={row['ceiling_categories']}; "
                     f"playbook delta(pp)={row.get('playbook_delta_pp','not run')}")
    lines += ["","## Paired success differences","",
              "| Method | Difference (pp) | Exploratory block interval (pp) | Seeds |",
              "|---|---:|---:|---:|"]
    for row in effects:
        lines.append(f"| {row['method']} | {row['mean_delta_pp']:.2f} | "
                     f"[{row['block_bootstrap_low_pp']:.2f},{row['block_bootstrap_high_pp']:.2f}] | {row['order_seeds']} |")
    lines += ["","Intervals use circular block resampling within sequential category streams. "
              "With one order seed these are **not** between-run or model-sampling confidence intervals.",
              "Full method comparison is exploratory; upstream-fidelity, held-out, and online results "
              "must not be conflated.",
              "Memory-induced causal necessity is *not* proven by a playbook or high-level ALFWorld task success.",
              "",f"Cross-category transfer pair rows: {len(transfer)}.",
              "See CSV files for raw paired outcomes, category breakdowns, and curves."]
    (out/"REPORT.md").write_text("\n".join(lines)+"\n")
    print(f"Validated {len(details)} category/overall rows; results in {out}")

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("run_dir",type=Path)
    args=parser.parse_args()
    analyze(args.run_dir)
