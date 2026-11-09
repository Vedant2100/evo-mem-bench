#!/usr/bin/env python3
"""Staged controlled runner around the unchanged official ALFWorld evaluator."""
from __future__ import annotations
import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import os
import re
import socket
import subprocess
import sys
import uuid
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from prepare_manifest import CATEGORIES, COMPONENT, create_manifest

def load_env():
    path=ROOT/".env"
    if path.exists():
        for raw in path.read_text().splitlines():
            line=raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key,val=line.split("=",1)
            os.environ.setdefault(key.strip(),val.strip().strip("'").strip('"'))

def read_json(path):
    return json.loads(path.read_text())

def substitution(value,variables):
    if isinstance(value,dict):
        return {k:substitution(v,variables) for k,v in value.items()}
    if isinstance(value,list):
        return [substitution(v,variables) for v in value]
    if isinstance(value,str):
        def expand(m):
            name=m.group(1)
            if name not in variables:
                raise ValueError(f"Missing config variable {name}")
            return str(variables[name])
        return re.sub(r"@@([A-Z0-9_]+)@@",expand,value)
    return value

def sanitized(obj):
    if isinstance(obj,dict):
        return {k:("***" if v else "") if ("api_key" in k.lower() or "secret" in k.lower())
                else sanitized(v) for k,v in obj.items()}
    if isinstance(obj,list):
        return [sanitized(v) for v in obj]
    return obj

def validate_outputs(outdir,indices):
    missing=[]
    bad=[]
    for i in indices:
        path=outdir/f"alfworld_{i}.json"
        if not path.exists():
            missing.append(i)
            continue
        try:
            result=read_json(path)
            assert result.get("item_id")==f"alfworld_{i}", "item_id mismatch"
            reward=float(result["reward"])
            success=int(result["success"])
            assert math.isfinite(reward) and success in (0,1)
            assert success==int(reward>=1.0)
        except Exception as exc:
            bad.append((i,str(exc)))
    if missing or bad:
        raise RuntimeError(f"INVALID condition {outdir}: missing={missing[:8]}, malformed={bad[:5]}")
    summ=outdir/"summary.json"
    if not summ.exists() or int(read_json(summ)["num_samples"])!=len(indices):
        raise RuntimeError(f"INVALID upstream summary {summ}")

def config_file(run_dir,seed,method,category,spec,memory_dir,variables):
    if spec["kind"]=="none":
        return None
    path=run_dir/"configs"/f"seed_{seed}"/method/f"{category}.json"
    path.parent.mkdir(parents=True,exist_ok=True)
    if spec["kind"]=="controlled":
        params={"mode":spec["mode"],"memory_dir":str(memory_dir),"top_k":10,
                "chunk_size":1024,"max_context_tokens":variables["MEMORY_CAP"],
                "random_seed":seed}
    else:
        params=substitution(copy.deepcopy(spec["params"]),{**variables,"MEMORY_DIR":str(memory_dir)})
    path.write_text(json.dumps(params,indent=2)+"\n")
    return path

def condition(*,run_dir,seed,method,spec,phase,category,indices,model,mode,rounds,
              cap,port,env,memory_dir=None,cfg=None,readonly=False,source=None):
    label=category if source is None else f"{source}__to__{category}"
    out=run_dir/f"seed_{seed}"/method/phase/label
    if out.exists():
        raise FileExistsError(f"Refusing to reuse condition output {out}")
    out.mkdir(parents=True)
    script=COMPONENT/"scripts/eval_alfworld_with_memory.py"
    cmd=[sys.executable,str(script),"--port",str(port),"--indices",json.dumps(indices),
         "--max_rounds",str(rounds),"--parallel","1","--api_mode",mode,
         "--output_dir",str(out)]
    if mode=="standard":
        cmd.extend(["--model",model])
    if spec["kind"]=="none":
        cmd.append("--no_memory")
    else:
        cmd.extend(["--memory_type", "controlled" if spec["kind"]=="controlled" else spec["backend"],
                    "--memory_config",str(cfg)])
        if readonly:
            cmd.append("--readonly_memory")
    meta={"seed":seed,"method":method,"phase":phase,"category":category,"source":source,
          "read_only":readonly,"item_ids":indices,"model":model,"api_mode":mode,
          "max_rounds":rounds,"memory_cap_tokens":cap,"memory_store":str(memory_dir) if memory_dir else None,
          "memory_config":sanitized(read_json(cfg)) if cfg else None}
    (out/"run_meta.json").write_text(json.dumps(meta,indent=2)+"\n")
    log=run_dir/"logs"/f"seed_{seed}"/method/phase/(label+".log")
    log.parent.mkdir(parents=True,exist_ok=True)
    print(f"RUN {phase} {method} {label} [{len(indices)} episodes]",flush=True)
    with log.open("w") as handle:
        completed=subprocess.run(cmd,cwd=COMPONENT,env=env,stdout=handle,stderr=subprocess.STDOUT)
    if completed.returncode:
        raise RuntimeError(f"Upstream evaluator returned {completed.returncode}; inspect {log}")
    validate_outputs(out,indices)

def create_plan(args):
    load_env()
    config=read_json(ROOT/"configs/experiment.json")
    specs=read_json(ROOT/"configs/methods.json")
    plan=copy.deepcopy(config["plans"][args.plan])
    methods=[x.strip() for x in args.methods.split(",") if x.strip()] if args.methods else plan["methods"]
    seeds=[int(x.strip()) for x in args.seeds.split(",") if x.strip()] if args.seeds else plan["seeds"]
    if not methods or "none" not in methods or len(methods)!=len(set(methods)):
        raise ValueError("Methods must include unique names and 'none' for paired baseline")
    if not seeds or len(seeds)!=len(set(seeds)):
        raise ValueError("Need unique order seeds")
    unknown=set(methods)-set(specs)
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    train_n=args.train_per_category if args.train_per_category is not None else plan.get("train_per_category")
    test_n=args.test_per_category if args.test_per_category is not None else plan.get("test_per_category")
    cap=args.memory_cap_tokens if args.memory_cap_tokens is not None else plan.get("memory_cap_tokens",config["worker"]["memory_cap_tokens"])
    if cap<0:
        raise ValueError("Memory cap must be nonnegative")
    worker=copy.deepcopy(config["worker"])
    worker["api_mode"]=args.api_mode or os.environ.get("WORKER_API_MODE") or worker["api_mode"]
    worker["model"]=args.model or os.environ.get("WORKER_MODEL") or worker["model"]
    if worker["api_mode"] not in ("standard","batch"):
        raise ValueError("Only upstream standard and ARK batch modes are supported")
    if worker["api_mode"]=="batch":
        worker["model"]=os.environ.get("BATCH_MODEL") or worker["model"]
    worker["port"]=int(os.environ.get("ALFWORLD_PORT") or worker["port"])
    worker["memory_cap_tokens"]=cap
    worker["temperature"]=os.environ.get("WORKER_TEMPERATURE") or worker.get("temperature")
    manifests={str(s):create_manifest(s,train_n,test_n) for s in seeds}
    protocol=plan["protocol"]
    writer_methods=[m for m in methods if m not in ("none","playbook")]
    train_count=sum(sum(map(len,manifest["train"].values())) for manifest in manifests.values())
    test_count=sum(sum(map(len,manifest["test"].values())) for manifest in manifests.values())
    if protocol=="heldout":
        episodes=train_count*len(writer_methods)+test_count*len(methods)
    elif protocol=="online":
        episodes=test_count*len(methods)
    else:
        episodes=test_count*len(methods)+test_count*5*len(writer_methods)
    return config,specs,plan,methods,seeds,manifests,worker,protocol,episodes

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--plan",choices=("smoke","competence","heldout","original","learned","transfer"),required=True)
    parser.add_argument("--methods")
    parser.add_argument("--seeds")
    parser.add_argument("--train-per-category",type=int)
    parser.add_argument("--test-per-category",type=int)
    parser.add_argument("--memory-cap-tokens",type=int)
    parser.add_argument("--model")
    parser.add_argument("--api-mode",choices=("standard","batch"))
    parser.add_argument("--dry-run",action="store_true")
    parser.add_argument("--output-dir")
    args=parser.parse_args()
    try:
        config,specs,plan,methods,seeds,maps,worker,protocol,episodes=create_plan(args)
    except (ValueError,FileNotFoundError) as e:
        parser.error(str(e))
    print(json.dumps({"plan":args.plan,"protocol":protocol,"methods":methods,"seeds":seeds,
                      "worker_model":worker["model"],"api_mode":worker["api_mode"],
                      "memory_cap_tokens":worker["memory_cap_tokens"],
                      "episodes":episodes,
                      "worker_call_upper_bound":episodes*worker["max_rounds"],
                      "warning":"Memory writer and embedding calls are additional and may be costly"},
                     indent=2))
    if args.dry_run:
        print("Dry-run only; no environment/inference calls or run directory created")
        return

    root=ROOT/"vendor/EvoMemBench"
    actual=subprocess.check_output(["git","-C",str(root),"rev-parse","HEAD"],text=True).strip()
    if actual!=config["upstream"]["commit"]:
        raise RuntimeError("Pinned upstream commit mismatch")
    if not (COMPONENT/".controlled_patch.json").exists():
        raise RuntimeError("Run scripts/bootstrap_upstream.sh to register the controlled adapter")
    if worker["api_mode"]=="standard":
        if not os.environ.get("OPENAI_BASE_URL") or not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError("Set OPENAI_BASE_URL and OPENAI_API_KEY for OpenAI-compatible worker")
    elif not os.environ.get("DEEPSEEK_API_KEY") or not os.environ.get("BATCH_MODEL"):
        raise RuntimeError("Set DEEPSEEK_API_KEY and BATCH_MODEL for ARK batch worker")
    for method in methods:
        for required in specs[method].get("requires",[]):
            if not os.environ.get(required):
                raise RuntimeError(f"{method} requires {required}")
    with socket.create_connection(("127.0.0.1",worker["port"]),timeout=5):
        pass

    run_id=dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:7]
    run_dir=Path(args.output_dir).resolve() if args.output_dir else ROOT/"runs"/run_id
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True)
    env=os.environ.copy()
    env["EVO_MEM_TEMPERATURE"]="" if worker["temperature"] is None else str(worker["temperature"])
    env["EVO_MEM_MAX_INJECT_TOKENS"]=str(worker["memory_cap_tokens"])
    env["PYTHONPATH"]=str(COMPONENT/"agentenv")+os.pathsep+env.get("PYTHONPATH","")
    if worker["api_mode"]=="batch":
        env["BATCH_MODEL"]=worker["model"]
    variables={
        **env,
        "MEMORY_MODEL":env.get("BATCH_MODEL") or env.get("MEMORY_MODEL") or "deepseek-v3-2-251201",
        "DEEPSEEK_BASE_URL":env.get("DEEPSEEK_BASE_URL") or "https://ark.cn-beijing.volces.com/api/v3",
        "DASHSCOPE_BASE_URL":env.get("DASHSCOPE_BASE_URL") or "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "MEMORY_CAP":worker["memory_cap_tokens"]
    }
    # The upstream evaluator loads its .env at import; process env overrides still take priority.
    (COMPONENT/".env").write_text(
        "\n".join(f"{k}={v}" for k,v in env.items()
                  if k in ("BATCH_MODEL","DEEPSEEK_API_KEY","DEEPSEEK_BASE_URL",
                           "OPENAI_API_KEY","OPENAI_BASE_URL","DASHSCOPE_API_KEY",
                           "DASHSCOPE_BASE_URL","EVO_MEM_TEMPERATURE","EVO_MEM_MAX_INJECT_TOKENS"))+"\n"
    )
    meta={"run_id":run_id,"plan":args.plan,"protocol":protocol,"worker":worker,
          "upstream_commit":actual,"methods":methods,"specs":specs,"seeds":seeds,
          "manifests":maps,"finished":False}
    (run_dir/"manifest.json").write_text(json.dumps(meta,indent=2)+"\n")
    for seed in seeds:
        selected=maps[str(seed)]
        for method in methods:
            spec=specs[method]
            dirs={}
            cfgs={}
            for cat in CATEGORIES:
                if spec["kind"]!="none":
                    store=run_dir/f"seed_{seed}"/method/"memory"/cat
                    store.mkdir(parents=True)
                    dirs[cat]=store
                    cfgs[cat]=config_file(run_dir,seed,method,cat,spec,store,variables)
            if protocol=="heldout":
                # Freeze memory before any evaluation; 'playbook' needs no training.
                if method not in ("none","playbook"):
                    for cat in CATEGORIES:
                        condition(run_dir=run_dir,seed=seed,method=method,spec=spec,
                                  phase="train",category=cat,indices=selected["train"][cat],
                                  model=worker["model"],mode=worker["api_mode"],rounds=worker["max_rounds"],
                                  cap=worker["memory_cap_tokens"],port=worker["port"],env=env,
                                  memory_dir=dirs[cat],cfg=cfgs[cat])
                for cat in CATEGORIES:
                    condition(run_dir=run_dir,seed=seed,method=method,spec=spec,
                              phase="eval",category=cat,indices=selected["test"][cat],
                              model=worker["model"],mode=worker["api_mode"],rounds=worker["max_rounds"],
                              cap=worker["memory_cap_tokens"],port=worker["port"],env=env,
                              memory_dir=dirs.get(cat),cfg=cfgs.get(cat),readonly=spec["kind"]!="none")
            else:
                for cat in CATEGORIES:
                    condition(run_dir=run_dir,seed=seed,method=method,spec=spec,
                              phase="online",category=cat,indices=selected["test"][cat],
                              model=worker["model"],mode=worker["api_mode"],rounds=worker["max_rounds"],
                              cap=worker["memory_cap_tokens"],port=worker["port"],env=env,
                              memory_dir=dirs.get(cat),cfg=cfgs.get(cat))
                if protocol=="transfer" and method not in ("none","playbook"):
                    for src in CATEGORIES:
                        for target in CATEGORIES:
                            if src==target:
                                continue
                            condition(run_dir=run_dir,seed=seed,method=method,spec=spec,
                                      phase="transfer",category=target,source=src,
                                      indices=selected["test"][target],model=worker["model"],
                                      mode=worker["api_mode"],rounds=worker["max_rounds"],
                                      cap=worker["memory_cap_tokens"],port=worker["port"],env=env,
                                      memory_dir=dirs[src],cfg=cfgs[src],readonly=True)
    meta["finished"]=True
    (run_dir/"manifest.json").write_text(json.dumps(meta,indent=2)+"\n")
    print(f"Completed validated run at {run_dir}")
    print(f"Analyze: python scripts/analyze_results.py {run_dir}")

if __name__=="__main__":
    main()
