#!/usr/bin/env python3
"""GitHub Actions controller for Kaggle free GPU notebooks.

Only an explicit launch or armed launch-request.json can consume GPU quota.
Monitoring and tests never launch GPU jobs.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
MODEL_ALLOWLIST = {
    'Qwen/Qwen2.5-0.5B-Instruct',
    'Qwen/Qwen2.5-1.5B-Instruct',
    'Qwen/Qwen2.5-3B-Instruct',
}
METHOD_ALLOWLIST = {
    'none', 'oracle', 'random', 'full_history', 'bm25',
    'success_only', 'reward_labeled', 'ucb', 'writer',
}
KERNEL_PATTERN = re.compile(r'^[a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+$')
SHA_PATTERN = re.compile(r'^[0-9a-f]{40}$')


def sh(*args: str) -> None:
    subprocess.run(args, check=True)


def validate_config(obj: dict) -> dict:
    if not isinstance(obj, dict):
        raise ValueError('Launch request must be an object')
    model = obj.get('model', 'Qwen/Qwen2.5-1.5B-Instruct')
    if model not in MODEL_ALLOWLIST:
        raise ValueError('Model not in Kaggle T4 allowlist')
    worlds = obj.get('worlds', 6)
    if type(worlds) is not int or not 6 <= worlds <= 24:
        raise ValueError('worlds must be an integer from 6 to 24')
    seeds = obj.get('seeds', [11])
    if not isinstance(seeds, list) or not 1 <= len(seeds) <= 3 or any(
        type(s) is not int or s < 0 or s > 100000 for s in seeds
    ):
        raise ValueError('seeds must be 1-3 nonnegative integers <=100000')
    if len(set(seeds)) != len(seeds):
        raise ValueError('Duplicate seeds')
    methods = obj.get('methods', ['none', 'oracle', 'bm25', 'success_only'])
    if not isinstance(methods, list) or not 2 <= len(methods) <= 9 or any(
        type(s) is not str for s in methods
    ):
        raise ValueError('methods must be a list of method names')
    if any(m not in METHOD_ALLOWLIST for m in methods) or len(set(methods)) != len(methods):
        raise ValueError('Invalid or duplicate memory methods')
    if 'none' not in methods or 'oracle' not in methods:
        raise ValueError('Both none and oracle are required for scientific gates')
    calls = worlds * len(seeds) * len(methods)
    if calls > 150:
        raise ValueError(f'Exceeded 150 planned worker calls ({calls})')
    return {'model': model, 'worlds': worlds, 'seeds': seeds, 'methods': methods}


def render_kernel(config: dict, source_sha: str, kernel_ref: str, destination: Path) -> None:
    if not SHA_PATTERN.fullmatch(source_sha):
        raise ValueError('Expected a 40-character SHA')
    if not KERNEL_PATTERN.fullmatch(kernel_ref):
        raise ValueError('KAGGLE_KERNEL_REF must be username/notebook-slug')
    payload = dict(validate_config(config), source_sha=source_sha,
                   repo_url='https://github.com/Vedant2100/evo-mem-bench.git')
    template = (ROOT / 'kaggle/run_hidden_rules.py').read_text()
    code = template.replace("'__KAGGLE_CONFIG_JSON__'", repr(json.dumps(payload)))
    assert '__KAGGLE_CONFIG_JSON__' not in code
    destination.mkdir(parents=True, exist_ok=True)
    (destination / 'run_hidden_rules.py').write_text(code)
    metadata = {
        'id': kernel_ref,
        'title': 'EvoMemBench Hidden Rule GPU Diagnostic',
        'code_file': 'run_hidden_rules.py',
        'language': 'python',
        'kernel_type': 'script',
        'is_private': True,
        'enable_internet': True,
        'enable_gpu': True,
        'machine_shape': 'NvidiaTeslaT4',
        'dataset_sources': [], 'competition_sources': [],
        'kernel_sources': [], 'model_sources': [],
    }
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    compile(code, 'run_hidden_rules.py', 'exec')


def status(kernel_ref: str, *, allow_missing: bool = False) -> str:
    proc = subprocess.run(['kaggle', 'kernels', 'status', kernel_ref],
                          text=True, capture_output=True)
    output = (proc.stdout + '\n' + proc.stderr).strip()
    if proc.returncode:
        if allow_missing and re.search(
            r'(^|\W)(404|not found|does not exist)(\W|$)', output, re.IGNORECASE
        ):
            print('Kernel does not exist yet; first creation requires explicit opt-in')
            return 'NOT_FOUND'
        raise RuntimeError('Kaggle status failed (no launch attempted): ' + output[:800])
    print(output)
    found = re.search(r'KernelWorkerStatus\.([A-Z_]+)', output)
    if not found:
        print('Unrecognized Kaggle status response; not making launch decisions')
        return 'UNKNOWN'
    return found.group(1)


def write_summary(operation: str, state: str, ref: str) -> None:
    result = {
        'observed_utc': datetime.now(timezone.utc).isoformat(),
        'operation': operation, 'kernel': ref, 'status': state,
        'kernel_url': f'https://www.kaggle.com/code/{ref}'
    }
    (ROOT / 'kaggle_status.json').write_text(json.dumps(result, indent=2) + '\n')
    summary = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary:
        with open(summary, 'a') as f:
            f.write(f'### Kaggle experiment\n\n- Kernel: https://www.kaggle.com/code/{ref}\n- Status: **{state}**\n- Checked: {result["observed_utc"]}\n')


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--action', choices=['launch', 'status', 'collect', 'from-request'], required=True)
    p.add_argument('--source-sha', default=os.getenv('GITHUB_SHA', ''))
    p.add_argument('--request', default='kaggle/launch-request.json')
    p.add_argument('--worlds', type=int, default=6)
    p.add_argument('--model', default='Qwen/Qwen2.5-1.5B-Instruct')
    p.add_argument('--seeds', default='11')
    p.add_argument('--methods', default='none,oracle,bm25,success_only')
    args = p.parse_args()

    if args.action == 'from-request':
        request = json.loads((ROOT / args.request).read_text())
        if request.get('armed') is not True:
            print('Launch request not armed; no GPU launched')
            return
        if not re.fullmatch(r'[0-9a-zA-Z_-]{8,80}', str(request.get('request_id', ''))):
            raise ValueError('Unique request_id required when armed')
        config = validate_config(request)
        operation = 'launch'
    else:
        config = validate_config({
            'model': args.model, 'worlds': args.worlds,
            'seeds': [int(v) for v in args.seeds.split(',')],
            'methods': args.methods.split(',')
        }) if args.action == 'launch' else {}
        operation = args.action

    ref = os.environ.get('KAGGLE_KERNEL_REF', '').strip()
    token = os.environ.get('KAGGLE_API_TOKEN', '').strip()
    if not ref or not token:
        if operation != 'launch':
            print('Kaggle secret/variable missing. Monitoring skipped.')
            return
        raise ValueError('Add GitHub secret KAGGLE_API_TOKEN and variable KAGGLE_KERNEL_REF')
    if not KERNEL_PATTERN.fullmatch(ref):
        raise ValueError('Invalid KAGGLE_KERNEL_REF')

    if operation == 'launch':
        current = status(ref, allow_missing=True)
        if current in {'RUNNING', 'QUEUED', 'NEW_SCRIPT', 'CANCEL_REQUESTED'}:
            raise RuntimeError(f'Existing Kaggle job {current}; refusing to overwrite')
        if current == 'UNKNOWN':
            raise RuntimeError('Could not determine Kaggle status; refusing to launch')
        if current == 'NOT_FOUND' and os.getenv('KAGGLE_ALLOW_FIRST_LAUNCH') != 'true':
            raise RuntimeError('For first creation temporarily set KAGGLE_ALLOW_FIRST_LAUNCH=true')
        dest = ROOT / 'kaggle-build'
        render_kernel(config, args.source_sha, ref, dest)
        print('Launching', config, 'from', args.source_sha, 'to', ref, flush=True)
        sh('kaggle', 'kernels', 'push', '-p', str(dest), '--timeout', '10800')
        state = status(ref, allow_missing=True)
    else:
        state = status(ref, allow_missing=True)
        if state not in {'UNKNOWN', 'NOT_FOUND'}:
            proc = subprocess.run(['kaggle', 'kernels', 'logs', ref],
                                  capture_output=True, text=True, timeout=90)
            (ROOT / 'kaggle_logs.txt').write_text((proc.stdout + '\n' + proc.stderr)[-30000:])
        if operation == 'collect' and state == 'COMPLETE':
            out = ROOT / 'kaggle-output'
            out.mkdir(exist_ok=True)
            sh('kaggle', 'kernels', 'output', ref, '-p', str(out),
               '--file-pattern', r'.*\.(json|txt)$')
    write_summary(operation, state, ref)
    if state == 'ERROR':
        print('::error::Kaggle experiment failed; inspect the logs')
        sys.exit(1)


if __name__ == '__main__':
    main()
