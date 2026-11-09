"""Kaggle Notebook Python-script payload, rendered by scripts/kaggle_controller.py.

Runs the repository's existing hidden-rule memory diagnostic (not official ALFWorld).
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

CONFIG = json.loads('__KAGGLE_CONFIG_JSON__')
WORKING = Path('/kaggle/working') if Path('/kaggle/working').is_dir() else Path.cwd()
SOURCE = WORKING / 'evo-mem-bench-src'
OUTPUT = WORKING / 'evomem_results'


def command(*args: str, cwd: Path | None = None) -> None:
    print('$ ' + ' '.join(args), flush=True)
    subprocess.run(args, check=True, cwd=str(cwd) if cwd else None)


def main() -> None:
    print(json.dumps({'phase': 'setup', 'config': CONFIG,
                      'cuda': os.environ.get('CUDA_VISIBLE_DEVICES')}, indent=2), flush=True)
    command('git', 'clone', '--filter=blob:none', '--no-checkout', CONFIG['repo_url'], str(SOURCE))
    command('git', '-C', str(SOURCE), 'fetch', '--depth=1', 'origin', CONFIG['source_sha'])
    command('git', '-C', str(SOURCE), 'checkout', '--detach', CONFIG['source_sha'])
    resolved = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    if resolved != CONFIG['source_sha']:
        raise RuntimeError('Git checkout did not match pinned SHA')

    command(sys.executable, '-m', 'pip', 'install', '-q', '-e', str(SOURCE),
            'openai>=1.0', 'transformers>=4.46,<5', 'accelerate>=0.34,<2')
    sys.path.insert(0, str(SOURCE / 'src'))
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('No GPU allocated by Kaggle; run requires GPU')
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_id = CONFIG['model']
    start = time.monotonic()
    print(f'Loading {model_id}; cuda={torch.cuda.get_device_name(0)}', flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float16, device_map='auto', trust_remote_code=False
    ).eval()

    class HFWorker:
        """Drop-in local model for evomembench.experiment.LocalWorker."""
        def __init__(self, model: str, base_url: str, api_key: str, temperature: float):
            if model != model_id:
                raise ValueError('Worker model mismatch')
            if float(temperature) != 0:
                raise ValueError('Kaggle worker supports deterministic decoding only')

        def generate(self, system: str, prompt: str, *, max_tokens: int = 128):
            messages = [{'role': 'system', 'content': system},
                        {'role': 'user', 'content': prompt}]
            ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, return_tensors='pt'
            ).to(model.device)
            with torch.inference_mode():
                tokens = model.generate(
                    ids, max_new_tokens=max_tokens, do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )[0]
            generated = tokens[ids.shape[-1]:]
            return tokenizer.decode(generated, skip_special_tokens=True).strip(), {
                'prompt_tokens': int(ids.shape[-1]),
                'completion_tokens': int(generated.numel()),
            }

    # Retain the repository's benchmark prompts, manifests, memory and scoring.
    import evomembench.experiment as experiment
    experiment.LocalWorker = HFWorker
    sys.argv = [
        'evomembench.experiment', '--model', model_id,
        '--worlds', str(CONFIG['worlds']),
        '--seeds', ','.join(str(x) for x in CONFIG['seeds']),
        '--methods', ','.join(CONFIG['methods']),
        '--output', str(OUTPUT), '--gate'
    ]
    print('Starting hidden-rule benchmark with competence gate', flush=True)
    experiment.main()
    report = {
        'result': 'complete', 'source_sha': resolved, 'config': CONFIG,
        'elapsed_seconds': round(time.monotonic()-start, 1),
        'summary': json.loads((OUTPUT / 'summary.json').read_text()),
    }
    (WORKING / 'kaggle_run_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        (WORKING / 'kaggle_run_failure.json').write_text(json.dumps({
            'result': 'failed', 'error_type': type(exc).__name__,
            'error': str(exc), 'source_sha': CONFIG['source_sha'],
        }, indent=2) + '\n')
        traceback.print_exc()
        raise
