#!/usr/bin/env python3
"""Strict, idempotent instrumentation of the *pinned* official evaluator."""
import argparse
import json
import shutil
from pathlib import Path

def replace_one(path, old, new, label):
    current = path.read_text()
    if new in current:
        return
    count = current.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one upstream match, got {count}; refusing unsafe patch")
    path.write_text(current.replace(old, new))

def patch(upstream, adapter):
    component = Path(upstream) / "Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB"
    evaluator = component / "scripts/eval_alfworld_with_memory.py"
    factory = component / "scripts/memory/base.py"
    if not evaluator.exists() or not factory.exists():
        raise FileNotFoundError("Official evaluator/factory missing; bootstrap upstream first")
    standard_before = '''                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )'''
    standard_after = '''                request_kwargs = {"model": self.model, "messages": messages}
                temperature = os.environ.get("EVO_MEM_TEMPERATURE")
                if temperature not in (None, ""):
                    request_kwargs["temperature"] = float(temperature)
                response = self.client.chat.completions.create(**request_kwargs)'''
    replace_one(evaluator, standard_before, standard_after, "standard temperature")
    batch_before = '''                response = self.client.batch.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )'''
    batch_after = '''                request_kwargs = {"model": self.model, "messages": messages}
                temperature = os.environ.get("EVO_MEM_TEMPERATURE")
                if temperature not in (None, ""):
                    request_kwargs["temperature"] = float(temperature)
                response = self.client.batch.chat.completions.create(**request_kwargs)'''
    replace_one(evaluator, batch_before, batch_after, "batch temperature")

    helper = '''def _cap_memory_suffix(conversation, original_prompt, budget):
    """Only truncate retrieved memory, never the task instructions."""
    if not conversation[0]["content"].startswith(original_prompt):
        raise ValueError("Memory backend changed the upstream prompt; cannot safely cap it")
    suffix = conversation[0]["content"][len(original_prompt):]
    if not suffix or budget <= 0:
        return conversation
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(suffix)
    if len(tokens) > budget:
        conversation[0]["content"] = original_prompt + enc.decode(tokens[:budget])
    return conversation


'''
    current = evaluator.read_text()
    if helper not in current:
        marker = "def eval_one(data_idx, env_args, agent, max_rounds, memory):"
        replace_one(evaluator, marker, helper + marker, "injection helper")
    injection_before = '''    if memory is not None:
        conversation, inject_stats = memory.inject(conversation)
'''
    injection_after = '''    if memory is not None:
        _base_prompt = conversation[0]["content"]
        conversation, inject_stats = memory.inject(conversation)
        _cap = int(os.environ.get("EVO_MEM_MAX_INJECT_TOKENS", "0") or 0)
        if _cap:
            conversation = _cap_memory_suffix(conversation, _base_prompt, _cap)
'''
    replace_one(evaluator, injection_before, injection_after, "memory suffix cap")
    factory_before = '''    raise ValueError(
        f"Unknown memory_type: {memory_type!r}. "'''
    factory_after = '''    if memory_type == "controlled":
        from .controlled_adapter import ControlledMemory
        return ControlledMemory(read_only=read_only, **kwargs)
    raise ValueError(
        f"Unknown memory_type: {memory_type!r}. "'''
    replace_one(factory, factory_before, factory_after, "controlled memory factory")
    # The upstream server retains every completed TextWorld game instance.
    # Sequential experiment runs can therefore exhaust simulator resources.
    # This explicit close operation only releases the finished episode's
    # environment; parallel episodes remain untouched.
    server = component / "agentenv-alfworld/agentenv_alfworld/server.py"
    model = component / "agentenv-alfworld/agentenv_alfworld/model.py"
    wrapper = component / "agentenv-alfworld/agentenv_alfworld/env_wrapper.py"
    client = component / "agentenv/agentenv/envs/alfworld.py"
    replace_one(
        model,
        "class StepRequestBody(BaseModel):",
        "class CloseRequestBody(BaseModel):\n    id: int\n\n\nclass StepRequestBody(BaseModel):",
        "close request model",
    )
    close_route = (
        '@app.post("/close")\n'
        "async def close(body: CloseRequestBody):\n"
        "    return server.close(body.id)\n\n\n"
    )
    replace_one(
        server,
        '@app.post("/reset")',
        close_route + '@app.post("/reset")',
        "close environment endpoint",
    )
    close_method = (
        "    def close(self, idx: int):\n"
        "        try:\n"
        "            self._check_id(idx, True)\n"
        "            instance = self.env_init.pop(idx, None)\n"
        "            if instance is not None:\n"
        "                instance.close()\n"
        "            self.env.pop(idx, None)\n"
        "            self.info.pop(idx, None)\n"
        "            if idx in self.ls:\n"
        "                self.ls.remove(idx)\n"
        "            return {'id': idx, 'closed': True}\n"
        "        except Exception as exc:\n"
        "            return {'error': 'close_failed: ' + str(exc)}\n\n"
    )
    replace_one(
        wrapper,
        "    def get_observation(self, idx: int):",
        close_method + "    def get_observation(self, idx: int):",
        "close completed TextWorld game",
    )
    replace_one(
        client,
        '        response = self._post("reset", {"game": game, "world_type": world_type})\n',
        '        response = self._post("reset", {"game": game, "world_type": world_type})\n'
        '        if "error" in response:\n'
        '            raise RuntimeError(f"ALFWorld server could not reset game {game}: {response.get('error')}")\n',
        "expose original server reset error",
    )
    replace_one(
        evaluator,
        "    return data_idx, conversation, reward, total_pt, total_ct, total_cached, sample_latency, total_llm_latency, rounds, inject_stats, update_stats",
        "    try:\n"
        "        response = client.close()\n"
        "        if isinstance(response, dict) and response.get('error'):\n"
        "            print('ALFWORLD_CLOSE_WARNING', response['error'])\n"
        "    except Exception as exc:\n"
        "        print('ALFWORLD_CLOSE_WARNING', repr(exc))\n"
        "    return data_idx, conversation, reward, total_pt, total_ct, total_cached, sample_latency, total_llm_latency, rounds, inject_stats, update_stats",
        "close finished episode from evaluator",
    )
    shutil.copy2(adapter, component / "scripts/memory/controlled_adapter.py")
    (component / ".controlled_patch.json").write_text(json.dumps({
        "patch_version": 1, "temperature": True, "memory_only_cap": True,
        "custom_adapter": "controlled"}, indent=2) + "\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--adapter", required=True)
    a = parser.parse_args()
    patch(a.upstream, a.adapter)
    print("Patched evaluator and memory factory")

if __name__ == "__main__":
    main()
