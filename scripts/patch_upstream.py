#!/usr/bin/env python3
"""Strict, idempotent instrumentation of the pinned official CROSSEP-EMB code."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import textwrap
from pathlib import Path


def replace_one(path: Path, old: str, new: str, label: str) -> None:
    current = path.read_text()
    if new in current:
        return
    count = current.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one upstream match, got {count}; refusing unsafe patch")
    path.write_text(current.replace(old, new))


def replace_section(path: Path, start: str, end: str, replacement: str, label: str) -> None:
    current = path.read_text()
    if replacement.strip() in current:
        return
    if current.count(start) != 1 or current.count(end) != 1:
        raise RuntimeError(f"{label}: expected unique section markers; refusing unsafe patch")
    left = current.index(start)
    right = current.index(end, left + len(start))
    if right <= left:
        raise RuntimeError(f"{label}: invalid section boundaries")
    path.write_text(current[:left] + replacement + current[right:])


def patch_episode_cleanup(evaluator: Path) -> None:
    current = evaluator.read_text()
    start = current.find("def eval_one(data_idx, env_args, agent, max_rounds, memory):")
    if start < 0:
        raise RuntimeError("eval_one signature changed; refusing cleanup patch")
    end = current.find("\ndef write_summary_txt", start)
    if end < 0:
        raise RuntimeError("could not locate end of eval_one")
    function = current[start:end]
    if "finally:" in function and "client.close()" in function:
        return
    client_line = "    client = AlfWorldEnvClient(**env_args)\n"
    if function.count(client_line) != 1:
        raise RuntimeError("eval_one client construction changed; refusing cleanup patch")
    prefix_end = start + function.index(client_line) + len(client_line)
    remainder = current[prefix_end:end]
    # Remove the old success-path-only close block from earlier patch versions.
    remainder = re.sub(
        r"\n    try:\n        response = client\.close\(\)\n"
        r"        if isinstance\(response, dict\) and response\.get\('error'\):\n"
        r"            print\('ALFWORLD_CLOSE_WARNING', response\['error'\]\)\n"
        r"    except Exception as exc:\n"
        r"        print\('ALFWORLD_CLOSE_WARNING', repr\(exc\)\)\n",
        "\n",
        remainder,
        count=1,
    )
    wrapped = (
        "    try:\n"
        + textwrap.indent(remainder, "    ")
        + "    finally:\n"
        + "        try:\n"
        + "            response = client.close()\n"
        + "            if isinstance(response, dict) and response.get('error'):\n"
        + "                print('ALFWORLD_CLOSE_WARNING', response['error'])\n"
        + "        except Exception as exc:\n"
        + "            print('ALFWORLD_CLOSE_WARNING', repr(exc))\n"
    )
    evaluator.write_text(current[:prefix_end] + wrapped + current[end:])


def patch_local_ark_utils(path: Path) -> None:
    replace_one(path, "import json\nimport threading\n", "import json\nimport os\nimport threading\n", "ARK helper os import")
    replacement = r'''class ArkBatchLLM:
    """Local OpenAI-compatible inference when EVO_MEM_LLM_BASE_URL is set; ARK otherwise."""

    def __init__(self, api_key: str, model: str, max_retries: int = 8, retry_delay: float = 5.0):
        self.model = model
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self._local_client = None
        local_base = os.getenv("EVO_MEM_LLM_BASE_URL")
        if local_base:
            from openai import OpenAI
            self._local_client = OpenAI(
                base_url=local_base,
                api_key=os.getenv("EVO_MEM_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "local",
                timeout=240.0,
                max_retries=2,
            )
            self.client = None
        else:
            try:
                from volcenginesdkarkruntime import Ark
            except ImportError as e:
                raise ImportError(
                    "ArkBatchLLM requires the volcengine SDK or EVO_MEM_LLM_BASE_URL."
                ) from e
            self.client = Ark(api_key=api_key)

    @staticmethod
    def _normalize_messages(messages: List[dict]) -> List[dict]:
        normalized = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
                    elif isinstance(part, str):
                        parts.append(part)
                content = "\n".join(part for part in parts if part)
            normalized.append({"role": message.get("role", "user"), "content": content})
        return normalized

    def __call__(self, messages):
        normalized = self._normalize_messages(messages)
        stats = _tls_stats()
        for attempt in range(self.max_retries):
            try:
                if self._local_client is not None:
                    response = self._local_client.chat.completions.create(
                        model=self.model, messages=normalized, temperature=0.0
                    )
                else:
                    response = self.client.batch.chat.completions.create(
                        model=self.model, messages=normalized
                    )
                if stats is not None and response.usage:
                    stats.input_tokens += response.usage.prompt_tokens or 0
                    stats.output_tokens += response.usage.completion_tokens or 0
                    details = getattr(response.usage, "prompt_tokens_details", None)
                    stats.cached_tokens += getattr(details, "cached_tokens", 0) or 0
                return SimpleNamespace(content=response.choices[0].message.content or "")
            except Exception as exc:
                if attempt == self.max_retries - 1:
                    raise
                print(
                    f"[Memory LLM retry {attempt + 1}/{self.max_retries}] {exc}. "
                    f"Retrying in {self.retry_delay:.1f}s..."
                )
                time.sleep(self.retry_delay)
'''
    replace_section(path, "class ArkBatchLLM:", "\ndef patch_embedding_model", replacement + "\n", "local memory LLM adapter")


def patch_direct_ark_adapter(path: Path, class_name: str) -> None:
    old_constructor = '''        # ARK batch API client
        self._ark_client = None
        self._batch_model = None
        if ark_batch_config:
            try:
                from volcenginesdkarkruntime import Ark
            except ImportError as e:
                raise ImportError(
                    "ark_batch_config requires volcengine SDK. Run:\\n"
                    "  pip install 'volcengine-python-sdk[ark]'"
                ) from e
            self._ark_client = Ark(api_key=ark_batch_config["api_key"])
            self._batch_model = ark_batch_config["model"]
'''
    new_constructor = '''        # Prefer the local GPU-served OpenAI-compatible memory model when configured.
        self._ark_client = None
        self._local_client = None
        self._batch_model = None
        if ark_batch_config:
            self._batch_model = ark_batch_config["model"]
            local_base = os.getenv("EVO_MEM_LLM_BASE_URL")
            if local_base:
                from openai import OpenAI
                self._local_client = OpenAI(
                    base_url=local_base,
                    api_key=os.getenv("EVO_MEM_LLM_API_KEY") or os.getenv("OPENAI_API_KEY") or "local",
                    timeout=240.0,
                    max_retries=2,
                )
            else:
                try:
                    from volcenginesdkarkruntime import Ark
                except ImportError as e:
                    raise ImportError(
                        "Set EVO_MEM_LLM_BASE_URL for local inference or install volcengine SDK."
                    ) from e
                self._ark_client = Ark(api_key=ark_batch_config["api_key"])
'''
    replace_one(path, old_constructor, new_constructor, f"{class_name} local constructor")

    if class_name == "AceMemory":
        start = "    def _call_llm(self, messages: list[dict], stats: MemoryCallStats) -> str:"
        end = "\n    def _format_trajectory"
    else:
        start = "    def _call_llm(self, system_prompt: str, user_content: str, stats: MemoryCallStats) -> str:"
        end = "\n    # ------------------------------------------------------------------\n    # Public interface"
    replacement = r'''    def _call_llm(self, messages, stats: MemoryCallStats) -> str:
        """Call the configured local GPU endpoint or the original ARK batch API."""
        if self._local_client is None and self._ark_client is None:
            raise RuntimeError("No memory LLM configured; set EVO_MEM_LLM_BASE_URL.")
        max_retries, retry_delay = 8, 5.0
        for attempt in range(max_retries):
            try:
                if self._local_client is not None:
                    response = self._local_client.chat.completions.create(
                        model=self._batch_model, messages=messages, temperature=0.0
                    )
                else:
                    response = self._ark_client.batch.chat.completions.create(
                        model=self._batch_model, messages=messages
                    )
                if response.usage:
                    stats.input_tokens += response.usage.prompt_tokens or 0
                    stats.output_tokens += response.usage.completion_tokens or 0
                    details = getattr(response.usage, "prompt_tokens_details", None)
                    stats.cached_tokens += getattr(details, "cached_tokens", 0) or 0
                return response.choices[0].message.content or ""
            except Exception as exc:
                if attempt == max_retries - 1:
                    raise
                print(
                    f"[{self.memory_type} LLM retry {attempt + 1}/{max_retries}] {exc}. "
                    f"Retrying in {retry_delay:.1f}s..."
                )
                time.sleep(retry_delay)
'''
    if class_name == "ReasoningBankMemory":
        # Its method has a different public signature; preserve the prompt wrapper.
        replacement = r'''    def _call_llm(self, system_prompt: str, user_content: str, stats: MemoryCallStats) -> str:
        """Call the configured local GPU endpoint or the original ARK batch API."""
        if self._local_client is None and self._ark_client is None:
            raise RuntimeError("No memory LLM configured; set EVO_MEM_LLM_BASE_URL.")
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ]
        max_retries, retry_delay = 8, 5.0
        for attempt in range(max_retries):
            try:
                if self._local_client is not None:
                    response = self._local_client.chat.completions.create(
                        model=self._batch_model, messages=messages, temperature=0.0
                    )
                else:
                    response = self._ark_client.batch.chat.completions.create(
                        model=self._batch_model, messages=messages
                    )
                if response.usage:
                    stats.input_tokens += response.usage.prompt_tokens or 0
                    stats.output_tokens += response.usage.completion_tokens or 0
                    details = getattr(response.usage, "prompt_tokens_details", None)
                    stats.cached_tokens += getattr(details, "cached_tokens", 0) or 0
                return response.choices[0].message.content or ""
            except Exception as exc:
                if attempt == max_retries - 1:
                    raise
                print(
                    f"[ReasoningBank LLM retry {attempt + 1}/{max_retries}] {exc}. "
                    f"Retrying in {retry_delay:.1f}s..."
                )
                time.sleep(retry_delay)
        return ""
'''
    replace_section(path, start, end, replacement + "\n", f"{class_name} local call")


def patch(upstream: Path, adapter: Path) -> None:
    component = upstream / "Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB"
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
    replace_one(evaluator, standard_before, standard_after, "standard worker temperature")

    batch_before = '''                response = self.client.batch.chat.completions.create(
                    model=self.model,
                    messages=messages,
                )'''
    batch_after = '''                request_kwargs = {"model": self.model, "messages": messages}
                temperature = os.environ.get("EVO_MEM_TEMPERATURE")
                if temperature not in (None, ""):
                    request_kwargs["temperature"] = float(temperature)
                response = self.client.batch.chat.completions.create(**request_kwargs)'''
    replace_one(evaluator, batch_before, batch_after, "batch worker temperature")

    helper = '''def _cap_memory_suffix(conversation, original_prompt, budget):
    """Only truncate retrieved memory; never truncate the official task instructions."""
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
        replace_one(evaluator, marker, helper + marker, "memory-only cap helper")

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
    replace_one(evaluator, injection_before, injection_after, "memory-only token cap")

    factory_before = '''    raise ValueError(
        f"Unknown memory_type: {memory_type!r}. "'''
    factory_after = '''    if memory_type == "controlled":
        from .controlled_adapter import ControlledMemory
        return ControlledMemory(read_only=read_only, **kwargs)
    raise ValueError(
        f"Unknown memory_type: {memory_type!r}. "'''
    replace_one(factory, factory_before, factory_after, "controlled memory factory")

    # Add explicit server-side close support. The pinned client already exposes close().
    server_dir = component / "agentenv-alfworld/agentenv_alfworld"
    model = server_dir / "model.py"
    server = server_dir / "server.py"
    wrapper = server_dir / "env_wrapper.py"
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
    replace_one(server, '@app.post("/reset")', close_route + '@app.post("/reset")', "close endpoint")
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
    replace_one(wrapper, "    def get_observation(self, idx: int):", close_method + "    def get_observation(self, idx: int):", "server environment cleanup")
    replace_one(
        client,
        '        response = self._post("reset", {"game": game, "world_type": world_type})\n',
        '        response = self._post("reset", {"game": game, "world_type": world_type})\n'
        '        if "error" in response:\n'
        '            raise RuntimeError("ALFWorld reset game %s failed: %s" % (game, response["error"]))\n',
        "surface original reset error",
    )
    if "    def close(self):" not in client.read_text():
        raise RuntimeError("Pinned ALFWorld client lacks close(); refusing to add unsafe cleanup")
    patch_episode_cleanup(evaluator)

    # Use the same local GPU-served model for memory writers in upstream adapters.
    scripts_memory = component / "scripts/memory"
    patch_local_ark_utils(scripts_memory / "_ark_utils.py")
    patch_direct_ark_adapter(scripts_memory / "ace_adapter.py", "AceMemory")
    patch_direct_ark_adapter(scripts_memory / "reasoning_bank_adapter.py", "ReasoningBankMemory")

    # AgentKB/AWM use the same OpenAI-compatible embedding endpoint, normally a
    # separately served Qwen3-Embedding model on port 8001.
    for provider_name in ("agent_kb_provider.py", "agent_workflow_memory_provider.py"):
        provider = scripts_memory / "_evolvelab/providers" / provider_name
        replace_one(
            provider,
            'DEFAULT_EMBEDDING_MODEL_NAME = "text-embedding-v4"',
            'DEFAULT_EMBEDDING_MODEL_NAME = os.getenv("EVO_MEM_EMBED_MODEL", "text-embedding-v4")',
            f"{provider_name} local embedding model",
        )

    shutil.copy2(adapter, scripts_memory / "controlled_adapter.py")
    (component / ".controlled_patch.json").write_text(json.dumps({
        "patch_version": 2,
        "worker_temperature": True,
        "memory_only_cap": True,
        "episode_cleanup_finally": True,
        "local_memory_llm": True,
        "local_embedding_model": True,
        "custom_adapter": "controlled",
    }, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream", required=True)
    parser.add_argument("--adapter", required=True)
    args = parser.parse_args()
    patch(Path(args.upstream), Path(args.adapter))
    print("Patched pinned evaluator: temperature, memory-only cap, cleanup, and local memory models")


if __name__ == "__main__":
    main()
