from pathlib import Path

from patch_upstream import patch


def _ark_adapter(class_name, call_signature, call_end):
    constructor = (
        "        # ARK batch API client\n"
        "        self._ark_client = None\n"
        "        self._batch_model = None\n"
        "        if ark_batch_config:\n"
        "            try:\n"
        "                from volcenginesdkarkruntime import Ark\n"
        "            except ImportError as e:\n"
        "                raise ImportError(\n"
        "                    \"ark_batch_config requires volcengine SDK. Run:\\n\"\n"
        "                    \"  pip install 'volcengine-python-sdk[ark]'\"\n"
        "                ) from e\n"
        "            self._ark_client = Ark(api_key=ark_batch_config[\"api_key\"])\n"
        "            self._batch_model = ark_batch_config[\"model\"]\n"
    )
    return (
        "import os\n"
        "class " + class_name + ":\n"
        "    def __init__(self, ark_batch_config=None):\n"
        + constructor
        + call_signature + "\n"
        "        pass\n"
        + call_end
    )


def test_patcher_is_strict_idempotent_and_cluster_local(tmp_path):
    component = tmp_path / "Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB"
    scripts = component / "scripts"
    memory = scripts / "memory"
    providers = memory / "_evolvelab/providers"
    memory.mkdir(parents=True)
    providers.mkdir(parents=True)

    evaluator = scripts / "eval_alfworld_with_memory.py"
    evaluator.write_text(
        "import os\n"
        "class A:\n"
        "    def f(self, messages):\n"
        "                response = self.client.chat.completions.create(\n"
        "                    model=self.model,\n"
        "                    messages=messages,\n"
        "                )\n"
        "                response = self.client.batch.chat.completions.create(\n"
        "                    model=self.model,\n"
        "                    messages=messages,\n"
        "                )\n"
        "def eval_one(data_idx, env_args, agent, max_rounds, memory):\n"
        "    sample_t0 = 0\n"
        "    client = AlfWorldEnvClient(**env_args)\n"
        "    client.reset(data_idx)\n"
        "    conversation = [{\"role\":\"user\",\"content\":\"task\"}]\n"
        "    if memory is not None:\n"
        "        conversation, inject_stats = memory.inject(conversation)\n"
        "    return data_idx, conversation, reward, total_pt, total_ct, total_cached, sample_latency, total_llm_latency, rounds, inject_stats, update_stats\n"
        "\n"
        "def write_summary_txt():\n"
        "    pass\n"
    )
    factory = memory / "base.py"
    factory.write_text(
        "def create_memory(memory_type, read_only=False, **kwargs):\n"
        "    raise ValueError(\n"
        "        f\"Unknown memory_type: {memory_type!r}. \"\n"
    )
    (memory / "_ark_utils.py").write_text(
        "class ArkBatchLLM:\n"
        "    pass\n"
        "\n"
        "def patch_embedding_model(embedding_model):\n"
        "    pass\n"
    )
    (memory / "ace_adapter.py").write_text(
        _ark_adapter(
            "AceMemory",
            "    def _call_llm(self, messages: list[dict], stats: MemoryCallStats) -> str:",
            "    def _format_trajectory(self, conversation):\n        pass\n",
        )
    )
    (memory / "reasoning_bank_adapter.py").write_text(
        _ark_adapter(
            "ReasoningBankMemory",
            "    def _call_llm(self, system_prompt: str, user_content: str, stats: MemoryCallStats) -> str:",
            "    # ------------------------------------------------------------------\n"
            "    # Public interface\n",
        )
    )
    for provider in ("agent_kb_provider.py", "agent_workflow_memory_provider.py"):
        (providers / provider).write_text(
            "import os\nDEFAULT_EMBEDDING_MODEL_NAME = \"text-embedding-v4\"\n"
        )

    serverpkg = component / "agentenv-alfworld/agentenv_alfworld"
    serverpkg.mkdir(parents=True)
    (serverpkg / "model.py").write_text("class StepRequestBody(BaseModel):\n    id: int\n")
    (serverpkg / "server.py").write_text('@app.post("/reset")\ndef reset(): pass\n')
    (serverpkg / "env_wrapper.py").write_text(
        "class Wrapper:\n"
        "    def get_observation(self, idx: int):\n"
        "        pass\n"
    )
    clientpkg = component / "agentenv/agentenv/envs"
    clientpkg.mkdir(parents=True)
    (clientpkg / "alfworld.py").write_text(
        "class Client:\n"
        "    def reset(self, game, world_type):\n"
        '        response = self._post("reset", {"game": game, "world_type": world_type})\n'
        "    def close(self):\n"
        "        return self._post('close', {})\n"
    )
    adapter = tmp_path / "adapter.py"
    adapter.write_text("class ControlledMemory: pass\n")

    patch(tmp_path, adapter)
    first = evaluator.read_text()
    patch(tmp_path, adapter)

    assert evaluator.read_text() == first
    assert "_cap_memory_suffix" in first
    assert "finally:" in first
    assert "client.close()" in first
    assert 'memory_type == "controlled"' in factory.read_text()
    assert (memory / "controlled_adapter.py").exists()
    assert "EVO_MEM_LLM_BASE_URL" in (memory / "_ark_utils.py").read_text()
    assert "EVO_MEM_LLM_BASE_URL" in (memory / "ace_adapter.py").read_text()
    assert "EVO_MEM_LLM_BASE_URL" in (memory / "reasoning_bank_adapter.py").read_text()
    assert "EVO_MEM_EMBED_MODEL" in (providers / "agent_kb_provider.py").read_text()
    assert "EVO_MEM_EMBED_MODEL" in (providers / "agent_workflow_memory_provider.py").read_text()
    assert '/close' in (serverpkg / "server.py").read_text()
    assert "response[\"error\"]" in (clientpkg / "alfworld.py").read_text()
