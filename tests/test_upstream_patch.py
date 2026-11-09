from pathlib import Path
from patch_upstream import patch


def test_patcher_is_strict_and_idempotent(tmp_path):
    component=tmp_path/"Cross-Episode-Execution/Embodied-AI/CROSSEP-EMB"
    scripts=component/"scripts"
    memory=scripts/"memory"
    memory.mkdir(parents=True)
    evaluator=scripts/"eval_alfworld_with_memory.py"
    evaluator.write_text(
        'import os\nclass A:\n'
        '    def f(self, messages):\n'
        '                response = self.client.chat.completions.create(\n'
        '                    model=self.model,\n'
        '                    messages=messages,\n'
        '                )\n'
        '                response = self.client.batch.chat.completions.create(\n'
        '                    model=self.model,\n'
        '                    messages=messages,\n'
        '                )\n'
        'def eval_one(data_idx, env_args, agent, max_rounds, memory):\n'
        '    if memory is not None:\n'
        '        conversation, inject_stats = memory.inject(conversation)\n'
        '    return data_idx, conversation, reward, total_pt, total_ct, total_cached, sample_latency, total_llm_latency, rounds, inject_stats, update_stats\n'
    )
    factory=memory/"base.py"
    factory.write_text(
        'def create_memory(memory_type, read_only=False, **kwargs):\n'
        '    raise ValueError(\n'
        '        f"Unknown memory_type: {memory_type!r}. "\n'
    )
    serverpkg=component/"agentenv-alfworld"/"agentenv_alfworld"
    serverpkg.mkdir(parents=True)
    (serverpkg/"model.py").write_text("class StepRequestBody(BaseModel):\n    id: int\n")
    (serverpkg/"server.py").write_text('@app.post("/reset")\ndef reset(): pass\n')
    (serverpkg/"env_wrapper.py").write_text(
        "class Wrapper:\n"
        "    def get_observation(self, idx: int):\n        pass\n"
    )
    clientpkg=component/"agentenv"/"agentenv"/"envs"
    clientpkg.mkdir(parents=True)
    (clientpkg/"alfworld.py").write_text(
        "class Client:\n"
        '    def reset(self):\n'
        '        response = self._post("reset", {"game": game, "world_type": world_type})\n'
    )
    adapter=tmp_path/"adapter.py"
    adapter.write_text("class ControlledMemory: pass\n")
    patch(tmp_path,adapter)
    first=evaluator.read_text()
    patch(tmp_path,adapter)
    assert evaluator.read_text()==first
    assert "_cap_memory_suffix" in first
    assert 'memory_type == "controlled"' in factory.read_text()
    assert (memory/"controlled_adapter.py").exists()
    assert '/close' in (serverpkg/"server.py").read_text()
    assert "response[\"error\"]" in (clientpkg/"alfworld.py").read_text()
