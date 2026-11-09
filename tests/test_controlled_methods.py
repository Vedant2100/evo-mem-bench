import json

from evo_mem_bench.control_memory import ControlledMemory
from evo_mem_bench.control_memory import MemoryCallStats


def conversation(task="Your task is to: clean the mug and put it on the shelf", action="go to sink"):
    return [
        {"role": "user", "content": "You are a household assistant."},
        {"role": "assistant", "content": "Understood."},
        {"role": "user", "content": task},
        {"role": "assistant", "content": action},
        {"role": "user", "content": "The action succeeded."},
        {"role": "assistant", "content": "take mug from sink"},
    ]


class FakeWriter:
    def __init__(self, text):
        self.text = text
        self.calls = 0

    def complete(self, system, user, max_tokens=256):
        self.calls += 1
        assert "Trajectory" in user or "completed episodes" in user.lower()
        return self.text, MemoryCallStats(input_tokens=17, output_tokens=8, latency=0.01)


def install_fake_writer(memory, text):
    fake = FakeWriter(text)
    memory._get_writer = lambda: fake
    return fake


def test_full_history_keeps_multiple_prior_episodes_and_respects_cap(tmp_path):
    memory = ControlledMemory(
        mode="full_history", memory_dir=str(tmp_path), max_context_tokens=512
    )
    memory.update(conversation("Your task is to: clean the mug"), 1, 1.0)
    memory.update(conversation("Your task is to: cool the apple"), 2, 0.0)
    injected, _ = memory.inject(conversation("Your task is to: clean the plate"))
    assert "clean the mug" in injected[0]["content"]
    assert "cool the apple" in injected[0]["content"]


def test_contextual_bandit_updates_and_persists_linucb_state(tmp_path):
    memory = ControlledMemory(mode="contextual_bandit", memory_dir=str(tmp_path), random_seed=7)
    memory.inject(conversation())
    assert memory.last_arm == "bm25"
    memory.update(conversation(), 1, 1.0)
    assert memory.counts["bm25"] == 1
    assert memory.lin_a["bm25"].trace() > memory.CONTEXT_DIM

    restored = ControlledMemory(mode="contextual_bandit", memory_dir=str(tmp_path), random_seed=7)
    assert restored.counts["bm25"] == 1
    assert restored.values["bm25"] == 1.0
    assert restored.lin_a["bm25"].trace() == memory.lin_a["bm25"].trace()


def test_bayesian_state_uses_only_past_reward_and_successful_procedure(tmp_path):
    memory = ControlledMemory(mode="bayesian_state", memory_dir=str(tmp_path))
    memory.update(conversation(action="clean mug with sink"), 1, 1.0)
    memory.update(conversation(action="invalid action"), 2, 0.0)
    injected, _ = memory.inject(conversation("Your task is to: clean a plate"))
    text = injected[0]["content"]
    assert "Beta(2.0, 2.0)" in text
    assert "clean mug with sink" in text


def test_reward_supervised_writer_stores_only_successful_grounded_lessons(tmp_path):
    memory = ControlledMemory(mode="supervised_writer", memory_dir=str(tmp_path))
    fake = install_fake_writer(memory, "Use the sink to clean the mug before placing it.")
    memory.update(conversation(), 1, 1.0)
    memory.update(conversation(), 2, 0.0)
    assert fake.calls == 1
    assert len(memory.entries) == 1
    injected, _ = memory.inject(conversation("Your task is to: clean a plate"))
    assert "Use the sink" in injected[0]["content"]


def test_reflexion_and_expel_have_distinct_update_state(tmp_path):
    reflexion = ControlledMemory(mode="reflexion", memory_dir=str(tmp_path / "reflexion"))
    install_fake_writer(reflexion, "Do not use an invalid receptacle; verify the available actions.")
    reflexion.update(conversation(), 1, 0.0)
    assert reflexion.entries[0]["outcome"] == "failure"
    assert "invalid receptacle" in reflexion.entries[0]["memory_text"]

    expel = ControlledMemory(mode="expel", memory_dir=str(tmp_path / "expel"))
    install_fake_writer(expel, "Clean the object before moving it to the destination.")
    expel.update(conversation(), 1, 1.0)
    assert expel.principles == ["Clean the object before moving it to the destination."]
    assert json.loads((tmp_path / "expel" / "expel_principles.json").read_text()) == expel.principles


def test_letta_style_consolidation_is_batched_and_persistent(tmp_path):
    memory = ControlledMemory(
        mode="letta_consolidation", memory_dir=str(tmp_path), consolidate_every=2
    )
    fake = install_fake_writer(memory, "- Clean the object before placing it.")
    memory.update(conversation(), 1, 1.0)
    assert fake.calls == 0
    memory.update(conversation("Your task is to: clean a plate"), 2, 1.0)
    assert fake.calls == 1
    assert memory.semantic_notes == ["Clean the object before placing it."]
    assert all(row.get("consolidated") for row in memory.entries)

    restored = ControlledMemory(
        mode="letta_consolidation", memory_dir=str(tmp_path), consolidate_every=2
    )
    injected, _ = restored.inject(conversation("Your task is to: clean a bowl"))
    assert "Clean the object before placing it." in injected[0]["content"]


def test_readonly_memory_does_not_mutate_bank_or_policy(tmp_path):
    writable = ControlledMemory(mode="contextual_bandit", memory_dir=str(tmp_path))
    writable.inject(conversation())
    writable.update(conversation(), 1, 1.0)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}

    readonly = ControlledMemory(mode="contextual_bandit", memory_dir=str(tmp_path), read_only=True)
    readonly.inject(conversation())
    readonly.update(conversation(), 2, 1.0)
    after = {p.name: p.read_bytes() for p in tmp_path.iterdir() if p.is_file()}
    assert before == after
