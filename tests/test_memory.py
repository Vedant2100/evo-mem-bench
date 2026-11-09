from evomembench.memory import Memory

class Stub:
    def generate(self,system,prompt,max_tokens):
        return "ST-ALPHA TAG-XYZ goes to RILO",{"prompt_tokens":20,"completion_tokens":10}

def make(mode):
    m=Memory(mode,seed=1,top_k=1,budget_chars=1000,worker=Stub())
    m.add("ST-ALPHA","REJECTED: ST-ALPHA TAG-XYZ -> TAVU",False)
    m.add("ST-ALPHA","VERIFIED: ST-ALPHA TAG-XYZ -> RILO",True)
    m.add("ST-BETA","VERIFIED: ST-BETA TAG-ABC -> KASA",True)
    return m
def inject(m):
    return m.inject(station="ST-ALPHA",query="station ST-ALPHA tag TAG-XYZ",
                    oracle="ST-ALPHA TAG-XYZ -> RILO")

def test_none_and_oracle():
    assert inject(make("none"))==""
    assert "RILO" in inject(make("oracle"))

def test_success_only_excludes_failure():
    m=make("success_only")
    assert len(m.notes)==2
    assert all(n.verified for n in m.notes)

def test_reward_labeled_retains_outcome():
    m=make("reward_labeled")
    assert "[REJECTED]" in m._render([m.notes[0]],label=True)
    assert "[VERIFIED]" in m._render([m.notes[1]],label=True)

def test_random_is_irrelevant():
    assert "ST-BETA" in inject(make("random"))
    assert "ST-ALPHA" not in inject(make("random"))

def test_writer_captures_rule_without_future_answer():
    m=make("writer")
    assert "RILO" in inject(m)
    assert len(m.writer_usage)==2

def test_ucb_learns_from_feedback():
    m=make("ucb")
    assert m.arm_count["bm25"]==0
    inject(m)
    m.feedback(True)
    assert m.arm_count["bm25"]==1
    assert m.arm_mean["bm25"]==1

def test_context_cap():
    m=make("full_history")
    assert len(inject(m))<=1000
