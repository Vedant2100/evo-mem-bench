from collections import Counter

from evomembench.hidden_rules import GATES, check_disjoint, generate, manifest_hash
from evomembench.local_model import normalize_answer
from evomembench.experiment import competence_summary, discovery_stream

def test_worlds_reproduce():
    a=generate(13,36);b=generate(13,36)
    assert a==b
    assert manifest_hash(a)==manifest_hash(b)
    check_disjoint(a)
    assert Counter(w.gate for w in a)=={g:6 for g in GATES}

def test_no_answer_in_current_prompt():
    worlds=generate(14,36)
    for w in worlds:
        assert w.gate not in w.task.split("Select the correct gate")[0]
        assert w.train_parcel not in w.task
        assert w.station in w.oracle and w.gate in w.oracle
        assert w.station in w.discovery

def test_rejected_and_verified_events():
    worlds=generate(3,12)
    events=discovery_stream(worlds,5)
    assert len(events)==24
    assert sum(verified for _,_,verified in events)==12
    for i in range(0,len(events),2):
        assert events[i][0]==events[i+1][0]
        assert not events[i][2] and events[i+1][2]

def test_parser_same_for_all_conditions():
    assert normalize_answer("Gate: RILO.")=="RILO"
    assert normalize_answer("no idea") is None

def test_headroom_gate():
    d={"none":{"n":36,"success_rate":.2},"oracle":{"n":36,"success_rate":.9}}
    assert competence_summary(d)["memory_headroom_confirmed"]
    d["oracle"]["success_rate"]=.3
    assert not competence_summary(d)["memory_headroom_confirmed"]
