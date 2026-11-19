import json
from pathlib import Path
import numpy as np
import pytest
from sellma_wsp.common import append_jsonl, completed_records, record_key, request_seed, digest
from sellma_wsp.data import build_rows, validate, selected
from sellma_wsp.scoring import score, extract_answer, language_label
from sellma_wsp.weights import random_delta, upstream
from sellma_wsp.analysis import paired_effect

def test_split_balance_and_gold():
    rows=build_rows(); assert validate(rows)["rows"]==116
    assert len([r for r in rows if r["split"]=="test"])==80
    for n in [16,24,40]:
        subset=[r for r in selected(rows,"test",n) if r["split"]=="test" and r["language"]=="sv"]
        assert len(subset)==n
        assert {sum(r["category"]==op for r in subset) for op in ["addition","subtraction","multiplication","division"]}=={n//4}
    broken=[dict(r) for r in rows]; broken[0]["gold"]="999"
    with pytest.raises(ValueError): validate(broken)

@pytest.mark.parametrize("text,thinking,finish,count,budget,expected",[
    ("The answer is 19. Svar: 19",True,"length",8192,8192,False),
    ("<think>19</think>\nSvar: 19",True,"stop",30,30,True),
    ("Svar: 19",False,"stop",5,5,True),
    ("Svar: 19",False,"length",5,5,False),
    ("</think>\nSvar: 19",True,"stop",30,20,False),
    ("</think>\nSvar: 19.5",True,"stop",30,30,False),
    ("</think>\nSvar: 19\nAnswer: 20",True,"stop",30,30,False),
    ("</think>\nSvar: 19\n<|im_end|>",True,"stop",30,30,True),
])
def test_scoring_boundaries(text,thinking,finish,count,budget,expected):
    assert score(text,thinking,"19","integer",finish,count,budget)["correct_completed"] is expected

@pytest.mark.parametrize("text,kind,expected",[
    ("**Svar:** 37","integer","37"), ("Svar: 37.","integer","37"), ("**Answer: 37**","integer","37"),
    ("Answer: `37`","integer","37"), ("Svar: **37**","integer","37"), ("Answer: regn.","text","regn"),
    ('Answer: "katt"',"text","katt"), ("Svar: 37 kronor","integer",None), ("Svar: 3.7","integer",None),
])
def test_answer_formatting_is_normalized_but_content_is_not(text,kind,expected):
    assert extract_answer(text,kind)==expected

def test_text_and_negative_scoring():
    assert extract_answer("Svar: −12")=="-12"
    assert extract_answer("Answer: AB-42","text")=="ab-42"
    assert extract_answer("19") is None
    assert extract_answer("Svar: 19 kronor") is None
    assert language_label("19")["label"]=="insufficient"

def test_random_control_geometry_and_rounding():
    rng=np.random.default_rng(3); delta=rng.normal(size=(32,8))
    sign=random_delta(delta,"sign",101)
    assert np.array_equal(np.abs(sign),np.abs(delta))
    orth=random_delta(delta,"orthogonal",201)
    assert np.linalg.matrix_rank(orth)==1
    assert np.linalg.norm(orth)==pytest.approx(np.linalg.norm(delta))
    assert np.sum(delta*orth)==pytest.approx(0,abs=1e-10)
    assert np.array_equal(orth,random_delta(delta,"orthogonal",201))
    S=upstream(Path(__file__).resolve().parents[1])
    # Rounding and upstream round-trip are tested without downloading model weights.
    base=np.array([1.,-2.,0.,0.5],dtype=np.float32)
    assert np.array_equal(S.bf16_to_f32(S.bf16_bits(base)),base)

def test_resume_conflicts_and_truncated_record(tmp_path):
    path=tmp_path/"predictions.jsonl"
    append_jsonl(path,[{"key":"a","value":1}]); done=completed_records(path)
    assert list(done)==["a"]
    # Existing completed keys are skipped; a second result for the same key is rejected.
    pending=[r for r in [{"key":"a"},{"key":"b"}] if r["key"] not in done]
    assert pending==[{"key":"b"}]
    append_jsonl(path,[{"key":"a","value":2}])
    with pytest.raises(ValueError): completed_records(path)
    path.write_text('{"key":',encoding="utf-8")
    with pytest.raises(ValueError,match="Incomplete/corrupt"): completed_records(path)

def test_keys_bind_config_dataset_arm_seed():
    r={"id":"test-1","language":"sv"}
    assert record_key("a","b","A",r,17)!=record_key("a","b","B",r,17)
    assert record_key("a","b","A",r,17)!=record_key("a","c","A",r,17)
    assert request_seed("x","sv")==request_seed("x","sv")

def test_paired_missingness_and_degenerate_intervals():
    rows=[]
    for i in range(4):
        for arm in ["A","B"]:
            rows.append({"id":str(i),"template_family_id":str(i//2),"condition":arm,
                         "language":"sv","split":"test","endpoint":{"correct_completed":False}})
    result=paired_effect(rows,replicates=100)
    assert result["n"]==4 and result["families"]==2
    assert result["degenerate_interval"] and result["discordant_families"]==0
    rows[-1]["endpoint"]["correct_completed"]=True
    result=paired_effect(rows,replicates=100)
    assert result["effect_pp"]==25 and result["rescued"]==1
    rows.pop()
    result=paired_effect(rows,replicates=100)
    assert result["n"]==3 and result["missing_one_arm"]==1

def test_analysis_exports_partial_coverage(tmp_path):
    # Hand-specified software fixture, never presented as an LLM experiment.
    from sellma_wsp.common import save_json
    from sellma_wsp.analysis import analyze
    from sellma_wsp.experiment import config, phase_dir
    root=Path(__file__).resolve().parents[1]
    cfg=config(root,tmp_path,tmp_path/'models',phase='pilot',cap=8192,arms=['A','B'])
    folder=phase_dir(cfg)
    examples=selected(build_rows(),'pilot',dev_pairs=4)
    fixture=[]
    for arm in ['A','B']:
        for item in examples:
            endpoint={'correct_completed':True,'completed':True,'answer_present':True}
            fixture.append({**item,'key':digest([arm,item['id'],item['language']]),'condition':arm,
                'endpoint':endpoint,'tokens':8,'cap_hit':False,'repetitive':False,
                'budgets':{'8192':endpoint},'language_views':{},'language_labels':{}})
    fixture.pop()  # Ensure missing requests remain visible, not silently treated as failures.
    append_jsonl(folder/'predictions.jsonl',fixture)
    summary,effects,coverage=analyze(cfg)
    assert coverage['expected']==16 and coverage['completed']==15 and len(coverage['missing'])==1
    assert (folder/'summary.csv').is_file() and (folder/'budget_curves.png').is_file()
    assert effects.query("contrast == 'B-A' and language == 'en'").iloc[0]['n']==3
