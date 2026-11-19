"""Paired analysis, coverage accounting and blind language audit exports."""
from pathlib import Path
import csv
import numpy as np
import pandas as pd
from .common import load_json, read_jsonl, save_json, digest
from .experiment import phase_dir
from .data import selected

def paired_effect(rows, positive="B", negative="A", language="sv", split="test", replicates=10000):
    table = {}
    for r in rows:
        if r["language"]==language and r["split"]==split and r["condition"] in {positive,negative}:
            key = r["id"]
            if r["condition"] in table.setdefault(key,{}): raise ValueError("Duplicate paired outcome")
            table[key][r["condition"]]=r
    pairs=[v for v in table.values() if positive in v and negative in v]
    if not pairs: return {"contrast":positive+"-"+negative,"language":language,"split":split,"n":0}
    family=[]; differences=[]; transitions={"both_wrong":0,"rescued":0,"regressed":0,"both_correct":0}
    for pair in pairs:
        a,b=pair[negative],pair[positive]
        x,y=int(a["endpoint"]["correct_completed"]),int(b["endpoint"]["correct_completed"])
        differences.append(y-x); family.append(a["template_family_id"])
        transitions[{(0,0):"both_wrong",(0,1):"rescued",(1,0):"regressed",(1,1):"both_correct"}[(x,y)]]+=1
    groups=sorted(set(family)); d=np.array(differences); f=np.array(family)
    sums=np.array([d[f==g].sum() for g in groups]); counts=np.array([(f==g).sum() for g in groups])
    rng=np.random.default_rng(20261118)
    draws=rng.integers(0,len(groups),size=(replicates,len(groups)))
    bootstrap=sums[draws].sum(axis=1)/counts[draws].sum(axis=1)
    low,high=np.quantile(bootstrap,[0.025,0.975])
    return {"contrast":positive+"-"+negative,"language":language,"split":split,"n":len(pairs),
            "families":len(groups),"effect_pp":float(d.mean()*100),"low_pp":float(low*100),"high_pp":float(high*100),
            "discordant_concepts":int(np.count_nonzero(d)),
            "discordant_families":int(sum(np.any(d[f==g]!=0) for g in groups)),
            "degenerate_interval":bool(low==high),"missing_one_arm":len(table)-len(pairs),**transitions}

def analyze(cfg):
    folder=phase_dir(cfg)
    rows=read_jsonl(folder/"predictions.jsonl")
    if not rows: raise ValueError("No real predictions; run a development pilot or test first")
    if len({r["key"] for r in rows})!=len(rows): raise ValueError("Duplicate results")
    expected=selected(read_jsonl(Path(cfg["root"])/"data/tasks.jsonl"),cfg["phase"],cfg["pairs"],cfg["dev_pairs"])
    expected_keys={(a,r["id"],r["language"]) for a in cfg["arms"] for r in expected}
    observed={(r["condition"],r["id"],r["language"]) for r in rows}
    if not observed.issubset(expected_keys): raise ValueError("Unexpected condition/example in results")
    missing=sorted(expected_keys-observed)
    save_json(folder/"coverage.json",{"expected":len(expected_keys),"completed":len(observed),"missing":missing,
             "complete":not missing,"phase":cfg["phase"],"scope":cfg["scope"]})
    records=[]; curves=[]
    for r in rows:
        records.append({"condition":r["condition"],"language":r["language"],"split":r["split"],
            "correct_completed":r["endpoint"]["correct_completed"],"completed":r["endpoint"]["completed"],
            "answer_present":r["endpoint"]["answer_present"],"cap_hit":r["cap_hit"],
            "tokens":r["tokens"],"repetitive":r["repetitive"]})
        for budget,s in r["budgets"].items():
            curves.append({"condition":r["condition"],"language":r["language"],"split":r["split"],
                           "budget":int(budget),"correct_completed":s["correct_completed"],"completed":s["completed"]})
    df=pd.DataFrame(records)
    summary=df.groupby(["split","language","condition"],sort=False).agg(
        n=("tokens","size"),correct_completed=("correct_completed","mean"),completed=("completed","mean"),
        answer_present=("answer_present","mean"),cap_hit=("cap_hit","mean"),mean_tokens=("tokens","mean"),
        repetition_rate=("repetitive","mean")).reset_index()
    summary.to_csv(folder/"summary.csv",index=False)
    if curves:
        pd.DataFrame(curves).groupby(["split","language","condition","budget"]).agg(
            n=("completed","size"),correct_completed=("correct_completed","mean"),completed=("completed","mean")).reset_index().to_csv(folder/"budget_curves.csv",index=False)
    split="test" if cfg["phase"]=="test" else "dev"
    effects=[paired_effect(rows,p,n,language,split) for language in ["sv","en"]
             for p,n in [("B","A"),("B","C"),("B","C_pen"),("C_pen","A"),("D","C"),("B","R1"),("B","S1")]]
    effects += [paired_effect(rows,"D","C",language,"regression") for language in ["sv","en"]]
    pd.DataFrame(effects).to_csv(folder/"paired_effects.csv",index=False)
    # Rescue overlap is descriptive, not mediation evidence.
    indexed={(r["condition"],r["id"],r["language"]):r for r in rows}
    rescues=[]
    for r in rows:
        a=indexed.get(("A",r["id"],r["language"]))
        if r["condition"]=="B" and r["split"]==split and a and r["endpoint"]["correct_completed"] and not a["endpoint"]["correct_completed"]:
            rescues.append({"id":r["id"],"language":r["language"],"original_truncated":a["cap_hit"],
                           "original_repetitive":a["repetitive"],"original_answer_present":a["endpoint"]["answer_present"]})
    save_json(folder/"rescues.json",rescues)
    export_language_audit(rows,folder)
    plot_results(folder,split)
    return summary, pd.DataFrame(effects), load_json(folder/"coverage.json")

def export_language_audit(rows,folder):
    candidates=[]
    for r in rows:
        for view,text in r["language_views"].items():
            candidates.append({"audit_id":digest([r["key"],view])[:16],"view":view,"text":text,
                               "automatic_label":r["language_labels"][view]["label"],
                               "condition":r["condition"],"id":r["id"]})
    # Deterministic selection balanced across views, blinded to condition and detector output.
    chosen=[]
    for view in ["prefix250","reasoning","final"]:
        pool=sorted([r for r in candidates if r["view"]==view],key=lambda x:x["audit_id"])
        chosen+=pool[:14 if view!="final" else 12]
    pd.DataFrame(candidates).to_csv(folder/"language_automatic_UNVALIDATED.csv",index=False)
    pd.DataFrame(chosen).to_csv(folder/"language_audit_key.csv",index=False)
    target=folder/"language_blind_review.csv"
    if not target.exists():
        pd.DataFrame([{k:r[k] for k in ["audit_id","view","text"]}|{"manual_label":""} for r in chosen]).to_csv(target,index=False)

def validate_language_audit(folder):
    folder=Path(folder)
    labels=pd.read_csv(folder/"language_blind_review.csv",keep_default_na=False)
    key=pd.read_csv(folder/"language_audit_key.csv",keep_default_na=False)
    allowed={"sv","en","mixed/other","insufficient"}
    if not labels.manual_label.isin(allowed).all(): raise ValueError("Fill all blind labels before validation")
    if labels.audit_id.duplicated().any(): raise ValueError("Duplicate audit IDs")
    merged=key.merge(labels[["audit_id","manual_label"]],on="audit_id",validate="one_to_one")
    if len(merged)!=len(key): raise ValueError("Missing audit labels")
    confusion=pd.crosstab(merged.manual_label,merged.automatic_label)
    confusion.to_csv(folder/"language_confusion.csv")
    result={"n":len(merged),"agreement":float((merged.manual_label==merged.automatic_label).mean()),
            "interpretation":"Descriptive sample validation, not proof of general detector reliability; report class-specific errors."}
    save_json(folder/"language_validation.json",result)
    return result,confusion

def plot_results(folder,split):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size":11,"axes.spines.top":False,"axes.spines.right":False,"figure.dpi":130})
    summary=pd.read_csv(folder/"summary.csv")
    table=summary[(summary.split==split)&(summary.language=="sv")]
    fig,ax=plt.subplots(figsize=(8,4))
    ax.bar(table.condition,100*table.correct_completed,color="#176B87")
    ax.set(ylim=(0,100),ylabel="Correct completed responses (%)",xlabel="Condition",
           title=f"Swedish {split} tasks: bounded success (observed rows)")
    for i,row in enumerate(table.itertuples()): ax.text(i,100*row.correct_completed+2,f"n={row.n}",ha="center",fontsize=9)
    ax.grid(axis="y",alpha=.2); fig.tight_layout(); fig.savefig(folder/"swedish_success.png"); plt.close(fig)
    curves_path=folder/"budget_curves.csv"
    if curves_path.exists():
        curves=pd.read_csv(curves_path); curves=curves[(curves.split==split)&(curves.language=="sv")]
        fig,ax=plt.subplots(figsize=(8,4))
        for arm,table in curves.groupby("condition",sort=False):
            ax.plot(table.budget,100*table.correct_completed,marker="o",label=arm)
        ax.set(xlabel="Output-token budget (shared trajectory)",ylabel="Correct completed responses (%)",ylim=(0,100),
               title="Completion budget curves; not wall-clock latency")
        ax.legend(ncol=4);ax.grid(alpha=.2);fig.tight_layout();fig.savefig(folder/"budget_curves.png");plt.close(fig)

def write_report(cfg):
    """Generate an evidence-backed draft; interpretation remains an explicit author step."""
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    from xml.sax.saxutils import escape
    folder=phase_dir(cfg)
    effects=pd.read_csv(folder/"paired_effects.csv")
    coverage=load_json(folder/"coverage.json")
    primary=effects[(effects.contrast=="B-A")&(effects.language=="sv")]
    if primary.empty or primary.iloc[0]["n"]==0: finding="The primary matched comparison is unavailable."
    else:
        r=primary.iloc[0]
        finding=f"Swedish B-A: {r.effect_pp:.1f} percentage points, family-bootstrap 95% interval [{r.low_pp:.1f}, {r.high_pp:.1f}], n={int(r.n)} matched concepts."
        if r.degenerate_interval: finding+=" The bootstrap interval is degenerate and must not be interpreted as evidence of equivalence."
    paragraphs=[
      ("SE.LLMA weight-edit transfer: evidence draft", "This is an automatically assembled draft. Review the results and write the recommendation before submission."),
      ("Question and critique", "Does the published p50c edit improve completed correct answers on Swedish arithmetic within a fixed token budget? The source's 10/25/50 forward-pass budgets produced separate edits, not repeated evaluations of one intervention. The p50c editor reported that 576 of 579 practice rescues were originally truncated; its random-control evidence was reported on practice data. These observations motivate explicit stopping, decoding and random-control comparisons, without proving a unique mechanism."),
      ("Method",f"Phase: {cfg['phase']}; scope: {cfg['scope']}; cap: {cfg['cap']}; conditions: {', '.join(cfg['arms'])}; engine: {cfg['engine']}; dtype: {cfg['dtype']}. Original synthetic Swedish/English pairs use disjoint development/test wording families. Each pair shares an integer answer. Primary success requires natural completion and one valid final answer marker. D-C with thinking off is the primary regression check. The arithmetic families remain similar and this small study is not a Swedish proficiency benchmark."),
      ("Observed coverage and primary result",f"Completed {coverage['completed']} of {coverage['expected']} planned requests. {finding} Unmatched requests are disclosed in coverage.json. Development findings must not be described as held-out results."),
      ("Recommendation to complete", "Author action: interpret the primary effect alongside B-C and B-C_pen, regression results, missingness, token use and uncertainty. State whether to continue targeted repair, prefer simpler inference settings, or conduct a more diagnostic follow-up. Do not infer equivalence from a nonsignificant difference."),
      ("Limitations and reproducibility", "One checkpoint, one decoding seed, two random-control draws and synthetic tasks limit generalization. The automatic language screen is unvalidated until the blind sample is labeled and its confusion table reviewed. See paired_effects.csv, summary.csv, predictions.jsonl, frozen configuration, environment lock and weight manifest. Source: RikiyaT/lm_understand-logs, revision 4d09ba6d6a7e891dd8063974dfe81b3d78e3494f; base revision 2fc06364715b967f1860aea9cf38778875588b17. AI assistance is disclosed in AI_USAGE.md. Record actual effort, including preparation.")]
    (folder/"research_note_DRAFT.md").write_text("\n\n".join("## "+h+"\n\n"+p for h,p in paragraphs),encoding="utf-8")
    styles=getSampleStyleSheet(); story=[]
    for heading,text in paragraphs:
        story.extend([Paragraph(escape(heading),styles["Heading2"]),Paragraph(escape(text),styles["BodyText"]),Spacer(1,8)])
    SimpleDocTemplate(str(folder/"research_note_DRAFT.pdf"),pagesize=A4).build(story)
    return folder/"research_note_DRAFT.md"
