"""Conservative scoring: a number inside unfinished reasoning never counts."""
import re
from collections import Counter

def sections(text, thinking):
    if thinking:
        if "</think>" not in text:
            return text.replace("<think>", ""), "", False
        reasoning, final = text.split("</think>", 1)
        return reasoning.replace("<think>", ""), final, True
    if "</think>" in text:
        reasoning, final = text.split("</think>", 1)
        return reasoning.replace("<think>", ""), final, True
    return "", text, True

FORMATTING = re.compile(r"[*_`]+")          # markdown emphasis/code marks, e.g. **Svar:** 37
WRAPPERS = "\"'“”‘’«»()[] "
TRAILING = ".,;:!?。 "

def extract_answer(final, answer_type="integer"):
    final = re.sub(r"<\|[^>]+\|>", "", final)
    final = FORMATTING.sub("", final).strip()
    markers = re.findall(r"(?im)^\s*(?:Svar|Answer)\s*:\s*(.*?)\s*$", final)
    if len(markers) != 1:
        return None
    # Formatting is normalized; content is not (units or explanations still fail).
    value = markers[0].strip().rstrip(TRAILING).strip(WRAPPERS).rstrip(TRAILING)
    if answer_type == "integer":
        # Full match disallows decimal fragments, units and trailing explanations.
        if not re.fullmatch(r"[+\-−]?\d+", value):
            return None
        return str(int(value.replace("−", "-")))
    return " ".join(value.casefold().split()) if value else None

def score(text, thinking, gold, answer_type, finish_reason, count, budget):
    _, final, closed = sections(text, thinking)
    answer = extract_answer(final, answer_type) if closed else None
    if count > budget and final and not final.endswith(('\n', '\r')):
        # A prefix ending mid-answer may be a fragment of a longer numeric token sequence.
        answer = None
    expected = str(gold) if answer_type == "integer" else " ".join(str(gold).casefold().split())
    complete = finish_reason == "stop" and count <= budget
    return {"answer": answer, "answer_present": answer is not None,
            "correct_answer_present": answer == expected,
            "completed": complete, "correct_completed": complete and answer == expected,
            "thinking_closed": closed}

def repetition(token_ids, special_ids=()):
    excluded = set(special_ids)
    ids = [x for x in token_ids if x not in excluded]
    grams = [tuple(ids[i:i+4]) for i in range(max(0, len(ids)-3))]
    fraction = 1 - len(set(grams)) / len(grams) if grams else 0.0
    return {"repetition_fraction": fraction,
            "repetitive": len(ids) >= 64 and fraction > 0.5}

# Conservative heuristic, deliberately not advertised as a validated detector.
SV = set("och att det är som en ett jag vi har hade får fick till från med på för av inte om så nu hur många kvar varje sedan blir måste alltså därför svar totalt böcker äpplen kronor".split())
EN = set("the and that is are was were have has had gets got from with for of not if so now how many left each then becomes must therefore answer total books apples crowns this we need let's".split())
SV -= EN
EN -= SV

def language_label(text):
    words = re.findall(r"[^\W\d_]+", text.casefold(), flags=re.UNICODE)
    if len(words) < 5:
        return {"label": "insufficient", "sv_hits": 0, "en_hits": 0}
    # Unique evidence prevents one repeated stopword dominating the detector.
    counts = Counter(words)
    sv, en = len(set(counts) & SV), len(set(counts) & EN)
    if sv + en < 3:
        label = "mixed/other"
    elif sv >= 2 and en >= 2 and min(sv, en) / (sv + en) >= 0.2:
        label = "mixed/other"
    else:
        label = "sv" if sv > en else "en"
    return {"label": label, "sv_hits": sv, "en_hits": en}
