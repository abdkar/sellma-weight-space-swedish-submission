"""Original synthetic tasks; translation pairs and template families stay together."""
from pathlib import Path
import random
from .common import append_jsonl, save_json, digest

# Five held-out wording families per operation. Two numeric variants per family.
TEST = {
"addition": [
 ("Maja har {a} böcker och får {b} till. Hur många böcker har hon nu?", "Maja has {a} books and receives {b} more. How many books does she have now?"),
 ("Det finns {a} röda och {b} blå kulor i en låda. Hur många kulor finns det sammanlagt?", "A box contains {a} red and {b} blue marbles. How many marbles are there altogether?"),
 ("En buss har {a} passagerare. Vid nästa hållplats stiger {b} personer på och ingen går av. Hur många passagerare finns då?", "A bus has {a} passengers. At the next stop {b} people board and nobody leaves. How many passengers are there then?"),
 ("Nils går {a} meter till parken och sedan {b} meter till skolan. Hur många meter går han totalt?", "Nils walks {a} metres to the park and then {b} metres to school. How many metres does he walk in total?"),
 ("Ett lag får {a} poäng i första omgången och {b} i andra. Vad är summan av lagets poäng?", "A team scores {a} points in the first round and {b} in the second. What is the sum of its points?")],
"subtraction": [
 ("En butik har {a} pennor och säljer {b}. Hur många pennor finns kvar?", "A shop has {a} pens and sells {b}. How many pens remain?"),
 ("Sara har {a} kronor och betalar {b} kronor för en biljett. Hur många kronor har hon kvar?", "Sara has {a} crowns and pays {b} crowns for a ticket. How many crowns does she have left?"),
 ("Av {a} platser är {b} upptagna. Hur många platser är lediga?", "Of {a} seats, {b} are occupied. How many seats are free?"),
 ("En bok har {a} sidor. Ali har läst {b} sidor. Hur många sidor återstår?", "A book has {a} pages. Ali has read {b} pages. How many pages remain?"),
 ("En behållare rymmer {a} liter och innehåller nu {b} liter. Hur många liter till behövs för att fylla den?", "A container holds {a} litres and currently contains {b} litres. How many more litres are needed to fill it?")],
"multiplication": [
 ("Det finns {a} lådor med {b} äpplen i varje. Hur många äpplen finns det?", "There are {a} boxes with {b} apples in each. How many apples are there?"),
 ("En biljett kostar {a} kronor. Hur många kronor kostar {b} biljetter tillsammans?", "One ticket costs {a} crowns. How many crowns do {b} tickets cost together?"),
 ("En sal har {a} rader med {b} stolar per rad. Hur många stolar finns i salen?", "A hall has {a} rows with {b} chairs per row. How many chairs are in the hall?"),
 ("Lina läser {a} sidor varje dag i {b} dagar. Hur många sidor läser hon totalt?", "Lina reads {a} pages each day for {b} days. How many pages does she read in total?"),
 ("Varje cykel behöver {a} reflexer. Hur många reflexer behövs till {b} cyklar?", "Each bicycle needs {a} reflectors. How many reflectors are needed for {b} bicycles?")],
"division": [
 ("{a} kakor delas lika mellan {b} barn. Hur många kakor får varje barn?", "{a} biscuits are shared equally among {b} children. How many biscuits does each child receive?"),
 ("{a} pennor packas i askar med {b} pennor i varje. Hur många askar behövs?", "{a} pens are packed into boxes of {b} pens each. How many boxes are needed?"),
 ("En sträcka är {a} meter. Varje varv är {b} meter. Hur många hela varv motsvarar sträckan?", "A distance is {a} metres. Each lap is {b} metres. How many complete laps equal that distance?"),
 ("En grupp betalar totalt {a} kronor för {b} lika dyra biljetter. Hur många kronor kostar en biljett?", "A group pays a total of {a} crowns for {b} equally priced tickets. How many crowns does one ticket cost?"),
 ("{a} bilder fördelas jämnt över {b} sidor. Hur många bilder finns på varje sida?", "{a} pictures are distributed evenly over {b} pages. How many pictures are on each page?")],
}
DEV = [
 ("addition", "Vad är summan av heltalen {a} och {b}?", "What is the sum of the integers {a} and {b}?"),
 ("subtraction", "Beräkna skillnaden {a} minus {b}.", "Calculate the difference {a} minus {b}."),
 ("multiplication", "Vilket tal får du om du multiplicerar {a} med {b}?", "What number do you get when you multiply {a} by {b}?"),
 ("division", "Beräkna kvoten när {a} divideras med {b}.", "Calculate the quotient when {a} is divided by {b}."),
 ("two_step", "Börja med {a}. Lägg till {b} och dra sedan bort 3. Vilket tal får du?", "Start with {a}. Add {b} and then subtract 3. What number do you get?")
]

def values(operation, rng):
    if operation == "division":
        b, gold = rng.randint(2, 8), rng.randint(2, 12)
        return b * gold, b, gold
    a, b = rng.randint(9, 30), rng.randint(2, 8)
    if operation == "addition": gold = a + b
    elif operation == "subtraction": gold = a - b
    elif operation == "multiplication": gold = a * b
    else: gold = a + b - 3
    return a, b, gold

def build_rows():
    rng = random.Random(20261118)
    rows = []
    def add(split, family, category, variant, sv, en):
        a, b, answer = values(category, rng)
        for language, template in [("sv", sv), ("en", en)]:
            rows.append({"id": f"{family}-v{variant}", "concept_id": f"{family}-v{variant}",
                         "template_family_id": family, "split": split, "category": category,
                         "language": language, "prompt": template.format(a=a, b=b),
                         "gold": str(answer), "answer_type": "integer", "a": a, "b": b})
    for i, (category, sv, en) in enumerate(DEV):
        for variant in range(2): add("dev", f"dev-{i:02}", category, variant, sv, en)
    # Round robin categories; nested 16/24/40 subsets remain balanced.
    for family in range(5):
        for category, templates in TEST.items():
            for variant in range(2):
                add("test", f"test-{category}-{family}", category, variant, *templates[family])
    probes = [
        ("Text: 'Ida bor i Umeå.' I vilken stad bor Ida?", "Text: 'Ida lives in Umeå.' In which city does Ida live?", "Umeå"),
        ("Skriv endast ordet blå efter svarsetiketten.", "Write only the word blue after the answer label.", ("blå", "blue")),
        ("Text: 'Paketet är märkt K7.' Vilken kod har paketet?", "Text: 'The parcel is labeled K7.' What is the parcel code?", "K7"),
        ("Välj det sista ordet i listan: sol, vind, regn.", "Choose the last word in the list: sun, wind, rain.", ("regn", "rain")),
        ("Text: 'Mötet är på tisdag.' Vilken veckodag är mötet?", "Text: 'The meeting is on Tuesday.' On which weekday is the meeting?", ("tisdag", "Tuesday")),
        ("Vilket av orden 'katt' och 'bil' är ett djur?", "Which of the words 'cat' and 'car' is an animal?", ("katt", "cat")),
        ("Text: 'Lådan är tom.' Innehåller lådan något? Svara ja eller nej.", "Text: 'The box is empty.' Does the box contain anything? Answer yes or no.", ("nej", "no")),
        ("Återge koden exakt: AB-42.", "Repeat the code exactly: AB-42.", "AB-42"),
    ]
    for i, (sv, en, gold) in enumerate(probes):
        for j, (language, prompt) in enumerate([("sv", sv), ("en", en)]):
            rows.append({"id": f"probe-{i:02}", "concept_id": f"probe-{i:02}",
                         "template_family_id": f"probe-{i:02}", "split": "regression",
                         "category": "instruction", "language": language, "prompt": prompt,
                         "gold": gold[j] if isinstance(gold, tuple) else gold, "answer_type": "text"})
    validate(rows)
    return rows

def validate(rows):
    seen, families = set(), {}
    pairs = {}
    for row in rows:
        key = row["id"], row["language"]
        if key in seen: raise ValueError(f"Duplicate dataset row {key}")
        seen.add(key)
        family = row["template_family_id"]
        if family in families and families[family] != row["split"]: raise ValueError("Family leaks across splits")
        families[family] = row["split"]
        pairs.setdefault(row["id"], []).append(row)
        if row["answer_type"] == "integer":
            a, b, op = row["a"], row["b"], row["category"]
            expected = {"addition": a+b, "subtraction": a-b, "multiplication": a*b,
                        "division": a//b, "two_step": a+b-3}[op]
            if str(expected) != row["gold"]: raise ValueError("Invalid arithmetic gold")
    for group in pairs.values():
        if {r["language"] for r in group} != {"sv", "en"}: raise ValueError("Unpaired translation")
        if group[0]["answer_type"] == "integer" and len({r["gold"] for r in group}) != 1:
            raise ValueError("Translation answer mismatch")
    return {"rows": len(rows), "concepts": len(pairs), "families": len(families)}

def selected(rows, phase, pairs=40, dev_pairs=4):
    if phase in {"smoke", "pilot", "audit"}:
        pool = [r for r in rows if r["split"] == "dev"]
        all_ids = list(dict.fromkeys(r["id"] for r in pool))
        ids = (all_ids[::2] + all_ids[1::2])[:dev_pairs]
        return [r for r in pool if r["id"] in ids]
    pool = [r for r in rows if r["split"] == "test"]
    ids = list(dict.fromkeys(r["id"] for r in pool))[:pairs]
    return [r for r in rows if r["id"] in ids or r["split"] == "regression"]

def render_user(row):
    if row["language"] == "sv":
        suffix = "Avsluta med en egen rad: Svar: <heltal>." if row["answer_type"] == "integer" else "Avsluta med en egen rad: Svar: <svar>."
    else:
        suffix = "End with a separate line: Answer: <integer>." if row["answer_type"] == "integer" else "End with a separate line: Answer: <answer>."
    return row["prompt"] + "\n" + suffix

def write_dataset(directory):
    directory = Path(directory)
    rows = build_rows()
    target = directory / "tasks.jsonl"
    if target.exists():
        raise FileExistsError("Dataset already exists; do not overwrite a frozen dataset")
    append_jsonl(target, rows)
    save_json(directory / "manifest.json", {"dataset_hash": digest(rows), **validate(rows),
              "origin": "Original synthetic wording, AI-assisted; no benchmark items copied",
              "seed": 20261118, "human_review": "Review the displayed pairs before freezing"})
