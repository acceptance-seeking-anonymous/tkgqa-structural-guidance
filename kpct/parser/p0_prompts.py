#!/usr/bin/env python3
"""p0_prompts.py -- the (up to) 3 prompt designs of P_0 (rule arms.P_0.prompt / .selection), fixed in writing with
sha256 before any P_0 dev_sel2000 score exists. All three were written before any score, so no design depends on a score.

Every design = instruction + JSON schema + the 251 KG relation names (exact KG form, sorted) + the 8 hand-written
exemplars (p0_exemplars.json) + "Question: <q>\\nJSON:" in one user turn (Llama-3.1 chat template, fixed date_string).
  d1: instruction (= the P_k instruction P_HEAD) + schema with allowed values + relation names + exemplars
  d2: d1 + field definitions (answer_role convention, the six qtype operators, granularity convention)
  d3: d2 + KG entity naming conventions ("Sector (Country)" etc.)
Selection on dev_sel2000: best Hits@1, tie -> shortest prompt (d1 < d2 < d3 by construction).
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pk_common as PC

SCHEMA = (
    "\nJSON schema (all keys required, in this order):\n"
    "- mentions: list of {\"text\": the entity mention copied from the question, \"entity\": its exact KG entity name}\n"
    "- relation: exactly one KG relation name from the list below\n"
    "- answer_role: subject | object\n"
    "- qtype: equal | before_after | first_last | equal_multi | after_first | before_last\n"
    "- answer_type: entity | time\n"
    "- granularity: day | month | year\n"
    "- time: the explicit time in the question as YYYY, YYYY-MM or YYYY-MM-DD, or null\n"
    "- anchor: the exact KG entity name of the reference event's entity in \"before/after <entity>\" questions, or null\n"
    "- order: before | after | first | last | null\n")

DEFS = (
    "\nDefinitions:\n"
    "- answer_role: object if the main entity of the question is the subject (actor) of the relation and the answer fills "
    "the object slot; subject if the main entity is the object (target) and the answer is the actor. For time answers: "
    "object if the first-mentioned entity is the actor towards the second, else subject.\n"
    "- qtype equal: a single event at the given time, or (time answer) when a single event happened.\n"
    "- qtype before_after: events before or after the given time (order = before | after).\n"
    "- qtype first_last: the first or last such event, or its time (order = first | last).\n"
    "- qtype equal_multi: the first or last event within the given time (order = first | last), or an event in the same "
    "day, month or year as the anchor's event.\n"
    "- qtype after_first: the first event after the event involving the anchor entity (order = after).\n"
    "- qtype before_last: the last event before the event involving the anchor entity (order = before).\n"
    "- granularity: for time answers the granularity asked for (\"when\" = day, \"in which month\" = month, \"in which "
    "year\" = year); for entity answers the granularity of the explicit time, day if there is none.\n")

NAMING = (
    "\nKG entity names: countries, persons and organisations by their full name (e.g. \"South Korea\", \"Barack Obama\", "
    "\"East African Community\"); sectors, roles and groups of a country as \"Sector (Country)\" with a singular sector "
    "(e.g. \"Citizen (Brazil)\", \"Police (Kenya)\", \"Head of Government (Greece)\", \"Ministry (Sudan)\"); groups of an "
    "organisation as \"Role (Organisation)\" (e.g. \"Insurgency (Al-Shabaab)\").\n")


def build(exemplars, relations):
    rel = "\nKG relation names (251): " + "; ".join(relations) + "\n"
    ex = "\nExamples:\n" + "".join(f"Question: {e['question']}\nJSON: {e['json']}\n\n" for e in exemplars)
    d1 = PC.P_HEAD + SCHEMA + rel + ex
    d2 = PC.P_HEAD + SCHEMA + DEFS + rel + ex
    d3 = PC.P_HEAD + SCHEMA + DEFS + NAMING + rel + ex
    return {"d1": d1, "d2": d2, "d3": d3}


def p0_user(head, question):
    return head + f"Question: {question}\nJSON:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exemplars", required=True); ap.add_argument("--relations", required=True, help="kg/relation2id.json")
    ap.add_argument("--out", required=True); ap.add_argument("--tokenizer", default=None)
    a = ap.parse_args()
    exd = json.load(open(a.exemplars))
    assert exd["n"] == 8 and exd["sha256"] == PC.sha_recipe(exd), "p0_exemplars.json sha256 mismatch"
    rels = sorted(json.load(open(a.relations)).keys())
    assert len(rels) == 251
    heads = build(exd["exemplars"], rels)
    os.makedirs(a.out, exist_ok=True)
    man = dict(name="p0_prompts", exemplars_sha256=exd["sha256"], exemplar_quids=exd["quids"],
               relations_sha256=PC.sha256_file(a.relations), chat_date=PC.CHAT_DATE, user_suffix="Question: {question}\\nJSON:",
               decoding="greedy, max 160 new tokens", designs={})
    tok = None
    if a.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(a.tokenizer)
    for d, h in heads.items():
        f = os.path.join(a.out, f"p0_prompt_{d}.txt")
        open(f, "w", encoding="utf-8").write(h)
        e = dict(file=os.path.basename(f), sha256=PC.sha256_text(h), chars=len(h))
        if tok is not None:
            e["prompt_tokens_example"] = len(PC.encode_prompt(tok, p0_user(h, exd["exemplars"][0]["question"])))
        man["designs"][d] = e
    man["sha256"] = PC.sha_recipe(man)
    json.dump(man, open(os.path.join(a.out, "p0_prompts_manifest.json"), "w"), indent=1)
    print(json.dumps({d: (v["sha256"][:12], v["chars"], v.get("prompt_tokens_example")) for d, v in man["designs"].items()}))


if __name__ == "__main__":
    main()
