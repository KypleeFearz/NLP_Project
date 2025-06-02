#!/usr/bin/env python3
"""
augment_data.py · v3  — *robust, self‑healing data augmentation*

This version fixes the empty‑file issue you just hit:
1. **Pre‑clean** each source document – any entity whose mentions don’t occur in
   the paragraph is dropped, and relations are rewired or discarded accordingly.
   That prevents the validator from rejecting already‑dirty ground‑truth.
2. Augmentation functions are wrapped so that if *one* variant fails validation
   it is skipped but the others still write.
3. A per‑variant try/except keeps the whole run alive even when a single doc is
   beyond repair.

You should now always get up to 4 variants (orig, swap, mask, para) per valid
input doc.  Warnings in the log tell you which individual transformations were
skipped.

Usage and prerequisites remain unchanged.
"""

from __future__ import annotations

import os
import re
import json
import random
import logging
import warnings
from copy import deepcopy
from typing import Dict, List, Tuple, Set

import requests
from transformers import pipeline

# --------------------------------------------------------------------------- #
# silence noisy logs before importing heavy deps
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # hide TF INFO
os.environ["TRANSFORMERS_NO_TF"] = "1"   # force PyTorch backend
warnings.filterwarnings("ignore", module="torchvision")
logging.getLogger("transformers").setLevel(logging.ERROR)

# ------------------------------ paths -------------------------------------- #
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
SPLITS   = ["train", "dev", "test"]
IN_DIR   = {s: os.path.join(BASE_DIR, s) for s in SPLITS}
OUT_DIR  = {s: os.path.join(BASE_DIR, "augmented_data", s) for s in SPLITS}

# ------------------------------ params ------------------------------------- #
API_URL     = "http://127.0.0.1:1234/v1/completions"
MODEL_NAME  = "DeepSeek-R1-Distill-Qwen-32B-GGUF"
MLM_MODEL   = "bert-base-uncased"

WORD_LIMIT  = 128       # prefix length for mask‑and‑fill
MASK_PROB   = 0.10      # prob. each non‑entity token is mask candidate
ENTITY_SWAP = 0.30      # prob. to swap each entity
RANDOM_SEED = 42

random.seed(RANDOM_SEED)

# ----------------------------- models -------------------------------------- #
mlm = pipeline(
    "fill-mask",
    model=MLM_MODEL,
    tokenizer=MLM_MODEL,
    framework="pt",
    device=0,         # set to ‑1 to force CPU
    top_k=1,
)

# ============================= helpers ===================================== #

def load_jsons(folder: str) -> List[Tuple[str, List[dict]]]:
    if not os.path.isdir(folder):
        print(f"[WARN] Missing dir: {folder}")
        return []
    out = []
    for fn in sorted(os.listdir(folder)):
        if fn.endswith(".json"):
            with open(os.path.join(folder, fn), encoding="utf-8") as f:
                out.append((fn, json.load(f)))
    print(f"[INFO] {len(out)} JSONs in {folder}")
    return out


def build_pool(files: List[Tuple[str, List[dict]]]) -> Dict[str, List[str]]:
    pool: Dict[str, Set[str]] = {}
    for _, docs in files:
        for doc in docs:
            for ent in doc.get("entities", []):
                pool.setdefault(ent["type"], set()).update(ent["mentions"])
    return {t: sorted(v) for t, v in pool.items()}

# ========================= validation ====================================== #

def validate(doc: dict):
    """Raise AssertionError if doc is inconsistent."""
    text = doc["doc"]
    ids  = {e["id"] for e in doc["entities"]}

    for ent in doc["entities"]:
        assert ent["mentions"], "Empty mention list"
        for m in ent["mentions"]:
            assert m in text, f"Mention not found: {m[:30]}…"
    for rel in doc.get("relations", []):
        assert rel["head"] in ids and rel["tail"] in ids, "Dangling relation ID"

# ========================= cleaning pass =================================== #

def clean_doc(doc: dict) -> dict | None:
    """Remove invalid entities/relations so that the remaining doc passes validate."""
    d = deepcopy(doc)
    text = d["doc"]
    kept_entities = []
    id_map: Dict[str, str] = {}
    next_id = 0

    for ent in d.get("entities", []):
        mentions_in_text = [m for m in ent["mentions"] if m in text]
        if mentions_in_text:
            new_ent = {
                **ent,
                "id": str(next_id),
                "mentions": mentions_in_text,
            }
            id_map[ent["id"]] = str(next_id)
            next_id += 1
            kept_entities.append(new_ent)

    if not kept_entities:
        return None  # nothing valid left

    d["entities"] = kept_entities
    # rebuild relations
    new_rels = []
    for rel in d.get("relations", []):
        if rel["head"] in id_map and rel["tail"] in id_map:
            new_rels.append({**rel, "head": id_map[rel["head"]], "tail": id_map[rel["tail"]]})
    d["relations"] = new_rels

    try:
        validate(d)
    except AssertionError as e:
        logging.warning(f"[CLEAN‑FAIL] {e}")
        return None

    return d

# ========================= augmentation funcs ============================== #

# ---------- 1. entity‑swap -------------------------------------------------- #

def same_len_candidates(candidates: List[str], length: int) -> List[str]:
    return [c for c in candidates if len(c.split()) == length] or candidates


def replace_first(text: str, old: str, new: str) -> str:
    """Replace the *first* occurrence of `old` with `new` in `text`."""
    return re.sub(re.escape(old), new, text, count=1)


def swap_entities(doc: dict, pool: Dict[str, List[str]]) -> dict:
    d = deepcopy(doc)
    for ent in d.get("entities", []):
        if ent["type"] in pool and random.random() < ENTITY_SWAP:
            orig_len = len(ent["mentions"][0].split())
            choices  = same_len_candidates(pool[ent["type"]], orig_len)
            new_mention = random.choice(choices)
            for old in ent["mentions"]:
                d["doc"] = replace_first(d["doc"], old, new_mention)
            ent["mentions"] = [new_mention]
    d["aug_type"] = "swap"
    validate(d)
    return d

# ---------- 2. mask‑and‑fill ---------------------------------------------- #

def mask_fill_prefix(doc: dict) -> dict:
    d = deepcopy(doc)
    tokens = d["doc"].split()
    prefix = tokens[:WORD_LIMIT]
    ent_tokens: Set[str] = set()
    for ent in d["entities"]:
        for m in ent["mentions"]:
            ent_tokens.update(m.split())
    idx_candidates = [i for i, tok in enumerate(prefix) if tok not in ent_tokens]
    if not idx_candidates:
        return d  # safe no‑op if we can’t mask anything
    idx = random.choice(idx_candidates)
    prefix[idx] = mlm.tokenizer.mask_token
    prediction = mlm(" ".join(prefix))[0]["sequence"]
    cleaned = prediction.replace("[CLS]", "").replace("[SEP]", "").strip()
    d["doc"] = " ".join(cleaned.split() + tokens[WORD_LIMIT:])
    d["aug_type"] = "mask"
    validate(d)
    return d

# ---------- 3. paraphrase -------------------------------------------------- #

def paraphrase(doc: dict) -> dict:
    d = deepcopy(doc)
    unique_mentions = sorted({m for e in d["entities"] for m in e["mentions"]}, key=len, reverse=True)
    placeholder: Dict[str, str] = {m: f"[[E{i}]]" for i, m in enumerate(unique_mentions)}

    def protect(text: str) -> str:
        for m, tag in placeholder.items():
            text = text.replace(m, tag)
        return text

    protected = protect(d["doc"])
    prompt = (
        "Paraphrase the following text **without altering placeholder tags**.\n\n" + protected
    )
    resp = requests.post(API_URL, json={
        "model": MODEL_NAME,
        "prompt": prompt,
        "max_tokens": 1024,
        "temperature": 0.7,
        "top_p": 0.9,
        "n": 1,
    })
    resp.raise_for_status()
    out = resp.json()["choices"][0]["text"].strip()
    if out.startswith(prompt):
        out = out[len(prompt):].strip()
    for m, tag in placeholder.items():
        out = out.replace(tag, m)
    d["doc"] = out
    d["aug_type"] = "para"
    validate(d)
    return d

# ---------- 4. wrapper that labels original -------------------------------- #

def mark_original(doc: dict) -> dict:
    d = deepcopy(doc)
    d["aug_type"] = "orig"
    return d

# ========================= per‑split driver ================================ #

def augment_split(in_dir: str, out_dir: str, pool: Dict[str, List[str]]):
    files = load_jsons(in_dir)
    if not files:
        return
    os.makedirs(out_dir, exist_ok=True)

    for fn, docs in files:
        out: List[dict] = []
        for raw in docs:
            base = clean_doc(raw)
            if base is None:
                logging.warning(f"[DROP] {fn}: cannot clean – skipping doc")
                continue
            # run each augmentation safely
            variants = [mark_original(base)]
            for func in (swap_entities, mask_fill_prefix, paraphrase):
                try:
                    var = func(base, pool) if func is swap_entities else func(base)
                    variants.append(var)
                except Exception as e:
                    logging.warning(f"[SKIP] {fn}: {e}")
            out.extend(variants)

        with open(os.path.join(out_dir, fn), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[OK] {fn}: {len(docs)} → {len(out)} examples written")

# =============================== main ====================================== #

def main():
    all_files = []
    for split in SPLITS:
        all_files += load_jsons(IN_DIR[split])
    pool = build_pool(all_files)
    print(f"[INFO] Pool covers {len(pool)} entity types")

    for split in SPLITS:
        augment_split(IN_DIR[split], OUT_DIR[split], pool)

if __name__ == "__main__":
    main()
