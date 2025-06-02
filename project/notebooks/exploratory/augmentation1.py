#!/usr/bin/env python3
"""
augment_data.py  ·  final

✓ Entity‑swap   ✓ Mask‑and‑fill on first 128 words (exactly one <mask>)  ✓ Full‑doc paraphrase
Writes 4×‑larger datasets into augmented_data/train, dev, test.

Prerequisites
-------------
• Your original JSONs live in train/, dev/, test/ beside this script.
• LM Studio server is running at http://127.0.0.1:1234
  and has loaded DeepSeek‑R1‑Distill‑Qwen‑32B‑GGUF.
• `pip install transformers requests`
"""

# --------------------------------------------------------------------------- #
# silence noisy logs before importing heavy deps
import os, warnings, random, json, requests, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # kill TF INFO
os.environ["TRANSFORMERS_NO_TF"]   = "1"   # force PyTorch backend
warnings.filterwarnings("ignore", module="torchvision")
logging.getLogger("transformers").setLevel(logging.ERROR)  # hide HF weight msgs

from copy import deepcopy
from transformers import pipeline

# ---------- paths ----------------------------------------------------------- #
BASE_DIR   = os.path.dirname(os.path.realpath(__file__))
SPLITS     = ["train", "dev", "test"]
IN_DIR     = {s: os.path.join(BASE_DIR, s) for s in SPLITS}
OUT_DIR    = {s: os.path.join(BASE_DIR, "augmented_data", s) for s in SPLITS}

# ---------- models & params ------------------------------------------------- #
API_URL     = "http://127.0.0.1:1234/v1/completions"
MODEL_NAME  = "DeepSeek-R1-Distill-Qwen-32B-GGUF"
MLM_MODEL   = "bert-base-uncased"

WORD_LIMIT  = 128      # first 128 words → <512 subtokens
MASK_PROB   = 0.10     # 10 % probability a given word becomes <mask>
ENTITY_SWAP = 0.30     # 30 % chance to swap each mention

# ---------- fast BERT fill‑mask -------------------------------------------- #
mlm = pipeline(
    "fill-mask",
    model=MLM_MODEL,
    tokenizer=MLM_MODEL,
    framework="pt",
    device=0,            # set to -1 to force CPU
    top_k=1,
)

# ---------- helpers --------------------------------------------------------- #
def load_jsons(folder):
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


def build_pool(files):
    pool = {}
    for _, docs in files:
        for doc in docs:
            for ent in doc.get("entities", []):
                pool.setdefault(ent["type"], set()).update(ent["mentions"])
    pool = {t: list(v) for t, v in pool.items()}
    print(f"[INFO] Entity types in pool: {list(pool)}")
    return pool

# ---------- augmentation funcs --------------------------------------------- #

def swap_entities(doc, pool):
    d = deepcopy(doc)
    for ent in d.get("entities", []):
        if ent["type"] in pool and random.random() < ENTITY_SWAP:
            k = min(len(ent["mentions"]), len(pool[ent["type"]]))
            ent["mentions"] = random.sample(pool[ent["type"]], k)
    return d


def mask_fill_prefix(text: str) -> str:
    """Mask‑and‑fill only the first 128 words with **exactly one** <mask>."""
    words = text.split()
    head  = words[:WORD_LIMIT]
    tail  = words[WORD_LIMIT:]

    # choose one index to mask (guaranteed single mask)
    idx = random.randrange(len(head))
    head[idx] = mlm.tokenizer.mask_token

    # run BERT fill‑mask (returns list with one dict)
    prediction = mlm(" ".join(head))[0]["sequence"]
    cleaned    = prediction.replace("[CLS]", "").replace("[SEP]", "").strip()

    return " ".join(cleaned.split() + tail)


def masked_doc(doc):
    d = deepcopy(doc)
    d["doc"] = mask_fill_prefix(d["doc"])
    return d


def paraphrase(doc):
    d      = deepcopy(doc)
    prompt = (
        "Paraphrase the following text while preserving all entity "
        "mentions exactly:\n\n" + d["doc"]
    )
    resp = requests.post(API_URL, json={
        "model":       MODEL_NAME,
        "prompt":      prompt,
        "max_tokens":  512,
        "temperature": 0.7,
        "top_p":       0.9,
        "n":           1,
    })
    resp.raise_for_status()
    txt = resp.json()["choices"][0]["text"]
    if txt.startswith(prompt):
        txt = txt[len(prompt):]
    d["doc"] = txt.strip()
    return d

# ---------- per‑split driver ----------------------------------------------- #

def augment_split(in_dir, out_dir, pool):
    files = load_jsons(in_dir)
    if not files:
        return
    os.makedirs(out_dir, exist_ok=True)

    for fn, docs in files:
        aug = []
        for d in docs:
            aug += [
                d,
                swap_entities(d, pool),
                masked_doc(d),
                paraphrase(d),
            ]
        with open(os.path.join(out_dir, fn), "w", encoding="utf-8") as f:
            json.dump(aug, f, ensure_ascii=False, indent=2)
        print(f"[OK] {fn}: {len(docs)} → {len(aug)}")

# ---------- main ----------------------------------------------------------- #

def main():
    random.seed(42)
    all_docs = []
    for s in SPLITS:
        all_docs += load_jsons(IN_DIR[s])
    pool = build_pool(all_docs)

    for s in SPLITS:
        augment_split(IN_DIR[s], OUT_DIR[s], pool)

if __name__ == "__main__":
    main()
