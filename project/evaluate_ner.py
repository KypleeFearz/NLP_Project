import json
from sklearn.metrics import precision_recall_fscore_support

def load_mentions_from_reference(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    all_mentions = []
    all_typed_mentions = []
    for entry in data.values():
        entity_mentions = set()
        typed_mentions = set()
        for ent in entry["entities"]:
            for mention in ent["mentions"]:
                mention_clean = mention.strip().lower()
                entity_mentions.add(mention_clean)
                typed_mentions.add((mention_clean, ent["type"]))
        all_mentions.append(entity_mentions)
        all_typed_mentions.append(typed_mentions)
    return all_mentions, all_typed_mentions

def load_mentions_from_prediction(file):
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)

    all_mentions = []
    all_typed_mentions = []
    for entry in data.values():
        entity_mentions = set()
        typed_mentions = set()
        for ent in entry["entities"]:
            mention_clean = ent["word"].strip().lower()
            entity_mentions.add(mention_clean)
            typed_mentions.add((mention_clean, ent["entity_group"]))
        all_mentions.append(entity_mentions)
        all_typed_mentions.append(typed_mentions)
    return all_mentions, all_typed_mentions

def compute_f1(pred_list, ref_list):
    tp = sum(1 for pred, ref in zip(pred_list, ref_list) if pred == ref)
    precision = tp / len(pred_list) if pred_list else 0
    recall = tp / len(ref_list) if ref_list else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
    return f1

def evaluate_ner_from_mentions(ref_file, pred_file):
    ref_mentions, ref_typed_mentions = load_mentions_from_reference(ref_file)
    pred_mentions, pred_typed_mentions = load_mentions_from_prediction(pred_file)

    f1_ei = compute_f1(pred_mentions, ref_mentions)
    f1_ec = compute_f1(pred_typed_mentions, ref_typed_mentions)

    return f1_ei, f1_ec

if __name__ == "__main__":
    ref_path = "C:/Users/nmilo/OneDrive/Desktop/Master/Semester2/NLP/project/input/ref/reference.json"
    pred_path = "C:/Users/nmilo/OneDrive/Desktop/Master/Semester2/NLP/project/input/res/results.json"

    f1_ei, f1_ec = evaluate_ner_from_mentions(ref_path, pred_path)
    print(f"F1_EI (Entity Identification): {f1_ei:.4f}")
    print(f"F1_EC (Entity Classification): {f1_ec:.4f}")
