import spacy
from spacy.pipeline import EntityRuler
import pandas as pd
from collections import Counter


STANDARD_LABELS = {
    "ORG", "GPE", "DATE", "LAW", "MONEY",
    "PERSON", "QUANTITY", "LOC", "EVENT", "WORK_OF_ART"
}

def create_nlp_with_ruler(position="before"):
    nlp = spacy.load("en_core_web_sm")

    patterns = [
        # policy / agreement 
        {"label": "AGREEMENT", "pattern": "Paris Agreement"},
        {"label": "AGREEMENT", "pattern": "Kyoto Protocol"},

        # events
        {"label": "CLIMATE_EVENT", "pattern": "COP28"},
        {"label": "CLIMATE_EVENT", "pattern": "COP27"},

        # reports
        {"label": "REPORT", "pattern": "IPCC AR6"},
        {"label": "REPORT", "pattern": "Sixth Assessment Report"},

        # organizations
        {"label": "ORG", "pattern": "IPCC"},
        {"label": "ORG", "pattern": "UNEP"},

        # thresholds (token patterns)
        {
            "label": "THRESHOLD",
            "pattern": [
                {"LIKE_NUM": True},
                {"IS_SPACE": True, "OP": "*"},
                {"TEXT": {"REGEX": "°?C"}}
            ] 
        },
        {
            "label": "THRESHOLD",
            "pattern": [
                {"LIKE_NUM": True},
                {"IS_SPACE": True, "OP": "*"},
                {"TEXT": {"IN": ["percent", "%"]}}
            ]
        }
    ]

    if position == "before":
        ruler = nlp.add_pipe("entity_ruler", before="ner")
    else:
        ruler = nlp.add_pipe("entity_ruler", after="ner")

    ruler.add_patterns(patterns)

    return nlp


def extract_entities_with_model(df, nlp):
    df_en = df[df['language'] == 'en']
    records = []

    for _, row in df_en.iterrows():
        doc = nlp(row['text'])

        for ent in doc.ents:
            records.append({
                "text_id": row['id'],
                "entity_text": ent.text,
                "entity_label": ent.label_
            })

    return pd.DataFrame(records)


def count_entities(df):
    return df['entity_label'].value_counts().to_dict()



def filter_standard(df):
    return df[df["entity_label"].isin(STANDARD_LABELS)]


def evaluate_ner(pred_df, gold_df):
    pred_set = set(zip(pred_df.text_id, pred_df.entity_text, pred_df.entity_label))
    gold_set = set(zip(gold_df.text_id, gold_df.entity_text, gold_df.entity_label))

    tp = len(pred_set & gold_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0

    return {"precision": precision, "recall": recall, "f1": f1}


def show_examples(df, label, n=3):
    examples = df[df["entity_label"] == label].head(n)
    print(f"\nExamples for {label}:")
    for _, row in examples.iterrows():
        print(f"- {row['entity_text']}")



if __name__ == "__main__":
    df = pd.read_csv("data/climate_articles.csv")
    gold = pd.read_csv("data/gold_entities.csv")

    # base model
    base_nlp = spacy.load("en_core_web_sm")

    # with rules BEFORE
    nlp_before = create_nlp_with_ruler("before")

    # with rules AFTER
    nlp_after = create_nlp_with_ruler("after")

    # extract entities
    base_entities = extract_entities_with_model(df, base_nlp)
    before_entities = extract_entities_with_model(df, nlp_before)
    after_entities = extract_entities_with_model(df, nlp_after)

    # count comparison
    print("\n=== ENTITY COUNTS ===")
    print("Base:", count_entities(base_entities))
    print("Before:", count_entities(before_entities))
    print("After:", count_entities(after_entities))

    # evaluate only standard labels
    base_std = filter_standard(base_entities)
    before_std = filter_standard(before_entities)
    after_std = filter_standard(after_entities)
    gold_std = filter_standard(gold)

    print("\n=== EVALUATION ===")
    print("Base:", evaluate_ner(base_std, gold_std))
    print("Before:", evaluate_ner(before_std, gold_std))
    print("After:", evaluate_ner(after_std, gold_std))


    # examples
    print("\n=== EXAMPLES ===")
    show_examples(before_entities, "AGREEMENT")
    show_examples(before_entities, "CLIMATE_EVENT")
    show_examples(before_entities, "THRESHOLD")