import pandas as pd
import spacy
import re
from transformers import  pipeline
from collections import Counter



LABEL_MAP = {
    "PER": "PERSON",
    "LOC": "GPE",
    "ORG": "ORG",
    "MISC": "MISC"
}

# Load Data
def load_data(path):

    df = pd.read_csv(path)

    print("\n=== DATASET LOADED ===")
    print(df.columns)

    return df


# Detect Arabic Text
def is_arabic(text):

    arabic_chars = re.findall(
        r'[\u0600-\u06FF]',
        str(text)
    )

    return len(arabic_chars) > 20


# Split Data By Language
def split_languages(df, text_column, limit=20):

    english_texts = []
    arabic_texts = []

    for text in df[text_column].dropna():

        text = str(text).strip()

        if len(text) < 10:
            continue

        if is_arabic(text):
            arabic_texts.append(text)

        else:
            english_texts.append(text)

    english_texts = english_texts[:limit]
    arabic_texts = arabic_texts[:limit]

    print("\n=== LANGUAGE SPLIT ===")
    print(f"English texts: {len(english_texts)}")
    print(f"Arabic texts: {len(arabic_texts)}")

    return english_texts, arabic_texts


# Load Spacy Model
def load_spacy_model():

    print("\nLoading spaCy multilingual model...")
    model = spacy.load("xx_ent_wiki_sm")

    return model


# Load Hugging Face Model
def load_hf_model():

    print("Loading Hugging Face multilingual model...")

    ner_pipeline = pipeline(
        "ner",
        model="Davlan/xlm-roberta-base-wikiann-ner",
        aggregation_strategy="simple"
    )

    return ner_pipeline


# Run Spacy NER
def run_spacy_ner(texts, spacy_model, language_name):

    entity_counter = Counter()

    total_entities = 0
    total_words = 0
    no_entity_count = 0

    examples = []

    for text in texts:

        doc = spacy_model(text)

        total_words += len(text.split())

        if len(doc.ents) == 0:
            no_entity_count += 1

        for ent in doc.ents:

            label = LABEL_MAP.get(
                ent.label_,
                ent.label_
            )

            entity_counter[label] += 1
            total_entities += 1

            if len(examples) < 3:

                examples.append(
                    (ent.text, label)
                )

    density = 0

    if total_words > 0:
        density = (
            total_entities / total_words
        ) * 100

    return {
        "Language": language_name,
        "Model": "spaCy_xx_ent_wiki_sm",
        "Total_Entities": total_entities,
        "Entity_Density": round(density, 2),
        "No_Entity_Rate": round(
            no_entity_count / len(texts),
            2
        ),
        "Entity_Types": dict(entity_counter),
        "Examples": examples
    }


# Run Hugging Face NER
def run_hf_ner(texts, hf_model, language_name):

    entity_counter = Counter()

    total_entities = 0
    total_words = 0
    no_entity_count = 0

    examples = []

    for text in texts:

        total_words += len(text.split())

        try:
            results = hf_model(text)

        except Exception as e:
            print("HF ERROR:", e)
            continue

        if len(results) == 0:
            no_entity_count += 1

        for ent in results:

            label = LABEL_MAP.get(
                ent["entity_group"],
                ent["entity_group"]
            )

            entity_counter[label] += 1
            total_entities += 1

            if len(examples) < 3:

                examples.append(
                    (ent["word"], label)
                )

    density = 0

    if total_words > 0:
        density = (
            total_entities / total_words
        ) * 100

    return {
        "Language": language_name,
        "Model": "HF_xlm_roberta_wikiann",
        "Total_Entities": total_entities,
        "Entity_Density": round(density, 2),
        "No_Entity_Rate": round(
            no_entity_count / len(texts),
            2
        ),
        "Entity_Types": dict(entity_counter),
        "Examples": examples
    }


# Print Results
def print_results(results):

    print("\n" + "=" * 60)
    print("MULTILINGUAL NER COMPARISON")
    print("=" * 60)

    for result in results:

        print(f"\nLanguage: {result['Language']}")
        print(f"Model: {result['Model']}")

        print(
            f"Total Entities: "
            f"{result['Total_Entities']}"
        )

        print(
            f"Entity Density: "
            f"{result['Entity_Density']}"
        )

        print(
            f"No Entity Rate: "
            f"{result['No_Entity_Rate']}"
        )

        print("\nEntity Types:")

        for entity_type, count in result[
            "Entity_Types"
        ].items():

            print(f"  {entity_type}: {count}")

        print("\nExamples:")

        for entity, label in result["Examples"]:

            print(f"  {entity} --> {label}")

        print("-" * 50)


# Save Comparison Table
def save_results(results):

    rows = []

    for result in results:

        rows.append({
            "Language": result["Language"],
            "Model": result["Model"],
            "Total_Entities":
                result["Total_Entities"],
            "Entity_Density":
                result["Entity_Density"],
            "No_Entity_Rate":
                result["No_Entity_Rate"],
            "Entity_Types":
                str(result["Entity_Types"]),
            "Examples":
                str(result["Examples"])
        })

    comparison_df = pd.DataFrame(rows)

    comparison_df.to_csv(
        "multilingual_ner_comparison.csv",
        index=False
    )

    print(
        "\nSaved: multilingual_ner_comparison.csv"
    )


# Main
def main():

    # Load dataset
    df = load_data("data/climate_articles.csv")

    # Split by language
    english_texts, arabic_texts = split_languages(
        df,
        "text" # Text Column
    )

    # Load models
    spacy_model = load_spacy_model()

    hf_model = load_hf_model()

    # Run experiments
    english_spacy = run_spacy_ner(
        english_texts,
        spacy_model,
        "English"
    )

    arabic_spacy = run_spacy_ner(
        arabic_texts,
        spacy_model,
        "Arabic"
    )

    english_hf = run_hf_ner(
        english_texts,
        hf_model,
        "English"
    )

    arabic_hf = run_hf_ner(
        arabic_texts,
        hf_model,
        "Arabic"
    )

    # Combine results
    all_results = [
        english_spacy,
        arabic_spacy,
        english_hf,
        arabic_hf
    ]

    # Print results
    print_results(all_results)

    # Save results
    save_results(all_results)

    print("\nDONE!")


if __name__ == "__main__":
    main()