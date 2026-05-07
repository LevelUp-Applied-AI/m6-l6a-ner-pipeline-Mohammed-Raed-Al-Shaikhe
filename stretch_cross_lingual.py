import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns

from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity


# config
DATA_PATH = "data/climate_articles.csv"
MODEL_NAME = "bert-base-multilingual-cased"
NUM_TEXTS_PER_LANGUAGE = 10
OUTPUT_HEATMAP = "cross_lingual_heatmap.png"


# load data
def load_data(filepath):

    print("Loading dataset...")

    df = pd.read_csv(filepath)

    print(f"Dataset shape: {df.shape}")

    return df


# filter language data
def filter_language_data(df, text_column):

    english_df = df[df["language"] == "en"].copy()

    arabic_df = df[df["language"] == "ar"].copy()

    # remove missing text
    english_df = english_df.dropna(subset=[text_column])

    arabic_df = arabic_df.dropna(subset=[text_column])

    # remove very short text
    english_df = english_df[
        english_df[text_column].str.len() > 50
    ]

    arabic_df = arabic_df[
        arabic_df[text_column].str.len() > 50
    ]

    print(f"English articles: {len(english_df)}")

    print(f"Arabic articles: {len(arabic_df)}")

    return english_df, arabic_df


# select texts
def select_texts(english_df, arabic_df, text_column, num_texts=10):
    
    english_texts = (
        english_df[text_column]
        .head(num_texts)
        .tolist()
    )

    arabic_texts = (
        arabic_df[text_column]
        .head(num_texts)
        .tolist()
    )

    return english_texts, arabic_texts


# load model
def load_model(model_name):

    print("Loading multilingual BERT...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = AutoModel.from_pretrained(model_name)

    model.eval()

    print("Model loaded successfully.")

    return tokenizer, model


# get embedding
def get_embedding(text, tokenizer, model):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )

    with torch.no_grad():

        outputs = model(**inputs)

    last_hidden_states = outputs.last_hidden_state

    # mean pooling
    embedding = last_hidden_states.mean(dim=1)

    return embedding.squeeze().numpy()


# generate embeddings
def generate_embeddings(texts, tokenizer, model, language_name):

    embeddings = []

    print(f"\nGenerating {language_name} embeddings...")

    for index, text in enumerate(texts):

        embedding = get_embedding(
            text,
            tokenizer,
            model
        )

        embeddings.append(embedding)

        print(
            f"{language_name} text "
            f"{index + 1}/{len(texts)} processed"
        )

    return np.array(embeddings)


# compute similarity matrix
def compute_similarity_matrix(
    english_embeddings,
    arabic_embeddings
):
    all_embeddings = np.vstack([
        english_embeddings,
        arabic_embeddings
    ])

    similarity_matrix = cosine_similarity(
        all_embeddings
    )

    return similarity_matrix


# create labels
def create_labels(english_texts, arabic_texts):

    english_labels = [
        f"EN{i+1}: "
        + text[:40].replace("\n", " ")
        for i, text in enumerate(english_texts)
    ]

    arabic_labels = [
        f"AR{i+1}: "
        + text[:40].replace("\n", " ")
        for i, text in enumerate(arabic_texts)
    ]

    return english_labels + arabic_labels


# plot heatmap
def plot_heatmap(
    similarity_matrix,
    labels,
    output_path
):

    print("\nCreating heatmap...")

    plt.figure(figsize=(16, 14))

    sns.heatmap(
        similarity_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap="viridis"
    )

    plt.title(
        "Cross-Lingual Similarity Heatmap\n"
        "bert-base-multilingual-cased"
    )

    plt.xticks(rotation=90, fontsize=8)

    plt.yticks(rotation=0, fontsize=8)

    plt.tight_layout()

    plt.savefig(output_path, dpi=300)

    print(f"Heatmap saved to: {output_path}")


# analyze cross-lingual similarity
def analyze_cross_lingual_similarity(
    english_texts,
    arabic_texts,
    english_embeddings,
    arabic_embeddings
):

    cross_similarity = cosine_similarity(
        english_embeddings,
        arabic_embeddings
    )

    print("\n" + "=" * 70)
    print("TOP CROSS-LINGUAL MATCHES")
    print("=" * 70)

    for i, english_text in enumerate(english_texts):

        similarities = cross_similarity[i]

        best_match_index = np.argmax(similarities)

        best_score = similarities[best_match_index]

        print("\n" + "-" * 70)

        print(f"ENGLISH ARTICLE {i+1}")

        print(
            english_text[:200]
            .replace("\n", " ")
        )

        print("\nBEST ARABIC MATCH")

        print(
            arabic_texts[best_match_index][:200]
            .replace("\n", " ")
        )

        print(
            f"\nCOSINE SIMILARITY: "
            f"{best_score:.4f}"
        )

    return cross_similarity


# same-topic vs random pairs
def compare_same_vs_random(
    cross_similarity,
    english_texts,
    arabic_texts
):

    same_topic_scores = []

    random_scores = []

    print("\n" + "=" * 70)
    print("SAME-TOPIC VS RANDOM PAIRS")
    print("=" * 70)

    for i in range(
        min(
            len(english_texts),
            len(arabic_texts)
        )
    ):

        # diagonal pair
        same_score = cross_similarity[i][i]

        same_topic_scores.append(same_score)

        # random pair
        random_index = (
            (i + 3)
            % len(arabic_texts)
        )

        random_score = (
            cross_similarity[i][random_index]
        )

        random_scores.append(random_score)

        print(f"\nPair {i+1}")

        print(
            f"Same-topic similarity: "
            f"{same_score:.4f}"
        )

        print(
            f"Random-pair similarity: "
            f"{random_score:.4f}"
        )

    avg_same = np.mean(same_topic_scores)

    avg_random = np.mean(random_scores)

    print("\n" + "=" * 70)

    print(
        f"Average same-topic similarity: "
        f"{avg_same:.4f}"
    )

    print(
        f"Average random-pair similarity: "
        f"{avg_random:.4f}"
    )

    print("=" * 70)


# main function
def main():

    # load data
    df = load_data(DATA_PATH)

    # text column
    text_column = "text"

    # filter languages
    english_df, arabic_df = filter_language_data(
        df,
        text_column
    )

    # select texts
    english_texts, arabic_texts = select_texts(
        english_df,
        arabic_df,
        text_column,
        NUM_TEXTS_PER_LANGUAGE
    )

    # load model
    tokenizer, model = load_model(MODEL_NAME)

    # generate embeddings
    english_embeddings = generate_embeddings(
        english_texts,
        tokenizer,
        model,
        "English"
    )

    arabic_embeddings = generate_embeddings(
        arabic_texts,
        tokenizer,
        model,
        "Arabic"
    )

    # similarity matrix
    similarity_matrix = compute_similarity_matrix(
        english_embeddings,
        arabic_embeddings
    )

    # labels
    labels = create_labels(
        english_texts,
        arabic_texts
    )

    # heatmap
    plot_heatmap(
        similarity_matrix,
        labels,
        OUTPUT_HEATMAP
    )

    # cross-lingual analysis
    cross_similarity = analyze_cross_lingual_similarity(
        english_texts,
        arabic_texts,
        english_embeddings,
        arabic_embeddings
    )

    # same-topic vs random
    compare_same_vs_random(
        cross_similarity,
        english_texts,
        arabic_texts
    )

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()