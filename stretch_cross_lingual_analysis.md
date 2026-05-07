# Cross-Lingual Embedding Analysis (BERT Multilingual)

## 1. Cross-lingual similarity quality

The results indicate that `bert-base-multilingual-cased` captures a moderate level of semantic alignment between English and Arabic climate-related texts. The cosine similarity scores for cross-lingual pairs range from approximately **0.5942 to 0.7516**, with the strongest alignment observed in clearly parallel climate topics such as IPCC-related content (0.7516) and food security reports (0.7327). These higher scores suggest that the model is capable of mapping semantically similar content across languages into a partially shared embedding space.

However, the alignment is inconsistent. Several English articles are matched with Arabic texts that are not closely aligned in topic (for example, climate policy content paired with biodiversity frameworks or unrelated climate governance texts). This indicates that while the model captures general domain similarity (climate change), it does not reliably preserve fine-grained topic alignment across languages.

---

## 2. Same-topic vs random-pair comparison and implications

The quantitative comparison between same-topic and random cross-lingual pairs shows only a slight difference. The average similarity for same-topic pairs is **0.6286**, while the average similarity for random pairs is **0.6098**. This narrow gap suggests that the model does not strongly distinguish between aligned and misaligned cross-lingual topic pairs in embedding space.

Additionally, some random pairs even outperform same-topic pairs, which further highlights the instability of cross-lingual alignment at the sentence embedding level. This implies that while multilingual BERT provides a useful shared representation space, it is not sufficiently precise for reliable cross-lingual retrieval or classification without further fine-tuning.

For NLP applications in the MENA region, this has important implications. A single multilingual model can support baseline bilingual search and clustering, but it may not be accurate enough for production-level systems that require precise semantic matching (e.g., policy analysis or climate monitoring tools). Improving performance would likely require domain-specific fine-tuning, contrastive learning approaches, or alignment techniques designed specifically for cross-lingual embedding consistency.
