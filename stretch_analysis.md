# Custom NER (EntityRuler) Analysis

## Overview
In this stretch assignment, I extended spacy base NER system using a custom `EntityRuler` to capture domain-specific climate entities that are not reliably detected by the pre-trained model. The custom pipeline introduces new entity types such as AGREEMENT, CLIMATE_EVENT, REPORT, and THRESHOLD, and is evaluated in both "before NER" and "after NER" configurations.

---

## Impact of Custom Rules

The EntityRuler successfully expanded entity coverage by capturing important climate-specific concepts:

- **AGREEMENT**: “Paris Agreement”, “Kyoto Protocol”
- **CLIMATE_EVENT**: “COP28”, “COP27”
- **REPORT**: “IPCC AR6”, “Sixth Assessment Report”
- **THRESHOLD**: numeric climate indicators such as “31%”, “75%”, and temperature-related values

These entities are not consistently detected by the base spacy model, confirming the value of rule-based augmentation for domain-specific NLP tasks.

---

## Before vs After Pipeline Behavior

The placement of the EntityRuler significantly affected results:

- When applied **before the NER component**, rule-based entities were consistently preserved and appeared in the output. This configuration produced the most complete coverage of custom climate terms.
- When applied **after the NER component**, the statistical model sometimes overshadowed or suppressed rule-based matches, resulting in fewer detected custom entities.

This demonstrates the importance of pipeline ordering in spaCy when combining rule-based and statistical components.

---

## THRESHOLD Improvements

A key improvement in this iteration was enhancing the THRESHOLD pattern to better capture real-world climate expressions. The updated rules successfully extracted values such as percentages (e.g., “31%”, “75%”) and temperature-related thresholds. This significantly improved domain relevance by capturing quantitative climate indicators that were previously missed.

---

## Evaluation Results

Evaluation was performed only on standard spaCy labels (ORG, GPE, DATE, etc.), as required by the assignment.

- Base model F1: ~0.091
- Before NER F1: ~0.092
- After NER F1: ~0.091

The small change in metrics is expected because custom entity types (e.g., AGREEMENT, CLIMATE_EVENT) are not included in the gold standard. Therefore, improvements are primarily qualitative rather than reflected in evaluation scores.

---

## Conclusion

The custom EntityRuler significantly improves the model’s ability to recognize climate-specific terminology that is absent from spacy’s training data. While it does not strongly affect standard evaluation metrics, it enhances domain coverage and interpretability. This demonstrates a key NLP engineering principle: combining statistical models with rule-based systems is essential for building robust, domain-aware information extraction pipelines.