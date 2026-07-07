# Detecting Suicidal Intent in Reddit Posts

A comparative study of interpretable machine learning classifiers for flagging posts that may express suicidal ideation, built as a tool to support (not replace) human review in mental-health monitoring. Bachelor's thesis in Econometrics & Data Science at Vrije Universiteit Amsterdam (2025).

## Overview

Many people in crisis express distress online before anywhere else. This project asks whether simple, transparent classifiers can reliably distinguish posts expressing suicidal ideation from ordinary posts, and — just as importantly — whether their decisions can be explained well enough to be trustworthy in a sensitive setting. The emphasis throughout is on interpretability and ethics, not just raw accuracy.

## What it does

- Cleans and normalises a large Reddit corpus (lowercasing, punctuation/digit removal, tokenisation and lemmatisation with `spaCy`), deliberately retaining emotionally informative stopwords such as *I*, *me*, *myself*, *because*.
- Vectorises the text with TF-IDF and evaluates the effect of vocabulary size (500 → 5000 features).
- Trains and compares four classifiers: Gaussian Naive Bayes, Logistic Regression, k-Nearest Neighbours, and Random Forest, with hyperparameters tuned on a held-out validation set (and GridSearchCV for Random Forest).
- Uses stratified splitting (70% train / 21% test / 9% validation) to preserve class balance and prevent leakage.
- Interprets predictions with TF-IDF co-occurrence analysis and LIME, checking that the model relies on psychologically meaningful language rather than spurious signals.

## Key findings

- **Random Forest** performed best overall (F1 ≈ 0.86, recall ≈ 0.85, accuracy ≈ 0.86), with Logistic Regression close behind.
- Recall is treated as the priority metric, since a missed high-risk post (false negative) is more costly than a false alarm.
- LIME confirmed the model keyed on psychologically salient terms (*kill*, *suicide*, *die*, *anymore*), supporting clinical face validity.
- The work argues that transparent, lightweight models can screen large volumes of content to surface posts for human review — provided any real deployment secures consent, protects privacy, and keeps expert oversight in the loop.

## Tech stack

Python · scikit-learn · spaCy · LIME · pandas · NumPy · Matplotlib

## Repository contents

This repository contains code only. The dataset and generated outputs (figures, LIME examples, tables) are not included; run the notebook against your own copy of the dataset to reproduce them.

```
Thesis-Kaka.ipynb   # full pipeline: preprocessing, TF-IDF, models, evaluation, interpretability
README.md
```

## Data & ethics

The analysis uses a publicly available Kaggle dataset of ~232,000 Reddit posts (balanced between r/SuicideWatch and r/teenagers), labelled by subreddit origin via weak supervision. The dataset is **not** redistributed in this repository, both to respect its source terms and because it contains sensitive user-generated content. It can be obtained from Kaggle: [Suicide Watch dataset](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch).

This project used only anonymised, publicly available posts (no usernames or timestamps) and was conducted in line with GDPR and Reddit's data-sharing policies. The models are intended as decision-support tools for human reviewers, not as automated diagnostic systems.

## Author

Varvara Kaka — BSc Econometrics & Data Science, VU Amsterdam
