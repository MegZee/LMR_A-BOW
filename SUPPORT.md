# Support Guide

This document explains the purpose of the main Python scripts in the repository and how they fit together when preparing linguistic metonymy resolution datasets.

## Repository Workflow Overview

1. **Data preparation (`dataPrep.py`)**
   * Loads labelled JSON datasets (e.g., `dataset/wimcor_test.json`).
   * Builds a pandas DataFrame with each sentence, its metonymy label (`0` literal, `1` metonymic), and the target token position.
   * Splits the DataFrame into literal and metonymic subsets and serialises the sentences and target positions to pickle files under `contextWords/<dataset>/preps/` for downstream use.

2. **Context window construction (`immediates.py` & `wordsSelecation.py`)**
   * `immediates.py` pads each sentence with a fixed window (`base_length`) of tokens to the left and right of the target position, emitting tuples of left/right context tokens for baseline models. The padded windows are stored as pickles in `baseline/<dataset>/`.
   * `wordsSelecation.py` loads curated GloVe feature lists and identifies overlapping features in each sentence. If a sentence contains more than one feature word, the intersection is recorded; otherwise, the function falls back to the padded left/right context window (length defined by `base`).

3. **Vocabulary filtering and frequency analysis (`cleaning.py` & `preprocessing.py`)**
   * Both scripts flatten the stored context windows, remove common stopwords, and compute token frequency dictionaries using simple counting helpers.
   * `cleaning.py` prints frequency tables for analysis, while `preprocessing.py` focuses on sorted frequency output for metonymic vs. literal contexts.

4. **GloVe feature expansion (`GloVeFeatures.py`)**
   * Loads seed metonymic and literal word lists via spaCy, retrieves their GloVe embeddings, and expands each list by finding the closest neighbouring vectors (excluding stop words and manually curated exclusions).
   * Provides helper output and (commented) pickle dump commands to persist the expanded feature sets in `Features extracted/`.

## Key Data Dependencies

* **GloVe vectors**: Expect `glove.6B.50d/glove.6B.50d.txt` locally to compute embedding neighbours.
* **Pickled context data**: Scripts read and write to `contextWords/` and `baseline/` directories. Ensure these folders exist with the expected dataset subdirectories (e.g., `relocar`, `wimcor`).
* **spaCy model**: Several scripts require the English model; install with `python -m spacy download en_core_web_sm` before running.

## Running the Pipeline

1. Prepare JSON datasets and run `dataPrep.py` to produce sentence/position pickles.
2. Generate context windows (`immediates.py`) or feature-aware contexts (`wordsSelecation.py`).
3. Use `cleaning.py` or `preprocessing.py` to inspect token frequencies for feature selection.
4. Optionally run `GloVeFeatures.py` to expand feature vocabularies using GloVe embeddings and save the results for later ingestion.

This high-level flow should help you trace the origin of each intermediate file and understand how the scripts collaborate to support metonymy resolution experiments.
