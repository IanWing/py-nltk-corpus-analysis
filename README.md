# py-nltk-analysis

A comparative linguistic analysis tool for English text files, built on top of [NLTK](https://www.nltk.org/).
Given two text files, it computes and prints either general linguistic statistics or entity-focused insights, side by side.

This project started as a computational linguistics exam assignment in 2020, and has since been rewritten from scratch with a modular architecture, proper caching and with better integration of python's tools.

## What it does

Two independent analysis, run against a pair of text files:

**`stats`** - general linguistic statistics:
- Sentence and token counts, average sentence/token length
- Vocabulary growth over the length of the text
- Verb/noun ratio
- Most frequent part-of-speech (PoS) tags
- Most likely PoS bigrams (conditional probability)
- Strongest PoS associations (Local Mutual Information)

**`entities`** - named-entity-focused analysis:
- The most frequently mentioned people in each text
- For each person: related people, locations, verbs, nouns, and dates found in the sentences that mention them
- The shortest and longest sentence mentioning each person
- A simple order-0 "most probable sentence" estimate (a token-frequency-based heuristic, not a real Markov model)

## Setup

Requires **Python 3.11+**

```bash
python3 -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
python scripts/setup_nltk_data.py
```

The setup script downloads all NLTK data resources the project depends on (tokenizer, PoS tagger, NER chunker, word list) in one go.


### Troubleshooting

**`ImportError: Blocked import ... from current working directory`**
during `pip install` or when running the setup script:

This comes from a security feature in recent NLTK versions
(`nltk/inisec.py`) that blocks imports resolved from the current working
directory, which includes packages installed inside a `venv/` nested in
the project folder.

Work around it by disabling the check for your session before running
any NLTK-related command:

    ```bash
    # Windows (PowerShell)
    $env:NLTK_DISABLE_IMPORT_SECURITY = "1"

    # macOS / Linux
    export NLTK_DISABLE_IMPORT_SECURITY=1
    ```

## Usage

```bash
python main.py stats file1.txt file2.txt
python main.py entities file1.txt file2.txt
```

Both files must be plain-text, UTF-8 encoded, in English (the underlying NLTK models are English-only).

Example:
```bash
python main.py stats tests/sample/corpus-1.txt tests/sample/corpus-2.txt
```

*Please note that the files corpus-1 and corpus-2 provided are fully AI generated, while Alice in Wonderland and Through the Looking-Glass are provided by the Gutemberg Project.*

## Notes and known limitations

- **The project was originally developed in 2020** by me, while I was much inexperienced. The main outcome of the project was to interface with one or more corpora and create a basic study for them, with simple data analysis.
  - In **August 2026**, I made a major refactor to improve repository structure and maintainability.
- This repository was created exclusively for the Computation Linguistics exam and is intended solely for *educational and demonstrative purposes*.

## License

This project builds on NLTK (Apache License 2.0). See NLTK's own licensing for the underlying models and corpora. The corpora offered

### Sample data

The sample corpora in `tests/sample/` derived from Project Gutenberg texts
(*Alice's Adventures in Wonderland* and *Through the Looking-Glass* by Lewis
Carroll) are in the public domain in the United States. The original,
unmodified e-book files are available free of charge at
https://www.gutenberg.org — see the Project Gutenberg License at
https://www.gutenberg.org/policy/license.html for redistribution terms.

The versions used here have been stripped of Project Gutenberg's standard
header/footer for text-processing purposes.