# analyses/entity_analysis.py

#1. for each notable person mentioned,finds the sentences that reference them, 
#2. extracts related people, locations, verbs, nouns, dates, and the most probable sentence (0 order Markov estimate) associated with each


import re
from collections import Counter

import nltk

DATE_PATTERN = re.compile(
    r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
    r"\.?\s*\d{0,2}[,]?\s*\d{0,4}"
    r"|\b(?:Mon|Tue|Wed(?:nes)?|Thu(?:rs)?|Fri|Sat(?:ur)?|Sun)day\b"
    r"|\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b"
)


def extract_named_entities(tagged_sentences) -> dict:
    """
    iterate through a list of PoS-tagged, chunked sentences and collect named entities by type;
    output: {"PERSON": [...], "GPE": [...]}
    """
    entities_by_type = {"PERSON": [], "GPE": []}
    for tagged_sentence in tagged_sentences:
        for chunk in nltk.ne_chunk(tagged_sentence):
            if isinstance(chunk, nltk.tree.Tree) and chunk.label() in entities_by_type:
                entity_text = " ".join(token for token, tag in chunk)
                entities_by_type[chunk.label()].append(entity_text)
    return entities_by_type


def top_person_names(corpus, top_n: int = 10) -> list[str]:
    # The top_n most frequently mentioned person names in the corpus, based on NER over each individually tagged sentence.

    tagged_sentences = [
        nltk.pos_tag(nltk.word_tokenize(sentence)) for sentence in corpus.sentences
    ]
    entities = extract_named_entities(tagged_sentences)
    name_counts = Counter(entities["PERSON"])
    return [name for name, _ in name_counts.most_common(top_n)]


def sentences_mentioning(names: list[str], sentences: list[str]) -> dict:
    #for each name, collect every sentence that mentions it, the shortest and longest sentence
    # {name: {"all": [...], "shortest": str, "longest": str}}

    result = {}
    for name in names:
        matching = [sentence for sentence in sentences if name in sentence]
        if not matching:
            result[name] = {"all": [], "shortest": None, "longest": None}
            continue
        result[name] = {
            "all": matching,
            "shortest": min(matching, key=len),
            "longest": max(matching, key=len),
        }
    return result


def _extract_dates(text: str) -> list[str]:
    return DATE_PATTERN.findall(text)


def _most_probable_sentence(sentences: list[str], token_frequencies: Counter, total_tokens: int) -> str | None:
    """
    markov-0 estimate: 
    among sentences of length 8-12 tokens, 
    return the one whose tokens have the highest joint probability.
    """
    best_sentence = None
    best_probability = 0.0

    for sentence in sentences:
        tokens = nltk.word_tokenize(sentence)
        if not (8 <= len(tokens) <= 12):
            continue

        probability = 1.0
        for token in tokens:
            token_probability = token_frequencies[token] / total_tokens
            probability *= token_probability

        if probability > best_probability:
            best_probability = probability
            best_sentence = sentence

    return best_sentence


def analyze_entity(name: str, sentences: list[str], corpus, top_n: int = 10) -> dict:

    #related people, locations, verbs,nouns, dates, most probable sentence, for a name.
   
    tagged_sentences = [
        nltk.pos_tag(nltk.word_tokenize(sentence)) for sentence in sentences
    ]

    verbs, nouns = [], []
    for tagged_sentence in tagged_sentences:
        for token, tag in tagged_sentence:
            if tag.startswith("VB"):
                verbs.append(token)
            elif tag.startswith("NN"):
                nouns.append(token)

    entities = extract_named_entities(tagged_sentences)
    dates = [date for sentence in sentences for date in _extract_dates(sentence)]

    return {
        "related_people": Counter(entities["PERSON"]).most_common(top_n),
        "related_locations": Counter(entities["GPE"]).most_common(top_n),
        "top_verbs": Counter(verbs).most_common(top_n),
        "top_nouns": Counter(nouns).most_common(top_n),
        "dates": Counter(dates).most_common(top_n),
        "most_probable_sentence": _most_probable_sentence(
            sentences, corpus.token_frequencies, len(corpus.tokens)
        ),
    }


def analyze(corpus, top_n: int = 10) -> dict:
    # find the top mentioned people then  analyze the sentences that mention them

    # {name: {"shortest_sentence": ..., "longest_sentence": ..., **analyze_entity output}}

    names = top_person_names(corpus, top_n=top_n)
    sentence_map = sentences_mentioning(names, corpus.sentences)

    results = {}
    for name in names:
        entry = sentence_map[name]
        entity_result = analyze_entity(name, entry["all"], corpus, top_n=top_n)
        entity_result["shortest_sentence"] = entry["shortest"]
        entity_result["longest_sentence"] = entry["longest"]
        results[name] = entity_result

    return results