"""
- sentence/token lengths,
- vocabulary growth,
- PoS distribution
- PoS bigram statistics (conditional probability and Local Mutual Information).
"""

import math
from collections import Counter
from nltk.util import bigrams


def average_sentence_length(corpus) -> float:
    # average number of tokens per sentence
    return len(corpus.tokens) / len(corpus.sentences)


def average_token_length(corpus) -> float:
    # average number of characters per token
    total_chars = sum(len(token) for token in corpus.tokens)
    return total_chars / len(corpus.tokens)


def vocabulary_growth(corpus, step: int = 1000) -> list[int]:
   # Vocabulary size (|unique tokens|) measured every |step| tokens
    growth = []
    for i in range(step, len(corpus.tokens), step):
        growth.append(len(set(corpus.tokens[:i])))
    growth.append(len(set(corpus.tokens)))
    return growth


def hapax_ratio(corpus, limit: int) -> float:
    #    ratio of hapax legomena (words occurring exactly once) within the first |limit| tokens, relative to total corpus length
    freq_dist = Counter(corpus.tokens[:limit])
    hapaxes = [word for word, count in freq_dist.items() if count == 1]
    return len(hapaxes) / len(corpus.tokens)


def verb_noun_ratio(corpus) -> float:
    # Ratio of verbs to nouns, based on PoS tag prefixes (VB.. / NN..)
    verb_count = sum(1 for _, tag in corpus.pos_tags if tag.startswith("VB"))
    noun_count = sum(1 for _, tag in corpus.pos_tags if tag.startswith("NN"))
    return verb_count / noun_count


def most_common_pos_tags(corpus, top_n: int = 10) -> list[tuple[str, int]]:
    # |top_n| most frequent PoS tags and their frequency
    tag_freq = Counter(tag for _, tag in corpus.pos_tags)
    return tag_freq.most_common(top_n)



def _pos_sequence(corpus) -> list[str]:
    # extract just the PoS tags, in order, from the tagged corpus
    return [tag for _, tag in corpus.pos_tags]


def most_likely_pos_bigrams(corpus, top_n: int = 10) -> list[tuple[tuple[str, str], float]]:
    #  |top_n| PoS bigrams with the highest conditional probability --- P(second_tag | first_tag)

    pos_sequence = _pos_sequence(corpus)
    pos_bigrams = list(bigrams(pos_sequence))
    pos_tag_freq = Counter(pos_sequence)

    conditional_probs = {}
    for bigram in set(pos_bigrams):
        bigram_freq = pos_bigrams.count(bigram)
        first_tag_freq = pos_tag_freq[bigram[0]]
        conditional_probs[bigram] = bigram_freq / first_tag_freq

    sorted_probs = sorted(conditional_probs.items(), key=lambda item: -item[1])
    return sorted_probs[:top_n]


def strongest_pos_associations(corpus, top_n: int = 10) -> list[tuple[tuple[str, str], float]]:
    
    # |top_n| PoS bigrams with the highest Local Mutual Information

    
    pos_sequence = _pos_sequence(corpus)
    pos_bigrams = list(bigrams(pos_sequence))
    bigram_freq = Counter(pos_bigrams)
    tag_freq = Counter(pos_sequence)
    total_tags = len(pos_sequence)

    lmi_scores = {}
    for bigram, freq in bigram_freq.items():
        prob_bigram = freq / total_tags
        prob_first = tag_freq[bigram[0]] / total_tags
        prob_second = tag_freq[bigram[1]] / total_tags
        lmi = freq * math.log2(prob_bigram / (prob_first * prob_second))
        lmi_scores[bigram] = lmi

    sorted_scores = sorted(lmi_scores.items(), key=lambda item: -item[1])
    return sorted_scores[:top_n]


def analyze(corpus, vocabulary_step: int = 1000) -> dict:
    return {
        "num_sentences": len(corpus.sentences),
        "num_tokens": len(corpus.tokens),
        "avg_sentence_length": average_sentence_length(corpus),
        "avg_token_length": average_token_length(corpus),
        "vocabulary_growth": vocabulary_growth(corpus, step=vocabulary_step),
        "verb_noun_ratio": verb_noun_ratio(corpus),
        "most_common_pos_tags": most_common_pos_tags(corpus),
        "most_likely_pos_bigrams": most_likely_pos_bigrams(corpus),
        "strongest_pos_associations": strongest_pos_associations(corpus),
    }