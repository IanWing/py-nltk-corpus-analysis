from analyses import basic_stats


def test_average_sentence_length_is_positive(sample_corpus):
    result = basic_stats.average_sentence_length(sample_corpus)
    assert result > 0


def test_average_sentence_length_matches_manual_count(sample_corpus):
    expected = len(sample_corpus.tokens) / len(sample_corpus.sentences)
    assert basic_stats.average_sentence_length(sample_corpus) == expected


def test_average_token_length_is_reasonable(sample_corpus):
    result = basic_stats.average_token_length(sample_corpus)
    assert 1 < result < 15


def test_vocabulary_growth_is_non_decreasing(sample_corpus):
    growth = basic_stats.vocabulary_growth(sample_corpus, step=5)
    assert growth == sorted(growth)


def test_vocabulary_growth_last_value_matches_full_vocabulary(sample_corpus):
    growth = basic_stats.vocabulary_growth(sample_corpus, step=5)
    assert growth[-1] == len(set(sample_corpus.tokens))


def test_verb_noun_ratio_is_positive(sample_corpus):
    result = basic_stats.verb_noun_ratio(sample_corpus)
    assert result > 0


def test_most_common_pos_tags_returns_requested_count(sample_corpus):
    result = basic_stats.most_common_pos_tags(sample_corpus, top_n=5)
    assert len(result) <= 5
    for tag, count in result:
        assert isinstance(tag, str)
        assert isinstance(count, int)
        assert count > 0


def test_most_common_pos_tags_are_sorted_descending(sample_corpus):
    result = basic_stats.most_common_pos_tags(sample_corpus, top_n=10)
    counts = [count for _, count in result]
    assert counts == sorted(counts, reverse=True)


def test_hapax_ratio_is_between_zero_and_one(sample_corpus):
    result = basic_stats.hapax_ratio(sample_corpus, limit=len(sample_corpus.tokens))
    assert 0 <= result <= 1


def test_most_likely_pos_bigrams_probabilities_are_valid(sample_corpus):
    result = basic_stats.most_likely_pos_bigrams(sample_corpus, top_n=5)
    for bigram, probability in result:
        assert 0 <= probability <= 1


def test_analyze_returns_all_expected_keys(sample_corpus):
    result = basic_stats.analyze(sample_corpus)
    expected_keys = {
        "num_sentences", "num_tokens", "avg_sentence_length",
        "avg_token_length", "vocabulary_growth", "verb_noun_ratio",
        "most_common_pos_tags", "most_likely_pos_bigrams",
        "strongest_pos_associations",
    }
    assert set(result.keys()) == expected_keys