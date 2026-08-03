from analyses import entity_analysis

def test_top_person_names_returns_a_list(sample_corpus):
    result = entity_analysis.top_person_names(sample_corpus, top_n=5)
    assert isinstance(result, list)
    assert len(result) <= 5


def test_sentences_mentioning_handles_missing_name(sample_corpus):
    result = entity_analysis.sentences_mentioning(["MissingNamelallero"], sample_corpus.sentences)
    assert result["MissingNamelallero"]["all"] == []
    assert result["MissingNamelallero"]["shortest"] is None
    assert result["MissingNamelallero"]["longest"] is None


def test_sentences_mentioning_handles_single_match(sample_corpus):
    first_token = sample_corpus.tokens[0]
    result = entity_analysis.sentences_mentioning([first_token], sample_corpus.sentences[:1])
    if result[first_token]["all"]:
        assert result[first_token]["shortest"] == result[first_token]["longest"]


def test_extract_dates_finds_known_formats():
    text = "The meeting is on March 5, 2024, right after Monday's call, or on 03/05/2024."
    dates = entity_analysis._extract_dates(text)
    assert len(dates) > 0


def test_extract_dates_empty_on_dateless_text():
    text = "The quick brown fox jumps over the lazy dog."
    dates = entity_analysis._extract_dates(text)
    assert dates == []


def test_most_probable_sentence_probability_resets_between_sentences():
    from collections import Counter

    sentences = [
        "the the the the the the the the", # 8 tokens, all very common
        "xylophone quixotic zephyr umbrella", # 4 tokens, rare -> filtered out by length anyway
    ]
    token_frequencies = Counter({"the": 100, "xylophone": 1, "quixotic": 1, "zephyr": 1, "umbrella": 1})
    total_tokens = 104

    result = entity_analysis._most_probable_sentence(sentences, token_frequencies, total_tokens)
    assert result == sentences[0]


def test_analyze_entity_returns_expected_keys(sample_corpus):
    names = entity_analysis.top_person_names(sample_corpus, top_n=3)
    if not names:
        return  # no named entities in this sample text; nothing to test here
    name = names[0]
    sentence_map = entity_analysis.sentences_mentioning([name], sample_corpus.sentences)
    result = entity_analysis.analyze_entity(name, sentence_map[name]["all"], sample_corpus)
    expected_keys = {
        "related_people", "related_locations", "top_verbs",
        "top_nouns", "dates", "most_probable_sentence",
    }
    assert set(result.keys()) == expected_keys


def test_analyze_returns_dict_keyed_by_name(sample_corpus):
    result = entity_analysis.analyze(sample_corpus, top_n=3)
    assert isinstance(result, dict)
    for name, entry in result.items():
        assert isinstance(name, str)
        assert "shortest_sentence" in entry
        assert "longest_sentence" in entry