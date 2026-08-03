def _section_header(title: str) -> None:
    print(f"\n{'-' * 3} {title} {'-' * 3}".center(100))


def _compare_values(label1: str, value1: float, label2: str, value2: float, metric_name: str) -> None:
    print(f"{label1}: {value1}")
    print(f"{label2}: {value2}")
    if value1 > value2:
        print(f"{label1} has a higher {metric_name} than {label2}.")
    elif value1 < value2:
        print(f"{label1} has a lower {metric_name} than {label2}.")
    else:
        print(f"{label1} and {label2} have the same {metric_name}.")




# !!!!
def print_basic_stats_comparison(results1: dict, name1: str, results2: dict, name2: str) -> None:
    _section_header("Sentence and token counts")
    _compare_values(name1, results1["num_sentences"], name2, results2["num_sentences"], "number of sentences")
    print()
    _compare_values(name1, results1["num_tokens"], name2, results2["num_tokens"], "number of tokens")

    _section_header("Average lengths")
    _compare_values(
        name1, results1["avg_sentence_length"], name2, results2["avg_sentence_length"],
        "average sentence length",
    )
    print()
    _compare_values(
        name1, results1["avg_token_length"], name2, results2["avg_token_length"],
        "average token length",
    )

    _section_header("Verb/noun ratio")
    _compare_values(name1, results1["verb_noun_ratio"], name2, results2["verb_noun_ratio"], "verb/noun ratio")

    for label, results, name in [("first", results1, name1), ("second", results2, name2)]:
        _section_header(f"Most frequent PoS tags ({name})")
        for rank, (tag, count) in enumerate(results["most_common_pos_tags"], start=1):
            print(f"{rank}.\t{tag}\tfrequency: {count}")

    for label, results, name in [("first", results1, name1), ("second", results2, name2)]:
        _section_header(f"Most likely PoS bigrams ({name})")
        for rank, (bigram, probability) in enumerate(results["most_likely_pos_bigrams"], start=1):
            print(f"{rank}.\t{bigram}\tconditional probability: {probability:.2f}")

        _section_header(f"Strongest PoS associations - Local Mutual Information ({name})")
        for rank, (bigram, lmi) in enumerate(results["strongest_pos_associations"], start=1):
            print(f"{rank}.\t{bigram}\tLMI: {lmi:.2f}")


def _print_entity_category(items: list, category_plural: str, category_singular: str, entity_name: str) -> None:
    count = len(items)
    if count == 0:
        print(f"No {category_plural} found in sentences mentioning '{entity_name}'.")
        return
    elif count == 1:
        print(f"Only one {category_singular} found in sentences mentioning '{entity_name}':")
    else:
        print(f"Top {category_plural} found in sentences mentioning '{entity_name}':")

    for rank, (item, freq) in enumerate(items, start=1):
        print(f"{rank}.\t{item}\tfrequency: {freq}")
    print()





#!!!!!
def print_entity_analysis(results: dict, corpus_name: str) -> None:
    _section_header(f"Entity analysis for {corpus_name}")

    for name, entry in results.items():
        print(f"\n--- {name} ---")

        if entry["shortest_sentence"]:
            print(f"Shortest sentence:  \"{entry['shortest_sentence']}\"")
        if entry["longest_sentence"]:
            print(f"Longest sentence: \"{entry['longest_sentence']}\"")

        _print_entity_category(entry["related_people"], "people", "person", name)
        _print_entity_category(entry["related_locations"], "locations", "location", name)
        _print_entity_category(entry["top_verbs"], "verbs", "verb", name)
        _print_entity_category(entry["top_nouns"], "nouns", "noun", name)
        _print_entity_category(entry["dates"], "dates", "date", name)

        if entry["most_probable_sentence"]:
            print(f"Most probable sentence (order-0 estimate): \"{entry['most_probable_sentence']}\"")
        else:
            print("No sentence of typical length (8-12 tokens) found.")