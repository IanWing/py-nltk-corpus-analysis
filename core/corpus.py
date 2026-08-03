import nltk
from collections import Counter

sent_tokenizer = nltk.data.load("tokenizers/punkt/english.pickle")


class Corpus:
    def __init__(self, filepath: str):
        self.filepath = filepath

        with open(filepath, "r", encoding="utf-8") as file:
            self.text = file.read()

        self.sentences = sent_tokenizer.tokenize(self.text)
        self.tokens = nltk.word_tokenize(self.text)

        self._pos_tags = None
        self._named_entities = None
        self._token_frequencies = None

    @property
    def pos_tags(self):
        #List of (token, tag) tuples for the whole corpus
        if self._pos_tags is None:
            self._pos_tags = nltk.pos_tag(self.tokens)
        return self._pos_tags

    @property
    def named_entities(self):
        #NER chunk tree (people, places...)
        if self._named_entities is None:
            self._named_entities = nltk.ne_chunk(self.pos_tags)
        return self._named_entities

    @property
    def token_frequencies(self):
        #counter of token frequencies, for O(1) lookups instead of list.count()
        if self._token_frequencies is None:
            self._token_frequencies = Counter(self.tokens)
        return self._token_frequencies

    def __repr__(self):
        return f"Corpus({self.filepath!r}, {len(self.tokens)} tokens, {len(self.sentences)} sentences)"