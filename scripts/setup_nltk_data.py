import nltk

REQUIRED_RESOURCES = [
    "punkt",
    "punkt_tab",
    "averaged_perceptron_tagger_eng",
    "maxent_ne_chunker_tab",
    "words",
]

for resource in REQUIRED_RESOURCES:
    nltk.download(resource)