import argparse

from core.corpus import Corpus
from analyses import basic_stats, entity_analysis
from reporting import printer


def run_stats(filepath1: str, filepath2: str) -> None:
    corpus1 = Corpus(filepath1)
    corpus2 = Corpus(filepath2)

    results1 = basic_stats.analyze(corpus1)
    results2 = basic_stats.analyze(corpus2)

    printer.print_basic_stats_comparison(results1, filepath1, results2, filepath2)


def run_entities(filepath1: str, filepath2: str) -> None:
    corpus1 = Corpus(filepath1)
    corpus2 = Corpus(filepath2)

    results1 = entity_analysis.analyze(corpus1)
    results2 = entity_analysis.analyze(corpus2)

    printer.print_entity_analysis(results1, filepath1)
    printer.print_entity_analysis(results2, filepath2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two text files using basic linguistic statistics or entity analysis"
    ) 
    #@TODO: add sentiment analysis like on the other side project
    
    parser.add_argument("mode", choices=["stats", "entities"], help="which analysis to run")
    parser.add_argument("file1", help="path to the first text file")
    parser.add_argument("file2", help="path to the second text file")
    args = parser.parse_args()

    if args.mode == "stats":
        run_stats(args.file1, args.file2)
    else:
        run_entities(args.file1, args.file2)


if __name__ == "__main__":
    main()