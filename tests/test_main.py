from main import run_stats, run_entities

SAMPLE_FILE = "tests/sample/corpus-1.txt"


def test_run_stats_does_not_crash(capsys):
    run_stats(SAMPLE_FILE, SAMPLE_FILE)
    captured = capsys.readouterr()
    assert "sentence" in captured.out.lower()


def test_run_entities_does_not_crash(capsys):
    run_entities(SAMPLE_FILE, SAMPLE_FILE)
    captured = capsys.readouterr()
    assert len(captured.out) > 0