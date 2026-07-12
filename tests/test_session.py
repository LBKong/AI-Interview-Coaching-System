from server import config
from server.session import pick_question


def test_pick_question_returns_one_from_bank():
    q = pick_question()
    assert q in config.QUESTIONS


def test_pick_question_deterministic_with_index():
    assert pick_question(index=0) == config.QUESTIONS[0]
    assert pick_question(index=1) == config.QUESTIONS[1]
