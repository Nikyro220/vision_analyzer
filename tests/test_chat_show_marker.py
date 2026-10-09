"""[SHOW: ...] в ответе модели: карточки под ответом = только подтверждённые моделью анализы."""

from vision_app.chat_tools.runner import apply_show_marker

CARDS = [{"id": 2}, {"id": 7}, {"id": 16}, {"id": 21}, {"id": 30}]


def test_keeps_only_confirmed_cards_and_strips_marker():
    reply, cards = apply_show_marker("Подходят два анализа.\n\n[SHOW: 2, 16]", CARDS)
    assert reply == "Подходят два анализа."
    assert [c["id"] for c in cards] == [2, 16]


def test_empty_marker_means_no_cards():
    reply, cards = apply_show_marker("Ничего не найдено. [SHOW: ]", CARDS)
    assert reply == "Ничего не найдено."
    assert cards == []


def test_no_marker_leaves_cards_untouched():
    reply, cards = apply_show_marker("Вот последние анализы.", CARDS)
    assert reply == "Вот последние анализы."
    assert cards == CARDS


def test_tolerates_hash_case_and_ids_without_cards():
    reply, cards = apply_show_marker("Готово [show: #7, #999]", CARDS)
    assert reply == "Готово"
    assert [c["id"] for c in cards] == [7]
