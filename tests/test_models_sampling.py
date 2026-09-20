"""Выбор модели и параметры генерации на странице «Статус сервера»."""

import io

from conftest import login, logout, make_png, register
from vision_app.extensions import db
from vision_app.models import Role, Setting, User
from vision_app.services import extract_model_names, get_models, get_sampling


def upload(client):
    return client.post("/", data={"image": (io.BytesIO(make_png()), "p.png", "image/png")},
                       content_type="multipart/form-data")


def promote(app, username, role):
    with app.app_context():
        db.session.scalars(db.select(User).where(User.username == username)).one().role = role
        db.session.commit()


def as_boss(client):
    register(client, "boss")  # первый пользователь = head_admin


# ------------------------------------------------------------------ отображение
def test_status_page_shows_forms_for_available_backends(client, vision):
    as_boss(client)
    page = client.get("/health/").get_data(as_text=True)
    assert "google/gemma-4-31B-it" in page and "qwen2.5-vl" in page          # модели vLLM
    assert "llama3.2-vision" in page and "gemma3:27b" in page                # модели Ollama
    assert 'name="vllm-temperature"' in page and 'name="ollama-temperature"' in page
    # num_ctx только у Ollama
    assert 'name="ollama-num_ctx"' in page
    assert 'name="vllm-num_ctx"' not in page
    # текущие значения подставлены из /sampling
    assert 'value="0.2"' in page and 'value="8192"' in page
    # seed=None -> пустое поле
    assert 'name="vllm-seed"' in page
    assert "Используется для анализа" in page


def test_unavailable_backend_has_no_forms(client, vision):
    vision.ollama_ok = False
    as_boss(client)
    page = client.get("/health/").get_data(as_text=True)
    assert 'name="vllm-temperature"' in page
    assert 'name="ollama-temperature"' not in page
    assert "Настройки станут доступны, когда бэкенд заработает." in page
    # для недоступного бэкенда список моделей даже не запрашивается
    assert [c for c in vision.calls if c[1] == "/models" and c[2]["backend"] == "ollama"] == []


def test_regular_user_sees_status_but_no_settings(app, client, vision):
    as_boss(client); logout(client)
    register(client, "user1")
    page = client.get("/health/").get_data(as_text=True)
    assert "Работает" in page
    assert "Параметры генерации" not in page and 'name="vllm-temperature"' not in page
    # и лишних запросов к /models и /sampling не делается
    assert [c for c in vision.calls if c[1] in ("/models", "/sampling")] == []


def test_settings_endpoints_require_staff(app, client, vision):
    as_boss(client); logout(client)
    register(client, "user1")
    for url, data in [("/health/backend/vllm/model", {"model": "qwen2.5-vl"}),
                      ("/health/backend/vllm/sampling", {"vllm-temperature": "0.5"}),
                      ("/health/reset-target", {})]:
        r = client.post(url, data=data, follow_redirects=True)
        assert "Недостаточно прав" in r.get_data(as_text=True), url
    assert vision.posts("/sampling") == []
    with app.app_context():
        assert db.session.scalar(db.select(db.func.count(Setting.key))) == 0


def test_admin_role_can_configure(app, client, vision):
    as_boss(client); logout(client)
    register(client, "adm")
    promote(app, "adm", Role.ADMIN)
    r = client.post("/health/backend/vllm/model", data={"model": "qwen2.5-vl"}, follow_redirects=True)
    assert "Анализы будут выполняться на «vllm»" in r.get_data(as_text=True)


def test_unknown_backend_404(client, vision):
    as_boss(client)
    assert client.post("/health/backend/openai/model", data={"model": "x"}).status_code == 404
    assert client.post("/health/backend/openai/sampling", data={}).status_code == 404


# ------------------------------------------------------------------ выбор модели
def test_choose_model_is_used_for_analysis(app, client, vision):
    as_boss(client)
    r = client.post("/health/backend/ollama/model", data={"model": "gemma3:27b"}, follow_redirects=True)
    text = r.get_data(as_text=True)
    assert "Анализы будут выполняться на «ollama», модель: gemma3:27b" in text
    assert "Анализ на: <strong>ollama</strong>" not in text  # это дашборд, тут другая страница

    # дашборд показывает выбор
    assert "ollama" in client.get("/").get_data(as_text=True) and "gemma3:27b" in client.get("/").get_data(as_text=True)

    # анализ уходит с backend и model
    upload(client)
    assert vision.analyze_params[-1] == {"lang": "ru", "backend": "ollama", "model": "gemma3:27b"}

    # выбор виден на странице статуса: карточка ollama помечена активной, выбранная модель selected
    page = client.get("/health/").get_data(as_text=True)
    assert 'value="gemma3:27b" selected' in page


def test_auto_model_sends_only_backend(client, vision):
    as_boss(client)
    client.post("/health/backend/ollama/model", data={"model": ""})
    upload(client)
    assert vision.analyze_params[-1] == {"lang": "ru", "backend": "ollama"}


def test_default_analysis_has_no_backend_or_model(client, vision):
    as_boss(client)
    upload(client)
    assert vision.analyze_params[-1] == {"lang": "ru"}


def test_reset_target(client, vision):
    as_boss(client)
    client.post("/health/backend/ollama/model", data={"model": "gemma3:27b"})
    r = client.post("/health/reset-target", follow_redirects=True)
    assert "Выбор сброшен" in r.get_data(as_text=True)
    upload(client)
    assert vision.analyze_params[-1] == {"lang": "ru"}


def test_unknown_model_rejected(app, client, vision):
    as_boss(client)
    r = client.post("/health/backend/vllm/model", data={"model": "чужая-модель"}, follow_redirects=True)
    assert "не найдена у бэкенда «vllm»" in r.get_data(as_text=True)
    # модель другого бэкенда для vllm тоже не подходит
    r = client.post("/health/backend/vllm/model", data={"model": "gemma3:27b"}, follow_redirects=True)
    assert "не найдена" in r.get_data(as_text=True)
    with app.app_context():
        assert db.session.scalar(db.select(db.func.count(Setting.key))) == 0


def test_unavailable_backend_cannot_be_selected(client, vision):
    vision.ollama_ok = False
    as_boss(client)
    r = client.post("/health/backend/ollama/model", data={"model": "gemma3:27b"}, follow_redirects=True)
    assert "сейчас недоступен" in r.get_data(as_text=True)
    upload(client)
    assert vision.analyze_params[-1] == {"lang": "ru"}


def test_model_list_unavailable_falls_back_to_text_input(client, vision):
    vision.models_shape = "error"
    as_boss(client)
    page = client.get("/health/").get_data(as_text=True)
    assert "Не удалось получить список моделей: connection refused" in page
    assert 'type="text" name="model"' in page
    # вручную введённое имя принимается (проверить его нельзя — список недоступен)
    r = client.post("/health/backend/vllm/model", data={"model": "my-model"}, follow_redirects=True)
    assert "модель: my-model" in r.get_data(as_text=True)


def test_saved_model_missing_from_list_stays_visible(client, vision):
    as_boss(client)
    client.post("/health/backend/vllm/model", data={"model": "qwen2.5-vl"})
    vision.models["vllm"] = ["google/gemma-4-31B-it"]  # модель пропала с сервера
    page = client.get("/health/").get_data(as_text=True)
    assert 'value="qwen2.5-vl" selected' in page


def test_too_long_model_name_rejected(client, vision):
    as_boss(client)
    r = client.post("/health/backend/vllm/model", data={"model": "x" * 300}, follow_redirects=True)
    assert "Слишком длинное" in r.get_data(as_text=True)


# ------------------------------------------------------------------ параметры генерации
def test_save_sampling_sends_only_filled_fields(client, vision):
    as_boss(client)
    r = client.post("/health/backend/vllm/sampling",
                    data={"vllm-temperature": "0.5", "vllm-top_p": "", "vllm-top_k": "", "vllm-seed": "42"},
                    follow_redirects=True)
    assert "Параметры генерации обновлены на сервере: temperature=0.5, seed=42." in r.get_data(as_text=True)
    (_, _, _, sent), = vision.posts("/sampling")
    assert sent == {"temperature": 0.5, "seed": 42}
    assert isinstance(sent["temperature"], float) and isinstance(sent["seed"], int)
    assert vision.sampling["temperature"] == 0.5 and vision.sampling["top_p"] == 0.9  # остальное не тронуто


def test_ollama_form_can_set_num_ctx(client, vision):
    as_boss(client)
    client.post("/health/backend/ollama/sampling", data={"ollama-num_ctx": "16384", "ollama-top_k": "20"})
    assert vision.posts("/sampling")[0][3] == {"top_k": 20, "num_ctx": 16384}


def test_vllm_cannot_set_num_ctx(client, vision):
    as_boss(client)
    r = client.post("/health/backend/vllm/sampling",
                    data={"vllm-num_ctx": "99999", "vllm-temperature": "0.1"}, follow_redirects=True)
    assert vision.posts("/sampling")[0][3] == {"temperature": 0.1}
    assert vision.sampling["num_ctx"] == 8192


def test_decimal_comma_is_accepted(client, vision):
    as_boss(client)
    client.post("/health/backend/vllm/sampling", data={"vllm-temperature": "0,7"})
    assert vision.posts("/sampling")[0][3] == {"temperature": 0.7}


def test_sampling_validation_errors_rerender_page(client, vision):
    as_boss(client)
    bad = {
        "vllm-temperature": "5",      # > 2
        "vllm-top_p": "0",            # должно быть > 0
        "vllm-top_k": "1.5",          # не целое
        "vllm-seed": "abc",
    }
    r = client.post("/health/backend/vllm/sampling", data=bad)
    text = r.get_data(as_text=True)
    assert r.status_code == 400
    assert "Параметры не сохранены" in text
    assert "Допустимое значение: от 0 до 2." in text
    assert "Допустимое значение: больше 0 до 1." in text
    assert text.count("Введите целое число.") == 2
    # введённые значения остались в форме, на сервер ничего не ушло
    assert 'value="5"' in text and 'value="abc"' in text
    assert vision.posts("/sampling") == []


def test_more_invalid_values(client, vision):
    as_boss(client)
    for name, value in [("vllm-temperature", "nan"), ("vllm-temperature", "inf"), ("vllm-temperature", "-0.1"),
                        ("vllm-top_p", "1.01"), ("vllm-top_k", "-2"), ("vllm-top_k", "999999999")]:
        r = client.post("/health/backend/vllm/sampling", data={name: value})
        assert r.status_code == 400, (name, value)
    for name, value in [("ollama-num_ctx", "64"), ("ollama-num_ctx", "2000000"), ("ollama-num_ctx", "8k")]:
        assert client.post("/health/backend/ollama/sampling", data={name: value}).status_code == 400, value
    assert vision.posts("/sampling") == []
    # граничные значения проходят
    for name, value in [("vllm-temperature", "0"), ("vllm-temperature", "2"), ("vllm-top_p", "1"), ("vllm-top_k", "-1")]:
        assert client.post("/health/backend/vllm/sampling", data={name: value}).status_code == 302, (name, value)


def test_empty_form_saves_nothing(client, vision):
    as_boss(client)
    r = client.post("/health/backend/vllm/sampling", data={"vllm-temperature": " "}, follow_redirects=True)
    assert "Нечего сохранять" in r.get_data(as_text=True)
    assert vision.posts("/sampling") == []


def test_server_error_on_sampling_is_reported(client, vision):
    as_boss(client)
    vision.sampling_fail = True
    r = client.post("/health/backend/vllm/sampling", data={"vllm-temperature": "0.3"}, follow_redirects=True)
    text = r.get_data(as_text=True)
    assert "Сервер вернул ошибку (400): запись запрещена" in text
    assert "Не удалось получить параметры генерации: Сервер вернул ошибку (500)" in text  # и GET упал
    assert "Параметры генерации" in text  # страница при этом работает


# ------------------------------------------------------------------ разбор ответов сервера
def test_models_shapes(vision):
    from vision_app import create_app

    app = create_app({"TESTING": True, "SQLALCHEMY_DATABASE_URI": "sqlite://", "VISION_API_BASE_URL": "http://v.test:6769"})
    with app.app_context():
        for shape in ("plain", "openai", "ollama"):
            vision.models_shape = shape
            assert get_models("vllm") == vision.models["vllm"], shape
        vision.models_shape = "error"
        import pytest
        from vision_app.services import VisionApiError
        with pytest.raises(VisionApiError, match="connection refused"):
            get_models("vllm")


def test_sampling_shapes():
    from vision_app.services import _pick_sampling

    flat = {"temperature": 0.1, "top_p": 1, "junk": 1}
    assert _pick_sampling(flat) == {"temperature": 0.1, "top_p": 1}
    assert _pick_sampling({"sampling": flat}) == {"temperature": 0.1, "top_p": 1}
    assert _pick_sampling({"current": {"seed": 5}}) == {"seed": 5}
    assert _pick_sampling([1, 2]) == {} and _pick_sampling(None) == {}


def test_extract_names_dedup_and_junk():
    assert extract_model_names({"models": ["a", "a", " b ", "", None]}, "vllm") == ["a", "b"]
    assert extract_model_names("solo") == ["solo"]
