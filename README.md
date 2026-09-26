# Vision Analyzer

Локальный риск-триаж изображений: vision-модель (vLLM или Ollama) смотрит на
картинку и возвращает структурированный отчёт — уровень риска, сигналы,
рекомендацию. В репозитории два компонента:

- **`inference/`** — HTTP-сервер анализа (aiohttp). Принимает изображение,
  обращается к vLLM или Ollama, возвращает JSON-отчёт. Работает сам по себе,
  через `curl` или из любого клиента.
- **`vision_app/`** — веб-панель на Flask поверх этого сервера: аккаунты и
  роли, очередь загрузок, история, статус бэкендов, выбор модели и
  параметров генерации, админка.

Панель — не обязательный слой: `inference/server.py` можно использовать
отдельно, без Flask вообще.

## Быстрый старт

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

export FLASK_SECRET_KEY="длинная-случайная-строка"

python run.py
```

`run.py` поднимает Flask-панель и рядом, отдельным подпроцессом, запускает
`inference/server.py` (у него свой event loop на aiohttp, поэтому он живёт не
в том же процессе). Панель — на `http://127.0.0.1:6967`, сервер анализа — на
`http://127.0.0.1:6769`. Первый зарегистрированный в панели пользователь
становится главным администратором.

Если нужен только сервер анализа, без панели:

```bash
cd inference && python server.py
```

Продакшен: `gunicorn wsgi:app` для панели; `inference/server.py` запускается
и следится отдельно (systemd/supervisor/Docker) — `wsgi.py` его не поднимает.

## `inference/`: сервер анализа

| Файл | Назначение |
|---|---|
| `server.py` | HTTP-эндпоинты (aiohttp): `/`, `/health`, `/analyze`, `/chat`, `/lang`, `/config`, `/sampling`, `/models`, `/categories` |
| `backends.py` | Обращение к vLLM (`/v1/chat/completions`) и Ollama (`/api/chat`) для `/analyze` |
| `chat.py`, `chat_backends.py` | То же самое, но для `/chat` — свободный диалог с историей, без JSON-схемы риск-отчёта |
| `config.py` | Константы, параметры сэмплинга, логирование, бутстрап `locales`/`prompt`/`image_upscaler` |
| `prompt.py` | Системный промпт `/analyze` (двухпроходный, по категориям) и дефолтная системная "личность" `/chat` |
| `locales.py`, `locales/*.json` | Тексты ответов сервера на разных языках |
| `image_upscaler.py` | Апскейл маленьких изображений (Real-ESRGAN) перед анализом |

### Эндпоинты

- `GET /` — список команд и примеров (то же, что ниже, простым текстом).
- `GET /health` — статус vLLM и Ollama: доступность и автоопределённая модель.
- `POST /analyze` — анализ одного или нескольких изображений. Тело запроса —
  любое из трёх: сырые байты картинки (`Content-Type: image/*`),
  `multipart/form-data` (`images`, plus `backend`/`model`/`lang`/`history`)
  или JSON (`image`/`images`, plus те же необязательные поля). Параметры
  можно передать и через query (`?backend=&model=&lang=`).
- `POST /chat` — свободный диалог с моделью (текст + необязательные
  картинки, с историей), без JSON-схемы риск-отчёта — ответ отдаётся как
  есть. Тело — JSON (`message`, `image`/`images`, `system`, `history`,
  `backend`/`model`/`lang`) либо `multipart/form-data` с теми же полями.
  Сервер сам историю не хранит — она целиком приходит от клиента на
  каждый запрос. Перед сообщениями всегда стоит дефолтный системный
  промпт: модель знает, что она ассистент ПО инструменту, а не сам
  риск-анализатор, и не выносит вердиктов по риск-сигналам вместо
  `/analyze`; переданный `system` добавляется к этому промпту, а не
  заменяет его. Если выбранный бэкенд недоступен, а `backend` не был
  передан явно, сервер один раз автоматически пробует второй бэкенд.
- `POST /lang` — сменить язык ответов по умолчанию (`ru`/`en`).
- `GET/POST /config` — посмотреть/поменять `backend`, `ollama_host`,
  `vllm_url` без перезапуска.
- `GET/POST /sampling` — посмотреть/поменять `temperature`, `top_p`,
  `top_k`, `seed`, `num_ctx`, `num_predict`, `think` (`true`/`false` или
  `low`/`medium`/`high`). Общие для всего сервера и для обоих бэкендов.
  `num_ctx` действует только для Ollama.
- `GET /models` — полный список моделей, которые прямо сейчас отдаёт бэкенд
  (`?backend=vllm|ollama`, без параметра — оба).

Примеры curl — в тексте `GET /`.

### Переменные окружения (`inference/`)

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `VISION_ANALYZER_BACKEND` | `vllm` | Бэкенд по умолчанию (`vllm` / `ollama`) |
| `VISION_ANALYZER_OLLAMA_HOST` | `http://127.0.0.1:11434` | Адрес Ollama |
| `VISION_ANALYZER_VLLM_URL` | `http://host.docker.internal:8000/v1` | Адрес vLLM (OpenAI-совместимый) |
| `VISION_ANALYZER_TEMPERATURE` / `_TOP_P` / `_TOP_K` / `_SEED` | `0` / `1.0` / `1` / `42` | Параметры сэмплинга по умолчанию |
| `VISION_ANALYZER_NUM_CTX` | — (не переопределяется) | Размер контекста (только Ollama) |
| `VISION_ANALYZER_NUM_PREDICT` | — (не переопределяется) | Лимит длины ответа |
| `VISION_ANALYZER_THINK` | `true` | Режим рассуждений модели |
| `VISION_LOG_DIR` | `<корень проекта>/logs` | Куда писать `analyzer.log` / `analyzer.error.log` |

Сервер слушает `0.0.0.0:6769`.

## `vision_app/`: веб-панель

| Модуль | Назначение |
|---|---|
| `models.py` | `User` (роли), `AnalysisResult` (статус, отчёт), `Setting` |
| `blueprints/accounts.py` | Регистрация, вход/выход, профиль, блокировка |
| `blueprints/analyzer.py` | Загрузка → очередь, история, детальный результат, статус сервера |
| `blueprints/panel.py` | Админка: пользователи, роли, все анализы, статистика |
| `queue_worker.py` | Фоновый поток, обрабатывающий очередь анализов по одному |
| `services.py` | Клиент к `inference/server.py` (`/analyze`, `/health`, `/models`, `/sampling`) |
| `settings_store.py` | Выбранные бэкенд/модель для анализа (хранятся в БД) |
| `history.py` | Удаление записей истории вместе с файлами |
| `schema.py` | Автодобавление новых колонок в существующую БД при старте |

### Возможности панели

- Роли: `user`, `admin`, `head_admin` (+ `blocked`). Первый зарегистрированный
  становится `head_admin`.
- Загрузка нескольких изображений за раз — они встают в очередь и
  обрабатываются по одному фоновым потоком; окно очереди на дашборде
  обновляется само, без перезагрузки страницы.
- История, детальный отчёт по каждому анализу, отмена ещё не начатых задач.
- Статус `/health` обоих бэкендов; для админов — выбор модели (из
  `GET /models`) и параметров генерации (`POST /sampling`) прямо из панели.
- Массовое и точечное удаление истории (записи + файлы) для `admin`/`head_admin`.

### Переменные окружения (`vision_app/`)

| Переменная | По умолчанию | Назначение |
|---|---|---|
| `FLASK_SECRET_KEY` | `dev-insecure-change-me` | Секретный ключ (сессии, CSRF) — обязательно задайте в проде |
| `DATABASE_URL` | `sqlite:///db.sqlite3` | Строка подключения SQLAlchemy |
| `UPLOAD_FOLDER` | `./media` | Куда сохраняются загруженные изображения |
| `MAX_CONTENT_LENGTH` | 50 МБ (в коде) | Максимальный размер запроса |
| `VISION_API_BASE_URL` | `http://127.0.0.1:6769` | Адрес `inference/server.py` |
| `VISION_API_TIMEOUT` | `500` | Таймаут запроса к `/analyze`, секунд |
| `APP_TIMEZONE` | `Asia/Aqtobe` | Часовой пояс отображения дат |
| `AUTO_CREATE_DB` | `1` | Создавать/обновлять таблицы при старте |
| `SESSION_COOKIE_SECURE` | `0` | `1` — cookie только по HTTPS |
| `QUEUE_WORKER_ENABLED` | `1` | Выключить фоновую обработку очереди |
| `QUEUE_POLL_SECONDS` | `5` | Как часто обработчик проверяет очередь, когда она пуста |
| `QUEUE_MAX_FILES_PER_UPLOAD` | `20` | Лимит файлов за одну загрузку |
| `VISION_ANALYZER_DIR` | `<корень проекта>/inference` | Где искать `server.py` для авто-запуска из `run.py` |
| `VISION_ANALYZER_PYTHON` | venv рядом с `inference/`, иначе текущий интерпретатор | Каким python запускать `inference/server.py` |
| `FLASK_RUN_HOST` / `FLASK_RUN_PORT` / `FLASK_DEBUG` | `127.0.0.1` / `6967` / `0` | Параметры `python run.py` |

### Команды

```bash
flask --app vision_app init-db                    # создать таблицы вручную
flask --app vision_app set-role <логин> <роль>    # blocked | user | admin | head_admin
```

## Требования

- Python 3.11+
- vLLM (OpenAI-совместимый эндпоинт) и/или локально запущенный Ollama с
  vision-моделью
- `requirements.txt` — общий для `inference/` и `vision_app/` (aiohttp/torch/
  Pillow для сервера анализа, Flask-стек для панели)

## Структура репозитория

```
inference/          сервер анализа (aiohttp)
vision_app/          Flask-панель
run.py                запуск панели + сервера анализа рядом (разработка)
wsgi.py                точка входа для gunicorn (только панель)
requirements.txt
```