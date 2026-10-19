"""Чтение лог-файлов из logs/ для страницы «Логи» (панель, только head_admin).

Два источника, они показываются раздельно:
  flask  — Flask-панель:    app.log, app.error.log            (logging_setup.py)
  server — сервер анализа:  analyzer.log, analyzer.error.log  (inference/config.py)

Для каждого источника доступны текущие файлы и ротированные копии вида
<имя>.YYYY-MM-DD. Имя файла берётся только из списка реально существующих файлов
этого источника, поэтому произвольный путь через параметр ?file= прочитать нельзя.
"""

from __future__ import annotations

import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from flask_babel import gettext, lazy_gettext

LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")
_LEVEL_RANK = {name: i for i, name in enumerate(LEVELS)}

LIMIT_CHOICES = (100, 200, 500, 1000)
DEFAULT_LIMIT = 200
TAIL_BYTES = 2 * 1024 * 1024  # читаем с конца файла не больше стольких байт
QUERY_MAX_LEN = 200

_DATE_SUFFIX = re.compile(r"\d{4}-\d{2}-\d{2}")
# «2026-10-14 05:16:01,123 [INFO] logger: message» — формат из logging_setup.py и inference/config.py
_HEAD = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:,\d{3})? \[(\w+)\] ")


@dataclass(frozen=True)
class LogSource:
    key: str
    title: str
    files: tuple[str, ...]  # имена текущих файлов; первый открывается по умолчанию


SOURCES: dict[str, LogSource] = {
    "flask": LogSource("flask", lazy_gettext("logs.source.flask"), ("app.log", "app.error.log")),
    "server": LogSource("server", lazy_gettext("logs.source.server"), ("analyzer.log", "analyzer.error.log")),
}
DEFAULT_SOURCE = "flask"


@dataclass(frozen=True)
class LogFile:
    name: str
    size: int
    mtime: datetime  # UTC
    rotated: bool

    @property
    def size_h(self) -> str:
        if self.size < 1024:
            return gettext("logs.size.b", n=self.size)
        if self.size < 1024 * 1024:
            return gettext("logs.size.kb", n=f"{self.size / 1024:.1f}")
        return gettext("logs.size.mb", n=f"{self.size / 1024 / 1024:.1f}")


@dataclass
class Entry:
    level: str  # "" — строки без заголовка (например, обрезанное начало)
    lines: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "\n".join(self.lines)


@dataclass
class LogView:
    source: LogSource
    sources: dict[str, LogSource]
    files: list[LogFile]
    file: LogFile | None
    entries: list[Entry]  # новые сверху
    matched: int  # сколько записей подошло под фильтр (показано не больше limit)
    truncated: bool  # файл больше TAIL_BYTES — прочитан только хвост
    level: str
    query: str
    limit: int
    levels: tuple[str, ...] = LEVELS
    limit_choices: tuple[int, ...] = LIMIT_CHOICES
    dir_exists: bool = True
    read_kb: int = TAIL_BYTES // 1024


def _belongs(source: LogSource, name: str) -> bool:
    for base in source.files:
        if name == base:
            return True
        if name.startswith(base + ".") and _DATE_SUFFIX.fullmatch(name[len(base) + 1:]):
            return True
    return False


def list_files(log_dir: Path, source: LogSource) -> list[LogFile]:
    """Файлы источника: сначала текущие (в порядке source.files), затем ротированные, новые выше."""
    if not log_dir.is_dir():
        return []
    found: list[LogFile] = []
    for path in log_dir.iterdir():
        if not _belongs(source, path.name):
            continue
        try:
            if not path.is_file():
                continue
            st = path.stat()
        except OSError:
            continue
        found.append(LogFile(
            name=path.name,
            size=st.st_size,
            mtime=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc),
            rotated=path.name not in source.files,
        ))

    current = [f for f in found if not f.rotated]
    rotated = [f for f in found if f.rotated]
    current.sort(key=lambda f: source.files.index(f.name))
    # ротированные: сначала по исходному файлу, внутри — новые даты выше
    rotated.sort(key=lambda f: f.name, reverse=True)
    rotated.sort(key=lambda f: source.files.index(f.name.rpartition(".")[0]))
    return current + rotated


def read_entries(path: Path, max_bytes: int = TAIL_BYTES) -> tuple[list[Entry], bool]:
    """Записи лога из хвоста файла. Многострочные (traceback) остаются одной записью."""
    size = path.stat().st_size
    start = max(0, size - max_bytes)
    with path.open("rb") as fh:
        fh.seek(start)
        data = fh.read()
    lines = data.decode("utf-8", errors="replace").splitlines()
    truncated = start > 0
    if truncated and lines:
        lines = lines[1:]  # первая строка при чтении с середины файла почти наверняка обрезана

    entries: list[Entry] = []
    current: Entry | None = None
    for line in lines:
        m = _HEAD.match(line)
        if m:
            current = Entry(level=m.group(1).upper(), lines=[line])
            entries.append(current)
        elif current is None:
            current = Entry(level="", lines=[line])
            entries.append(current)
        else:
            current.lines.append(line)
    return entries, truncated


def select(entries: list[Entry], *, level: str, query: str, limit: int) -> tuple[list[Entry], int]:
    """Фильтр по минимальному уровню и подстроке; последние `limit` записей, новые сверху."""
    min_rank = _LEVEL_RANK.get(level)
    needle = query.casefold()
    kept: deque[Entry] = deque(maxlen=limit)
    matched = 0
    for entry in entries:
        if min_rank is not None and _LEVEL_RANK.get(entry.level, -1) < min_rank:
            continue
        if needle and needle not in entry.text.casefold():
            continue
        matched += 1
        kept.append(entry)
    return list(reversed(kept)), matched


def build_view(log_dir: str | Path, *, source: str = "", file: str = "", level: str = "",
               query: str = "", limit: str | int = "") -> LogView:
    src = SOURCES.get(source) or SOURCES[DEFAULT_SOURCE]
    level = level.upper() if level.upper() in _LEVEL_RANK else ""
    query = query.strip()[:QUERY_MAX_LEN]
    try:
        limit_n = int(limit)
    except (TypeError, ValueError):
        limit_n = DEFAULT_LIMIT
    if limit_n not in LIMIT_CHOICES:
        limit_n = DEFAULT_LIMIT

    root = Path(log_dir)
    files = list_files(root, src)
    chosen = next((f for f in files if f.name == file), files[0] if files else None)

    entries: list[Entry] = []
    matched = 0
    truncated = False
    if chosen is not None:
        try:
            all_entries, truncated = read_entries(root / chosen.name)
            entries, matched = select(all_entries, level=level, query=query, limit=limit_n)
        except OSError:
            chosen = None  # файл пропал или занят между listdir и чтением

    return LogView(
        source=src, sources=SOURCES, files=files, file=chosen, entries=entries, matched=matched,
        truncated=truncated, level=level, query=query, limit=limit_n, dir_exists=root.is_dir(),
    )
