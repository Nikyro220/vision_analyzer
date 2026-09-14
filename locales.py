import os, json

LOCALE_FILE_NAME = "locales.json"
LOCALE_LOCATION = None
LOCALES = {}


def get_locales_path(name, directory = None):
	return os.path.join(LOCALE_LOCATION, LOCALE_FILE_NAME) if LOCALE_LOCATION else LOCALE_FILE_NAME

def load_locales(locales, path):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            locales.update(json.load(f))		

def format(s, **kwargs):
    return s.format(**kwargs)

def get_locale(k):
	return LOCALES.get(k, "404")

def get_formatted(k, **kwargs):
	return format(get_locale(k), **kwargs)



load_locales(LOCALES, get_locales_path(LOCALE_FILE_NAME, LOCALE_LOCATION))

print(get_formatted("test", test="testttt"))
