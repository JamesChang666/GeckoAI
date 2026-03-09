from typing import Dict


def looks_garbled_text(text: str) -> bool:
    if not text:
        return False
    if "??" in text:
        return True
    return text.count("?") >= 2


def normalize_lang_map(lang_map: Dict[str, Dict[str, str]]) -> None:
    en_map = lang_map.get("en", {})
    zh_map = lang_map.setdefault("zh", {})
    for key, en_val in en_map.items():
        zh_val = str(zh_map.get(key, ""))
        if key not in zh_map or looks_garbled_text(zh_val):
            zh_map[key] = en_val
