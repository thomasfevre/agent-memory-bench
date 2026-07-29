from collections.abc import Iterable

from .contracts import Evidence


def merge_evidence(items: Iterable[Evidence | str]) -> Evidence:
    texts = []
    source_ids = []
    for item in items:
        if isinstance(item, str):
            texts.append(item)
        else:
            texts.append(item.text)
            source_ids.extend(item.source_ids)
    return Evidence(text="\n".join(texts), source_ids=tuple(source_ids))
