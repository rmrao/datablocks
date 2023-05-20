import dataclasses
from typing import Any


def fields_dict(cls, init: bool = True, repr: bool = True) -> dict[str, Any]:
    fields = dataclasses.fields(cls)
    vars_: dict[str, Any] = {}
    for field in fields:
        if not init and not field.init:
            continue
        if not repr and not field.repr:
            continue
        vars_[field.name] = getattr(cls, field.name)
    return vars_
