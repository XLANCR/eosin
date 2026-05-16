from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = ["BankParserService", "app", "create_app"]


def __getattr__(name: str) -> Any:
    if name == "BankParserService":
        return import_module("eosin.backend.bank_parser_service").BankParserService
    if name in {"app", "create_app"}:
        module = import_module("eosin.backend.bank_parser_api")
        return getattr(module, name)
    raise AttributeError(name)
