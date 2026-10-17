"""Redact credentials from text before it is stored in the palace.

Claude Code transcripts from IT and MSP work routinely contain passwords,
API keys, tokens and connection strings. MemPalace stores text verbatim, so
without this layer those secrets would sit in the palace and come back in
search results. This module replaces each detected secret with a marker such
as ``[REDACTED:github-token]`` and leaves every other character untouched.

It is ON by default in this fork. Turn it off with
``MEMPALACE_SCRUB_SECRETS=0`` or ``"scrub_secrets": false`` in config.json.

Design choices:

* **Precision over recall for prose.** Patterns target formats that are
  unambiguously secrets (vendor key prefixes, PEM blocks, ``password=``
  assignments, credentials inside URLs). Ordinary words are never touched.
* **Identifiers are not secrets.** Tenant IDs, subscription IDs, object IDs
  and other GUIDs are kept: they are what makes an MSP note useful, and they
  grant nothing on their own.
* **Pure and fast.** No I/O; a few precompiled regexes per document.

Only text written AFTER this is enabled is scrubbed. Existing drawers and the
original source files on disk are not modified.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Iterable, Optional

logger = logging.getLogger("mempalace_mcp")

_TRUTHY = ("1", "true", "yes", "on")
_FALSY = ("0", "false", "no", "off")

# (kind, compiled pattern, group to redact or None for the whole match)
_RULES: list = []


def _rule(kind: str, pattern: str, group: Optional[int] = None, flags: int = 0) -> None:
    _RULES.append((kind, re.compile(pattern, flags), group))


# ── whole-block secrets ──────────────────────────────────────────────────────
_rule(
    "private-key",
    r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----[\s\S]+?-----END (?:[A-Z0-9]+ )*PRIVATE KEY-----",
)

# ── vendor-prefixed tokens (unambiguous formats) ─────────────────────────────
_rule("anthropic-key", r"\bsk-ant-[A-Za-z0-9_\-]{20,}")
_rule("openai-key", r"\bsk-(?:proj-|svcacct-)?[A-Za-z0-9_\-]{20,}")
_rule("github-token", r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{30,}")
_rule("github-token", r"\bgithub_pat_[A-Za-z0-9_]{40,}")
_rule("slack-token", r"\bxox[abprs]-[A-Za-z0-9\-]{10,}")
_rule("aws-access-key", r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")
_rule("google-api-key", r"\bAIza[0-9A-Za-z_\-]{35}")
_rule("stripe-key", r"\b(?:sk|rk)_live_[0-9A-Za-z]{20,}")
# Entra ID (Azure AD) client secrets: three chars, a digit, "Q~", then 31-34.
_rule("entra-client-secret", r"(?<![A-Za-z0-9_~.\-])[A-Za-z0-9_~.\-]{3}\dQ~[A-Za-z0-9_~.\-]{31,34}")
_rule("jwt", r"\beyJ[A-Za-z0-9_\-]{8,}\.eyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}")

# ── secrets inside larger structures ─────────────────────────────────────────
# scheme://user:password@host  -> keep user and host, redact the password
_rule("url-password", r"\b[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s:/@]+:([^\s@/]{3,})@", group=1)
# Azure storage / Service Bus / Event Hub connection strings
_rule("azure-key", r"(?i)\b(?:AccountKey|SharedAccessKey)=([A-Za-z0-9+/=]{20,})", group=1)
_rule("sas-signature", r"(?i)[?&]sig=([A-Za-z0-9%+/=]{20,})", group=1)
# Authorization headers
_rule("bearer-token", r"(?i)\bBearer\s+([A-Za-z0-9_\-.~+/]{20,}=*)", group=1)
_rule("basic-auth", r"(?i)\bBasic\s+([A-Za-z0-9+/]{16,}={0,2})", group=1)
# PowerShell: ConvertTo-SecureString "secret" -AsPlainText
_rule(
    "powershell-secret",
    r"(?i)ConvertTo-SecureString\s+(?:-String\s+)?([\"'])(.+?)\1",
    group=2,
)

# ── key = value assignments (config files, scripts, env, conn strings) ───────
_SECRET_NAMES = (
    r"password|passwd|pwd|passphrase|secret|client[_\-]?secret|api[_\-]?key|apikey|"
    r"access[_\-]?key|secret[_\-]?key|private[_\-]?key|auth[_\-]?token|access[_\-]?token|"
    r"refresh[_\-]?token|token"
)
# quoted value: password = "hunter2!", $Password = 'x', "api_key": "abc"
_rule(
    "assigned-secret",
    r"(?i)(?<![A-Za-z0-9])\$?[\"']?(?:[A-Za-z0-9_]*[_\-])?(?:"
    + _SECRET_NAMES
    + r")[\"']?\s*[:=]\s*"
    r"([\"'])([^\"'\r\n]{4,})\1",
    group=2,
)
# unquoted value up to whitespace or ';' (env files, connection strings)
_rule(
    "assigned-secret",
    r"(?i)(?<![A-Za-z0-9])(?:[A-Za-z0-9_]*[_\-])?(?:" + _SECRET_NAMES + r")\s*[:=]\s*"
    r"(?![\"'\[$({<])([^\s;\"',(\[]{6,})(?![(\[\w.!@#%^&*+/=~-])",
    group=1,
)

# Values that look like an assignment but are not secrets.
_PLACEHOLDER_RE = re.compile(
    r"(?i)^(?:\*+|x+|\.{3,}|<[^>]*>|\$\{?[A-Za-z_][A-Za-z0-9_]*\}?|%[A-Za-z_]+%|"
    r"none|null|true|false|changeme|redacted|example|placeholder|your[_\-].*|"
    r"\[redacted:[a-z\-]+\]|"
    # dotted references such as config.password or os.environ are code, not values
    r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)$"
)


def enabled(file_config: Optional[dict] = None) -> bool:
    raw = os.environ.get("MEMPALACE_SCRUB_SECRETS", "").strip().lower()
    if raw in _TRUTHY:
        return True
    if raw in _FALSY:
        return False
    if file_config is None:
        try:
            from .config import MempalaceConfig

            file_config = getattr(MempalaceConfig(), "_file_config", None) or {}
        except Exception:
            file_config = {}
    value = file_config.get("scrub_secrets", True)
    if isinstance(value, str):
        return value.strip().lower() not in _FALSY
    return bool(value)


def scrub(text: str) -> tuple:
    """Return ``(clean_text, counts)``. ``counts`` maps kind -> redactions."""
    if not isinstance(text, str) or not text:
        return text, {}
    counts: dict = {}
    for kind, pattern, group in _RULES:

        def _replace(match, kind=kind, group=group):
            if group is None:
                counts[kind] = counts.get(kind, 0) + 1
                return f"[REDACTED:{kind}]"
            value = match.group(group)
            if not value or _PLACEHOLDER_RE.match(value) or value.startswith("[REDACTED:"):
                return match.group(0)
            counts[kind] = counts.get(kind, 0) + 1
            start, end = match.span(group)
            offset = match.start(0)
            whole = match.group(0)
            return whole[: start - offset] + f"[REDACTED:{kind}]" + whole[end - offset :]

        text = pattern.sub(_replace, text)
    return text, counts


def scrub_documents(documents: Optional[Iterable]) -> tuple:
    """Scrub a batch. Returns ``(documents, total_redactions)``; ``None`` passes through."""
    if documents is None:
        return None, 0
    single = isinstance(documents, str)
    docs = [documents] if single else list(documents)
    total = 0
    cleaned = []
    for doc in docs:
        new, counts = scrub(doc) if isinstance(doc, str) else (doc, {})
        total += sum(counts.values())
        cleaned.append(new)
    return (cleaned[0] if single else cleaned), total


def install(collection):
    """Make every write through ``collection`` scrub its documents first.

    Wraps the instance's ``add``, ``upsert`` and ``update`` in place, so the
    object keeps its type and every other attribute. Idempotent.
    """
    if collection is None or getattr(collection, "_mempalace_scrubbing", False):
        return collection
    for name in ("add", "upsert", "update"):
        original = getattr(collection, name, None)
        if original is None:
            continue

        def wrapper(*args, _original=original, _name=name, **kwargs):
            if "documents" in kwargs and kwargs["documents"] is not None:
                kwargs["documents"], redacted = scrub_documents(kwargs["documents"])
                if redacted:
                    logger.info("secret_scrubber: redacted %d secret(s) on %s", redacted, _name)
            return _original(*args, **kwargs)

        setattr(collection, name, wrapper)
    collection._mempalace_scrubbing = True
    return collection
