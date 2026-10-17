"""Client walls: one physically separate palace per client.

An MSP's notes for Client A must never come back in a search done while
working for Client B. Wings inside a single palace are not a wall: an
unfiltered search spans every wing, and tunnels/hallways deliberately link
across them. So each client gets its own palace directory, which means its
own vector index, knowledge graph, tunnels and hallways. Nothing is shared
but the program.

Configure in ``config.json``::

    "client_walls": {
        "root": "~/.mempalace/clients",
        "clients": {
            "acme-law": ["~/Clients/AcmeLaw"],
            "beltway-defense": ["~/Clients/Beltway", "~/Projects/beltway-cmmc"]
        },
        "strict": true
    }

How the palace is chosen (first match wins):

1. An explicit palace (``--palace`` / ``MEMPALACE_PALACE_PATH``) as before.
2. ``MEMPALACE_CLIENT=<name>`` pins a client's palace, for an MCP server or
   hook whose working folder is not inside the client's folders.
3. The current working folder: inside one of a client's folders means that
   client's palace at ``<root>/<name>/palace``.
4. Otherwise the normal default palace (internal and personal work).

``strict`` (default true): when the working folder belongs to one client,
an explicit palace that points into a DIFFERENT client's palace is refused
with :class:`ClientWallError` instead of silently crossing the wall.

Not walled (shared, by design, in the config folder): the global entity
registry and people map, and Claude Code's own transcript files on disk.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Optional

_NAME_RE = re.compile(r"^[a-z0-9][a-z0-9._\-]{0,63}$")
DEFAULT_ROOT = "~/.mempalace/clients"


class ClientWallError(RuntimeError):
    """An operation would read or write another client's palace."""


def _expand(path: str) -> str:
    return os.path.realpath(os.path.abspath(os.path.expanduser(str(path))))


def settings(file_config: Optional[dict]) -> Optional[dict]:
    """Normalized ``client_walls`` settings, or None when not configured."""
    raw = (file_config or {}).get("client_walls")
    if not isinstance(raw, dict):
        return None
    clients = raw.get("clients")
    if not isinstance(clients, dict) or not clients:
        return None
    normalized: dict = {}
    for name, folders in clients.items():
        key = str(name).strip().lower()
        if not _NAME_RE.match(key):
            raise ValueError(
                f"client_walls: invalid client name {name!r} "
                "(use lowercase letters, digits, '.', '_' or '-')"
            )
        if isinstance(folders, str):
            folders = [folders]
        normalized[key] = [_expand(f) for f in folders or [] if str(f).strip()]
    strict = raw.get("strict", True)
    if isinstance(strict, str):
        strict = strict.strip().lower() not in ("0", "false", "no", "off")
    return {
        "root": _expand(raw.get("root") or DEFAULT_ROOT),
        "clients": normalized,
        "strict": bool(strict),
    }


def client_palace(cfg: dict, name: str) -> str:
    return os.path.join(cfg["root"], name, "palace")


def _is_within(path: str, folder: str) -> bool:
    try:
        return os.path.commonpath([path, folder]) == folder
    except ValueError:  # different drives on Windows
        return False


def client_for_folder(cfg: dict, folder: Optional[str] = None) -> Optional[str]:
    """The client whose folders contain ``folder`` (default: the cwd).

    The deepest matching folder wins, so a client folder nested inside
    another client's folder belongs to the inner client.
    """
    try:
        here = _expand(folder or os.getcwd())
    except OSError:
        return None
    best, best_len = None, -1
    for name, folders in cfg["clients"].items():
        for f in folders:
            if _is_within(here, f) and len(f) > best_len:
                best, best_len = name, len(f)
    return best


def client_for_palace(cfg: dict, palace_path: str) -> Optional[str]:
    """The client whose palace directory contains ``palace_path``, if any."""
    target = _expand(palace_path)
    for name in cfg["clients"]:
        if _is_within(target, _expand(os.path.join(cfg["root"], name))):
            return name
    return None


def resolve(file_config: Optional[dict], explicit: Optional[str]) -> Optional[str]:
    """Return the palace path the walls require, or None to use the default.

    ``explicit`` is a palace chosen by override or environment, if any.
    Raises :class:`ClientWallError` on a strict-mode crossing.
    """
    cfg = settings(file_config)
    if cfg is None:
        return None
    here_client = client_for_folder(cfg)
    pinned = os.environ.get("MEMPALACE_CLIENT", "").strip().lower()
    if pinned and pinned not in cfg["clients"]:
        raise ClientWallError(
            f"MEMPALACE_CLIENT={pinned!r} is not a configured client "
            f"(known: {', '.join(sorted(cfg['clients']))})"
        )
    active = pinned or here_client

    if explicit:
        if cfg["strict"] and active:
            owner = client_for_palace(cfg, explicit)
            if owner and owner != active:
                raise ClientWallError(
                    f"Refusing to open {owner!r}'s palace while working as {active!r}. "
                    "Change folders or MEMPALACE_CLIENT, or set client_walls.strict=false."
                )
        return explicit
    if active:
        return client_palace(cfg, active)
    return None


def describe(file_config: Optional[dict], palace_path: str) -> str:
    """Human-readable summary for ``mempalace clients``."""
    cfg = settings(file_config)
    if cfg is None:
        return (
            'Client walls are not configured. Add a "client_walls" section to '
            "config.json (see mempalace/client_walls.py for the format)."
        )
    here = client_for_folder(cfg)
    pinned = os.environ.get("MEMPALACE_CLIENT", "").strip().lower() or None
    lines = [
        f"Client palaces root: {cfg['root']}",
        f"Strict mode: {'on' if cfg['strict'] else 'off'}",
        "",
        "Clients:",
    ]
    for name in sorted(cfg["clients"]):
        exists = Path(client_palace(cfg, name)).exists()
        lines.append(f"  {name:<24} {'palace exists' if exists else 'no palace yet'}")
        for folder in cfg["clients"][name]:
            lines.append(f"      folder: {folder}")
    lines += [
        "",
        f"This folder belongs to: {here or '(no client)'}",
        f"Pinned by MEMPALACE_CLIENT: {pinned or '(not set)'}",
        f"Active palace: {palace_path}",
    ]
    return "\n".join(lines)
