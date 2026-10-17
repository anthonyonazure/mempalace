"""Tests for per-client palace isolation."""

import json
import os

import pytest

from mempalace import client_walls as cw
from mempalace.config import MempalaceConfig


@pytest.fixture
def layout(tmp_path, monkeypatch):
    for name in (
        "MEMPALACE_PALACE_PATH",
        "MEMPALACE_PALACE",
        "MEMPAL_PALACE_PATH",
        "MEMPALACE_CLIENT",
    ):
        monkeypatch.delenv(name, raising=False)
    acme = tmp_path / "Clients" / "Acme"
    beltway = tmp_path / "Clients" / "Beltway"
    nested = beltway / "subcontract" / "Nested"
    internal = tmp_path / "Internal"
    for d in (acme / "src", beltway, nested, internal):
        d.mkdir(parents=True)
    root = tmp_path / "palaces"
    config_dir = tmp_path / "cfg"
    config_dir.mkdir()
    (config_dir / "config.json").write_text(
        json.dumps(
            {
                "client_walls": {
                    "root": str(root),
                    "clients": {
                        "acme": [str(acme)],
                        "beltway": str(beltway),
                        "nested-co": [str(nested)],
                    },
                }
            }
        )
    )
    return {
        "acme": acme,
        "beltway": beltway,
        "nested": nested,
        "internal": internal,
        "root": root,
        "config_dir": config_dir,
    }


def _palace(layout, cwd, monkeypatch, **kw):
    monkeypatch.chdir(cwd)
    return MempalaceConfig(config_dir=str(layout["config_dir"]), **kw).palace_path


def _real(p):
    return os.path.realpath(str(p))


def test_client_folder_gets_its_own_palace(layout, monkeypatch):
    got = _palace(layout, layout["acme"] / "src", monkeypatch)
    assert got == _real(layout["root"] / "acme" / "palace")


def test_two_clients_never_share(layout, monkeypatch):
    a = _palace(layout, layout["acme"], monkeypatch)
    b = _palace(layout, layout["beltway"], monkeypatch)
    assert a != b
    assert not a.startswith(os.path.dirname(b))


def test_deepest_client_folder_wins(layout, monkeypatch):
    got = _palace(layout, layout["nested"], monkeypatch)
    assert got == _real(layout["root"] / "nested-co" / "palace")


def test_non_client_folder_uses_default_palace(layout, monkeypatch):
    got = _palace(layout, layout["internal"], monkeypatch)
    assert got == os.path.join(str(layout["config_dir"]), "palace")


def test_pinning_by_env(layout, monkeypatch):
    monkeypatch.setenv("MEMPALACE_CLIENT", "beltway")
    got = _palace(layout, layout["internal"], monkeypatch)
    assert got == _real(layout["root"] / "beltway" / "palace")


def test_unknown_pinned_client_is_an_error(layout, monkeypatch):
    monkeypatch.setenv("MEMPALACE_CLIENT", "nobody")
    with pytest.raises(cw.ClientWallError):
        _palace(layout, layout["internal"], monkeypatch)


def test_strict_refuses_another_clients_palace(layout, monkeypatch):
    other = _real(layout["root"] / "beltway" / "palace")
    monkeypatch.setenv("MEMPALACE_PALACE_PATH", other)
    with pytest.raises(cw.ClientWallError):
        _palace(layout, layout["acme"], monkeypatch)


def test_strict_refuses_override_too(layout, monkeypatch):
    other = _real(layout["root"] / "beltway" / "palace")
    with pytest.raises(cw.ClientWallError):
        _palace(layout, layout["acme"], monkeypatch, palace_path=other)


def test_explicit_own_palace_is_allowed(layout, monkeypatch):
    own = _real(layout["root"] / "acme" / "palace")
    monkeypatch.setenv("MEMPALACE_PALACE_PATH", own)
    assert _palace(layout, layout["acme"], monkeypatch) == own


def test_non_strict_allows_crossing(layout, monkeypatch):
    cfg_file = layout["config_dir"] / "config.json"
    data = json.loads(cfg_file.read_text())
    data["client_walls"]["strict"] = False
    cfg_file.write_text(json.dumps(data))
    other = _real(layout["root"] / "beltway" / "palace")
    monkeypatch.setenv("MEMPALACE_PALACE_PATH", other)
    assert _palace(layout, layout["acme"], monkeypatch) == other


def test_side_files_follow_the_client_palace(layout, monkeypatch):
    monkeypatch.chdir(layout["acme"])
    cfg = MempalaceConfig(config_dir=str(layout["config_dir"]))
    client_dir = _real(layout["root"] / "acme")
    assert cfg.tunnel_file.startswith(client_dir)
    assert cfg.hallway_file.startswith(client_dir)


def test_unconfigured_changes_nothing(tmp_path, monkeypatch):
    for name in (
        "MEMPALACE_PALACE_PATH",
        "MEMPALACE_PALACE",
        "MEMPAL_PALACE_PATH",
        "MEMPALACE_CLIENT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)
    cfg = MempalaceConfig(config_dir=str(tmp_path))
    assert cfg.palace_path == os.path.join(str(tmp_path), "palace")


def test_bad_client_name_rejected():
    with pytest.raises(ValueError):
        cw.settings({"client_walls": {"clients": {"Acme Law!": ["/x"]}}})


def test_search_cannot_see_the_other_client(layout, monkeypatch):
    from mempalace.palace import get_collection
    from mempalace.searcher import search_memories

    for client, text in (
        ("acme", "Acme uses Intune for laptops"),
        ("beltway", "Beltway needs CMMC level 2"),
    ):
        monkeypatch.chdir(layout[client])
        path = MempalaceConfig(config_dir=str(layout["config_dir"])).palace_path
        col = get_collection(path, create=True)
        col.add(
            ids=[f"{client}_1"],
            documents=[text],
            metadatas=[{"wing": client, "room": "notes", "source_file": "n.md"}],
        )

    monkeypatch.chdir(layout["acme"])
    acme_path = MempalaceConfig(config_dir=str(layout["config_dir"])).palace_path
    result = search_memories("CMMC level 2", acme_path, n_results=5)
    assert all("Beltway" not in h["text"] for h in result["results"])
