"""Tests for credential redaction on the write path."""

import pytest

from mempalace import secret_scrubber as ss

REDACT_CASES = [
    ("export GITHUB_TOKEN=ghp_abcdefghijklmnopqrstuvwxyz0123456789", "github-token"),
    ('password = "Hunter2!Secret"', "assigned-secret"),
    ("$Password = 'P@ssw0rd123'", "assigned-secret"),
    ("Server=tcp:sql1;Database=crm;User ID=admin;Password=Sup3rS3cret!;", "assigned-secret"),
    ("AccountName=acme;AccountKey=abcdEFGHijklMNOPqrstUVWXyz0123456789+/abcd==;", "azure-key"),
    ("git clone https://anthony:s3cr3tpass@github.com/acme/repo.git", "url-password"),
    ('$s = ConvertTo-SecureString "Winter2026!" -AsPlainText -Force', "powershell-secret"),
    ("client secret abc8Q~abcdefghijklmnopqrstuvwxyz0123456 here", "entra-client-secret"),
    ("AWS key AKIAIOSFODNN7EXAMPLE in the script", "aws-access-key"),
    ("ANTHROPIC_API_KEY=sk-ant-api03-abcdefghijklmnopqrstuvwxyz", "anthropic-key"),
    ("xoxb-1234567890-abcdefghijkl", "slack-token"),
    (
        "https://acct.blob.core.windows.net/c?sv=2022&sig=abcdEFGHijklMNOPqrstUVWX%2Fyz12",
        "sas-signature",
    ),
    (
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA7\nabc\n-----END RSA PRIVATE KEY-----",
        "private-key",
    ),
]

KEEP_CASES = [
    "Tenant ID 72f988bf-86f1-41af-91ab-2d7cd011db47 uses Intune",
    "password = getpass.getpass()",
    "password = config.password",
    "token_count: 4096 and max_tokens=1024",
    "The password policy requires 14 characters.",
    "password: <your password here>",
    "api_key = os.environ['API_KEY']",
    "secret: true",
    "password=$env:SQL_PASSWORD",
    "We rotated the client secret on Friday.",
]


@pytest.mark.parametrize("text,kind", REDACT_CASES)
def test_redacts_known_secret_formats(text, kind):
    out, counts = ss.scrub(text)
    assert f"[REDACTED:{kind}]" in out
    assert counts.get(kind, 0) >= 1


@pytest.mark.parametrize("text", KEEP_CASES)
def test_leaves_non_secrets_alone(text):
    out, counts = ss.scrub(text)
    assert out == text
    assert counts == {}


def test_keeps_surrounding_text_verbatim():
    text = "Connect as admin with password='Sup3r!Secret' then run Get-MgUser."
    out, _ = ss.scrub(text)
    assert out == "Connect as admin with password='[REDACTED:assigned-secret]' then run Get-MgUser."


def test_idempotent():
    once, _ = ss.scrub('api_key = "abcdef123456"')
    twice, counts = ss.scrub(once)
    assert once == twice
    assert counts == {}


def test_enabled_by_default_and_switchable(monkeypatch):
    monkeypatch.delenv("MEMPALACE_SCRUB_SECRETS", raising=False)
    assert ss.enabled({}) is True
    assert ss.enabled({"scrub_secrets": False}) is False
    monkeypatch.setenv("MEMPALACE_SCRUB_SECRETS", "0")
    assert ss.enabled({"scrub_secrets": True}) is False


def test_scrub_documents_batch():
    docs, total = ss.scrub_documents(["no secrets", 'token = "abcdefghij"'])
    assert docs[0] == "no secrets"
    assert "[REDACTED:assigned-secret]" in docs[1]
    assert total == 1
    assert ss.scrub_documents(None) == (None, 0)


class TestWritePath:
    def _open(self, palace_path):
        from mempalace.palace import get_collection

        return get_collection(palace_path, create=True)

    def test_secrets_never_reach_storage(self, palace_path, monkeypatch):
        monkeypatch.delenv("MEMPALACE_SCRUB_SECRETS", raising=False)
        col = self._open(palace_path)
        col.add(
            ids=["d1"],
            documents=["Reset the SQL login. Password=Sup3rS3cret! then reconnect."],
            metadatas=[{"wing": "acme", "room": "sql", "source_file": "notes.md"}],
        )
        col.upsert(
            ids=["d2"],
            documents=["token: ghp_abcdefghijklmnopqrstuvwxyz0123456789"],
            metadatas=[{"wing": "acme", "room": "git", "source_file": "notes.md"}],
        )
        stored = col.get(ids=["d1", "d2"], include=["documents"])
        joined = " ".join(stored.documents if hasattr(stored, "documents") else stored["documents"])
        assert "Sup3rS3cret" not in joined
        assert "ghp_abcdefghijklmnopqrstuvwxyz0123456789" not in joined
        assert "[REDACTED:" in joined

    def test_disabled_stores_verbatim(self, palace_path, monkeypatch):
        monkeypatch.setenv("MEMPALACE_SCRUB_SECRETS", "0")
        col = self._open(palace_path)
        col.add(
            ids=["d1"],
            documents=["Password=Sup3rS3cret!"],
            metadatas=[{"wing": "acme", "room": "sql", "source_file": "notes.md"}],
        )
        stored = col.get(ids=["d1"], include=["documents"])
        docs = stored.documents if hasattr(stored, "documents") else stored["documents"]
        assert docs == ["Password=Sup3rS3cret!"]
