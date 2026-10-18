"""Opening a collection to read must not write to chroma.sqlite3.

Every search opens the collection. A write on open moves the database file's mtime,
which invalidates the hnsw_capacity_status cache (it is keyed on file mtimes) and
turns a ~50 ms search into a ~300 ms one on a large palace.
"""

import os

from mempalace.palace import get_collection


def _db_mtime_ns(palace_path):
    return os.stat(os.path.join(palace_path, "chroma.sqlite3")).st_mtime_ns


def test_read_open_does_not_touch_the_database_file(palace_path):
    col = get_collection(palace_path, create=True)
    col.add(
        ids=["d1"],
        documents=["a note about backups and the nightly job"],
        metadatas=[{"wing": "w", "room": "r", "source_file": "a.md"}],
    )
    # First read open may settle client files; measure from the second one on.
    get_collection(palace_path, create=False, read_only=True)
    before = _db_mtime_ns(palace_path)
    for _ in range(3):
        get_collection(palace_path, create=False, read_only=True)
    assert _db_mtime_ns(palace_path) == before
