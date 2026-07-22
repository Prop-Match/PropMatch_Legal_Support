import json

from app.chunking import chunk_corpus


def test_chunk_corpus_keeps_article_and_manifest_metadata(tmp_path):
    manifest = [
        {
            "file": "law.txt",
            "title": "قانون الاختبار",
            "source": "https://example.test/law",
            "method": "text extraction",
        }
    ]
    (tmp_path / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (tmp_path / "law.txt").write_text(
        "معلومات رأسية\n==================== بداية النص ====================\n"
        "المادة الأولى\nهذا نص المادة الأولى.\n\nالمادة الثانية\nهذا نص المادة الثانية.",
        encoding="utf-8",
    )

    chunks = chunk_corpus(tmp_path, size=300, overlap=20)

    assert len(chunks) == 2
    assert chunks[0].metadata["title"] == "قانون الاختبار"
    assert chunks[0].metadata["article"] == "المادة الأولى"
    assert chunks[0].metadata["source_url"] == "https://example.test/law"
    assert chunks[0].id != chunks[1].id


def test_chunk_ids_are_deterministic(tmp_path):
    (tmp_path / "manifest.json").write_text(
        json.dumps([{"file": "law.txt", "title": "قانون"}]), encoding="utf-8"
    )
    (tmp_path / "law.txt").write_text("المادة الأولى\nنص ثابت", encoding="utf-8")

    first = chunk_corpus(tmp_path, size=300, overlap=20)
    second = chunk_corpus(tmp_path, size=300, overlap=20)

    assert [chunk.id for chunk in first] == [chunk.id for chunk in second]

