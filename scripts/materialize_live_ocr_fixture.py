"""Materialize live Modal OCR responses as an isolated evaluation fixture."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize_fixture(
    artifact_dir: Path,
    output_root: Path,
    *,
    corpus_root: Path | None = None,
    expected_documents: int | None = None,
) -> dict[str, object]:
    artifacts = sorted(artifact_dir.glob("*.json"))
    if expected_documents is not None and len(artifacts) != expected_documents:
        raise ValueError(
            f"expected {expected_documents} warmed artifacts, found {len(artifacts)}"
        )
    if not artifacts:
        raise ValueError(f"no warmed JSON artifacts found in {artifact_dir}")

    output_root.mkdir(parents=True, exist_ok=True)
    documents: list[dict[str, object]] = []
    for artifact_path in artifacts:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
        request = artifact.get("request", {})
        payload = artifact.get("payload", {})
        if not isinstance(request, Mapping) or not isinstance(payload, Mapping):
            raise ValueError(f"invalid replay artifact structure: {artifact_path}")
        relative_pdf = Path(str(request.get("pdf_path", "")))
        if not relative_pdf.name:
            raise ValueError(f"artifact has no request PDF path: {artifact_path}")
        response = payload.get("direct_ocr_response", {})
        if not isinstance(response, Mapping):
            raise ValueError(f"artifact has no direct OCR response: {artifact_path}")
        pages = response.get("pages", [])
        if not isinstance(pages, list) or not pages:
            raise ValueError(f"artifact has no OCR pages: {artifact_path}")

        stem = relative_pdf.stem
        document_dir = output_root / stem
        if document_dir.exists():
            raise FileExistsError(f"fixture document already exists: {document_dir}")
        document_dir.mkdir()
        html_map = {
            str(int(page.get("page_number", index + 1))): str(page.get("raw_html", ""))
            for index, page in enumerate(pages)
            if isinstance(page, Mapping)
        }
        (document_dir / "glm_html_cache.json").write_text(
            json.dumps(html_map, indent=2, ensure_ascii=False), encoding="utf-8"
        )

        source_pdf = corpus_root / relative_pdf if corpus_root is not None else None
        metadata = {
            "source_relative_pdf": relative_pdf.as_posix(),
            "content_sha256": (
                _sha256(source_pdf) if source_pdf is not None and source_pdf.is_file() else None
            ),
            "artifact_sha256": _sha256(artifact_path),
            "page_count": int(response.get("page_count", len(html_map))),
            "suspicious_page_count": sum(
                bool(page.get("suspicious"))
                for page in pages
                if isinstance(page, Mapping)
            ),
            "timings": dict(response.get("timings", {})),
            "ocr_metrics": dict(response.get("ocr_metrics", {})),
            "document_type": response.get("document_type", "bank_statement"),
            "ocr_task_type": response.get("ocr_task_type", "table"),
        }
        (document_dir / "live_ocr_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        documents.append({"stem": stem, **metadata})

    manifest = {
        "fixture_kind": "live_modal_ocr_evaluation_only",
        "document_count": len(documents),
        "total_pages": sum(int(item["page_count"]) for item in documents),
        "documents": documents,
    }
    (output_root / "live_ocr_fixture_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path)
    parser.add_argument("--expected-documents", type=int)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    manifest = materialize_fixture(
        args.artifact_dir,
        args.output_root,
        corpus_root=args.corpus_root,
        expected_documents=args.expected_documents,
    )
    print(json.dumps({
        "document_count": manifest["document_count"],
        "total_pages": manifest["total_pages"],
        "output_root": str(args.output_root),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
