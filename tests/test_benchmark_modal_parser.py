from __future__ import annotations

import ast
import importlib.util
import json
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "benchmark_modal_parser.py"


def _load_module():
    if not MODULE_PATH.exists():
        pytest.fail("scripts/benchmark_modal_parser.py has not been implemented")
    spec = importlib.util.spec_from_file_location("benchmark_modal_parser", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_aggregate_records_reports_required_throughput_and_gpu_metrics() -> None:
    benchmark = _load_module()
    records = (
        benchmark.RequestRecord("a.pdf", 2, 1.0, True, 1, "artifacts/a.json", None),
        benchmark.RequestRecord("b.pdf", 3, 2.0, True, 2, "artifacts/b.json", None),
        benchmark.RequestRecord("c.pdf", 0, 4.0, False, 0, None, "rpc failed"),
    )
    gpu_samples = (
        benchmark.GpuSample(utilization_percent=20.0, memory_used_bytes=100),
        benchmark.GpuSample(utilization_percent=80.0, memory_used_bytes=250),
    )

    summary = benchmark.aggregate_records(
        records,
        elapsed_seconds=2.5,
        cold_request_seconds=0.75,
        gpu_samples=gpu_samples,
    )

    assert summary["total_pages"] == 5
    assert summary["pages_per_second"] == 2.0
    assert summary["success_rate"] == pytest.approx(2 / 3)
    assert summary["retry_count"] == 3
    assert summary["document_latency_seconds"]["median"] == 2.0
    assert summary["document_latency_seconds"]["p95"] == 3.8
    assert summary["gpu_utilization_percent"] == {
        "sample_count": 2,
        "mean": 50.0,
        "median": 50.0,
        "p95": 77.0,
        "peak": 80.0,
    }
    assert summary["peak_gpu_memory_bytes"] == 250
    assert summary["quality_artifact_references"] == [
        "artifacts/a.json",
        "artifacts/b.json",
    ]
    assert summary["cold_request_seconds"] == 0.75
    json.dumps(summary)


def test_concurrency_is_clamped_to_modal_container_admission_limit() -> None:
    benchmark = _load_module()

    assert benchmark.clamp_concurrency(0) == 1
    assert benchmark.clamp_concurrency(12) == 12
    assert benchmark.clamp_concurrency(36) == 35


def test_direct_modal_rpc_writes_replay_compatible_quality_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    benchmark = _load_module()
    calls: list[tuple[object, ...]] = []

    class FakeFunctionCall:
        def get(self, timeout: float):
            calls.append(("get", timeout))
            return {
                "source_pdf": "sample.pdf",
                "page_count": 2,
                "pages": [
                    {"page_number": 1, "raw_html": "<table>one</table>"},
                    {"page_number": 2, "raw_html": "<table>two</table>"},
                ],
                "ocr_metrics": {"retry_count": 2},
            }

    class FakeRemoteMethod:
        def spawn(self, *args):
            calls.append(("spawn", *args))
            return FakeFunctionCall()

    class FakeInstance:
        extract_glm_page_html = FakeRemoteMethod()

    class FakeRemoteClass:
        def __call__(self):
            return FakeInstance()

    class FakeCls:
        @staticmethod
        def from_name(app_name: str, class_name: str):
            calls.append(("from_name", app_name, class_name))
            return FakeRemoteClass()

    monkeypatch.setitem(sys.modules, "modal", SimpleNamespace(Cls=FakeCls))
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-fake")
    item = benchmark.PlannedRequest(
        request_index=1,
        repetition=1,
        pdf_path=pdf_path,
        artifact_pdf_path="Bank/sample.pdf",
    )

    instance = benchmark.resolve_modal_instance("bank-parser", "BankParserApp")
    record = benchmark.invoke_request(
        instance,
        item,
        tmp_path / "quality",
        request_timeout_seconds=17.0,
        clock=iter((10.0, 11.5)).__next__,
    )

    assert calls == [
        ("from_name", "bank-parser", "BankParserApp"),
        ("spawn", "sample.pdf", b"%PDF-fake"),
        ("get", 17.0),
    ]
    assert record.success is True
    assert record.page_count == 2
    assert record.retry_count == 2
    artifact = json.loads(Path(record.quality_artifact).read_text(encoding="utf-8"))
    assert artifact["request"] == {"pdf_path": "Bank/sample.pdf"}
    assert artifact["payload"]["debug"]["page_ocr"] == [
        {"page_index": 0, "raw_html": "<table>one</table>"},
        {"page_index": 1, "raw_html": "<table>two</table>"},
    ]
    assert artifact["payload"]["direct_ocr_response"]["page_count"] == 2
    with pytest.raises(FrozenInstanceError):
        item.repetition = 2


def test_typed_document_rpc_and_request_telemetry_are_persisted(tmp_path: Path) -> None:
    benchmark = _load_module()
    calls: list[tuple[object, ...]] = []

    class FakeFunctionCall:
        def get(self, timeout: float):
            calls.append(("get", timeout))
            return {
                "page_count": 1,
                "pages": [{"page_number": 1, "raw_html": "invoice"}],
                "timings": {"service_total": 1.25},
                "ocr_metrics": {"request_mean": 1.0},
            }

    class FakeRemoteMethod:
        def spawn(self, *args):
            calls.append(("spawn", *args))
            return FakeFunctionCall()

    pdf_path = tmp_path / "invoice.pdf"
    pdf_path.write_bytes(b"%PDF-invoice")
    item = benchmark.PlannedRequest(1, 1, pdf_path, "invoice.pdf")
    record = benchmark.invoke_request(
        SimpleNamespace(extract_document_evidence=FakeRemoteMethod()),
        item,
        tmp_path / "responses",
        document_type="invoice",
        clock=iter((3.0, 4.5)).__next__,
    )
    summary = benchmark.aggregate_records(
        (record,), elapsed_seconds=1.5, cold_request_seconds=0.0
    )

    assert calls == [
        ("spawn", "invoice.pdf", b"%PDF-invoice", "invoice"),
        ("get", 600.0),
    ]
    assert summary["request_records"][0]["service_timings"] == {
        "service_total": 1.25
    }
    assert summary["request_records"][0]["ocr_metrics"] == {
        "request_mean": 1.0
    }


def test_direct_modal_timeout_cancels_without_terminating_container(
    tmp_path: Path,
) -> None:
    benchmark = _load_module()
    calls: list[tuple[object, ...]] = []

    class FakeFunctionCall:
        def get(self, timeout: float):
            calls.append(("get", timeout))
            raise TimeoutError("request deadline")

        def cancel(self, *, terminate_containers: bool = False):
            calls.append(("cancel", terminate_containers))

    class FakeRemoteMethod:
        def spawn(self, filename: str, pdf_bytes: bytes):
            calls.append(("spawn", filename, pdf_bytes))
            return FakeFunctionCall()

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-timeout")
    item = benchmark.PlannedRequest(1, 1, pdf_path, "sample.pdf")
    instance = SimpleNamespace(extract_glm_page_html=FakeRemoteMethod())

    record = benchmark.invoke_request(
        instance,
        item,
        tmp_path / "responses",
        request_timeout_seconds=0.25,
        clock=iter((1.0, 1.25)).__next__,
    )

    assert calls == [
        ("spawn", "sample.pdf", b"%PDF-timeout"),
        ("get", 0.25),
        ("cancel", False),
    ]
    assert record.success is False
    assert record.error == "TimeoutError: request deadline"
    assert record.quality_artifact is None


def test_timeout_while_spawning_returns_failure_without_unbound_cancel(
    tmp_path: Path,
) -> None:
    benchmark = _load_module()

    class FakeRemoteMethod:
        def spawn(self, _filename: str, _pdf_bytes: bytes):
            raise TimeoutError("spawn deadline")

    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-timeout")
    item = benchmark.PlannedRequest(1, 1, pdf_path, "sample.pdf")

    record = benchmark.invoke_request(
        SimpleNamespace(extract_glm_page_html=FakeRemoteMethod()),
        item,
        tmp_path / "responses",
        clock=iter((1.0, 1.25)).__next__,
    )

    assert record.success is False
    assert record.error == "TimeoutError: spawn deadline"


def test_failed_cold_request_aborts_before_warmed_requests(tmp_path: Path) -> None:
    benchmark = _load_module()
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"sample")
    plan = (benchmark.PlannedRequest(1, 1, pdf_path, "sample.pdf"),)
    calls = 0

    def fake_invoke(_instance, item, _artifact_dir, **_kwargs):
        nonlocal calls
        calls += 1
        return benchmark.RequestRecord(
            str(item.pdf_path), 0, 0.1, False, 0, None, "TimeoutError: cold"
        )

    summary = benchmark.run_benchmark(
        SimpleNamespace(),
        plan,
        concurrency=1,
        artifact_dir=tmp_path / "responses",
        invoke=fake_invoke,
        clock=iter((0.0, 0.1)).__next__,
    )

    assert calls == 1
    assert summary["cold_request_success"] is False
    assert summary["request_count"] == 0
    assert summary["warmed_elapsed_seconds"] == 0.0


def test_cold_request_is_separate_and_warmed_run_reuses_fixed_plan(tmp_path: Path) -> None:
    benchmark = _load_module()
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"%PDF-fake")
    plan = (
        benchmark.PlannedRequest(1, 1, pdf_path, "sample.pdf"),
        benchmark.PlannedRequest(2, 2, pdf_path, "sample.pdf"),
    )
    invoked: list[tuple[object, Path]] = []
    events: list[tuple[object, ...]] = []

    def fake_invoke(
        _instance,
        item,
        artifact_dir,
        *,
        request_timeout_seconds: float,
    ):
        invoked.append((item, artifact_dir))
        events.append(("invoke", artifact_dir.name, request_timeout_seconds))
        return benchmark.RequestRecord(
            str(item.pdf_path),
            1,
            0.5,
            True,
            0,
            str(artifact_dir / f"{item.request_index}.json"),
            None,
        )

    class FakeMetricsMethod:
        def __init__(self):
            self.calls = 0

        def remote(self, **kwargs):
            self.calls += 1
            events.append(("metrics", kwargs))
            if kwargs == {"reset": True}:
                return {
                    "samples": [
                        {"gpu_utilization_percent": 99, "gpu_memory_used_bytes": 9999}
                    ]
                }
            return {
                "samples": [
                    {"gpu_utilization_percent": 60, "gpu_memory_used_bytes": 1024}
                ]
            }

    metrics_method = FakeMetricsMethod()
    instance = SimpleNamespace(gpu_metrics=metrics_method)
    times = iter((0.0, 1.25, 10.0, 14.0))

    def clock() -> float:
        value = next(times)
        events.append(("clock", value))
        return value

    summary = benchmark.run_benchmark(
        instance,
        plan,
        concurrency=99,
        artifact_dir=tmp_path / "artifacts",
        request_timeout_seconds=42.0,
        metrics_method_name="gpu_metrics",
        invoke=fake_invoke,
        clock=clock,
    )

    assert invoked[0][0] is plan[0]
    assert [entry[0] for entry in invoked[1:]] == list(plan)
    assert invoked[0][1].name == "cold"
    assert all(entry[1] == tmp_path / "artifacts" for entry in invoked[1:])
    assert summary["cold_request_seconds"] == 1.25
    assert summary["warmed_elapsed_seconds"] == 4.0
    assert summary["configured_concurrency"] == 35
    assert summary["total_pages"] == 2
    assert summary["gpu_utilization_percent"]["mean"] == 60.0
    assert metrics_method.calls == 2
    assert [event for event in events if event[0] == "metrics"] == [
        ("metrics", {"reset": True}),
        ("metrics", {}),
    ]
    reset_index = events.index(("metrics", {"reset": True}))
    warm_start_index = events.index(("clock", 10.0))
    warm_stop_index = events.index(("clock", 14.0))
    final_metrics_index = events.index(("metrics", {}))
    assert reset_index < warm_start_index < warm_stop_index < final_metrics_index
    assert [event[2] for event in events if event[0] == "invoke"] == [42.0] * 3


def test_run_benchmark_uses_non_waiting_executor_shutdown(tmp_path: Path) -> None:
    benchmark = _load_module()
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"sample")
    plan = (benchmark.PlannedRequest(1, 1, pdf_path, "sample.pdf"),)
    shutdown_calls: list[tuple[bool, bool]] = []

    class FakeExecutor:
        def __init__(self, *, max_workers: int):
            assert max_workers == 1

        def map(self, function, items):
            return [function(item) for item in items]

        def shutdown(self, *, wait: bool, cancel_futures: bool):
            shutdown_calls.append((wait, cancel_futures))

    def fake_invoke(_instance, item, _artifact_dir, **_kwargs):
        return benchmark.RequestRecord(
            str(item.pdf_path), 1, 0.1, True, 0, "response.json", None
        )

    benchmark.run_benchmark(
        SimpleNamespace(),
        plan,
        concurrency=1,
        artifact_dir=tmp_path / "responses",
        invoke=fake_invoke,
        executor_factory=FakeExecutor,
        clock=iter((0.0, 0.1, 1.0, 1.2)).__next__,
    )

    assert shutdown_calls == [(False, True)]


def test_run_benchmark_root_contains_only_replay_compatible_warmed_artifacts(
    tmp_path: Path,
) -> None:
    benchmark = _load_module()
    first = tmp_path / "corpus" / "Bank" / "first.pdf"
    second = tmp_path / "corpus" / "Bank" / "second.pdf"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"%PDF-first")
    second.write_bytes(b"%PDF-second")
    plan = (
        benchmark.PlannedRequest(1, 1, first, "Bank/first.pdf"),
        benchmark.PlannedRequest(2, 1, second, "Bank/second.pdf"),
    )

    class FakeFunctionCall:
        def __init__(self, filename: str):
            self.filename = filename

        def get(self, timeout: float):
            assert timeout == 600.0
            return {
                "source_pdf": self.filename,
                "page_count": 1,
                "pages": [
                    {
                        "page_number": 1,
                        "raw_html": f"<table>{self.filename}</table>",
                    }
                ],
            }

    class FakeRemoteMethod:
        def spawn(self, filename: str, _pdf_bytes: bytes):
            return FakeFunctionCall(filename)

    instance = SimpleNamespace(extract_glm_page_html=FakeRemoteMethod())
    response_dir = tmp_path / "responses"

    summary = benchmark.run_benchmark(
        instance,
        plan,
        concurrency=2,
        artifact_dir=response_dir,
        clock=iter((0.0, 1.0, 10.0, 12.0)).__next__,
    )

    warmed_paths = sorted(response_dir.glob("*.json"))
    assert len(warmed_paths) == len(plan)
    assert sorted(summary["quality_artifact_references"]) == sorted(
        str(path) for path in warmed_paths
    )
    assert len(list((response_dir / "cold").glob("*.json"))) == 1
    assert Path(summary["cold_quality_artifact_reference"]).parent.name == "cold"
    payloads = [json.loads(path.read_text(encoding="utf-8")) for path in warmed_paths]
    assert [payload["request"]["pdf_path"] for payload in payloads] == [
        "Bank/first.pdf",
        "Bank/second.pdf",
    ]
    assert all(payload["payload"]["debug"]["page_ocr"] for payload in payloads)


def test_cli_accepts_explicit_pdfs_or_corpus_and_repetitions(tmp_path: Path) -> None:
    benchmark = _load_module()
    corpus_root = tmp_path / "corpus"
    first = corpus_root / "Bank" / "first.pdf"
    first_csv = first.with_suffix(".csv")
    second = corpus_root / "nested" / "second.PDF"
    first.parent.mkdir(parents=True)
    first.write_bytes(b"one")
    first_csv.write_text("Date,Amount\n", encoding="utf-8")
    second.parent.mkdir()
    second.write_bytes(b"two")

    explicit = benchmark.parse_args(
        [
            "--app-name",
            "app",
            "--class-name",
            "Class",
            "--pdf",
            str(first),
            "--corpus-root",
            str(corpus_root),
            "--concurrency",
            "40",
            "--output-json",
            str(tmp_path / "result.json"),
            "--repetitions",
            "2",
            "--request-timeout-seconds",
            "30",
            "--metrics-method",
            "gpu_metrics",
        ]
    )
    corpus = benchmark.parse_args(
        [
            "--app-name",
            "app",
            "--class-name",
            "Class",
            "--corpus-root",
            str(corpus_root),
            "--output-json",
            str(tmp_path / "corpus.json"),
        ]
    )

    assert explicit.pdf_paths == [first]
    assert explicit.concurrency == 40
    assert explicit.repetitions == 2
    assert explicit.request_timeout_seconds == 30.0
    assert explicit.metrics_method == "gpu_metrics"
    assert explicit.corpus_root == corpus_root
    assert corpus.corpus_root == corpus_root
    assert corpus.request_timeout_seconds == 600.0
    explicit_plan = benchmark.build_request_plan(
        pdf_paths=explicit.pdf_paths,
        corpus_root=explicit.corpus_root,
        repetitions=explicit.repetitions,
    )
    assert [item.pdf_path for item in explicit_plan] == [first, first]
    artifact = benchmark._artifact_payload(explicit_plan[0], {"pages": []})
    request_pdf_path = Path(artifact["request"]["pdf_path"])
    assert request_pdf_path == Path("Bank/first.pdf")
    assert corpus_root / request_pdf_path.with_suffix(".csv") == first_csv
    plan = benchmark.build_request_plan(
        pdf_paths=None,
        corpus_root=corpus_root,
        repetitions=2,
    )
    assert [item.pdf_path for item in plan] == [first, second, first, second]
    assert [item.artifact_pdf_path for item in plan] == [
        "Bank/first.pdf",
        "nested/second.PDF",
        "Bank/first.pdf",
        "nested/second.PDF",
    ]


def test_explicit_pdf_must_be_under_corpus_root(tmp_path: Path) -> None:
    benchmark = _load_module()
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    outside_pdf = tmp_path / "outside.pdf"
    outside_pdf.write_bytes(b"outside")

    with pytest.raises(ValueError, match="outside corpus root"):
        benchmark.build_request_plan(
            pdf_paths=[outside_pdf],
            corpus_root=corpus_root,
            repetitions=1,
        )


def test_explicit_pdf_requires_corpus_root_at_cli_and_plan_boundaries(
    tmp_path: Path,
) -> None:
    benchmark = _load_module()
    pdf_path = tmp_path / "sample.pdf"
    pdf_path.write_bytes(b"sample")

    with pytest.raises(SystemExit):
        benchmark.parse_args(
            [
                "--app-name",
                "app",
                "--class-name",
                "Class",
                "--pdf",
                str(pdf_path),
                "--output-json",
                str(tmp_path / "result.json"),
            ]
        )
    with pytest.raises(ValueError, match="requires corpus root"):
        benchmark.build_request_plan(
            pdf_paths=[pdf_path],
            corpus_root=None,
            repetitions=1,
        )


def test_cli_rejects_non_positive_request_timeout(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    benchmark = _load_module()
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()

    with pytest.raises(SystemExit):
        benchmark.parse_args(
            [
                "--app-name",
                "app",
                "--class-name",
                "Class",
                "--corpus-root",
                str(corpus_root),
                "--request-timeout-seconds",
                "0",
                "--output-json",
                str(tmp_path / "result.json"),
            ]
        )

    assert "positive" in capsys.readouterr().err


def test_main_forwards_cli_request_timeout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    benchmark = _load_module()
    corpus_root = tmp_path / "corpus"
    corpus_root.mkdir()
    (corpus_root / "sample.pdf").write_bytes(b"sample")
    captured: dict[str, object] = {}

    monkeypatch.setattr(benchmark, "resolve_modal_instance", lambda *_args: object())

    def fake_run(_instance, _plan, **kwargs):
        captured.update(kwargs)
        return {"success_rate": 1.0, "cold_request_success": True}

    monkeypatch.setattr(benchmark, "run_benchmark", fake_run)

    exit_code = benchmark.main(
        [
            "--app-name",
            "app",
            "--class-name",
            "Class",
            "--corpus-root",
            str(corpus_root),
            "--request-timeout-seconds",
            "12.5",
            "--output-json",
            str(tmp_path / "result.json"),
        ]
    )

    assert exit_code == 0
    assert captured["request_timeout_seconds"] == 12.5


def test_benchmark_source_has_no_http_client_or_public_health_url() -> None:
    source = MODULE_PATH.read_text(encoding="utf-8") if MODULE_PATH.exists() else ""
    tree = ast.parse(source)
    imports = {
        alias.name.split(".", 1)[0]
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in (
            node.names
            if isinstance(node, ast.Import)
            else [SimpleNamespace(name=node.module or "")]
        )
    }

    assert "modal.Cls.from_name" in source
    assert "extract_glm_page_html.spawn" in source
    assert ".get(timeout=" in source
    assert ".cancel(terminate_containers=False)" in source
    assert imports.isdisjoint({"requests", "httpx", "urllib", "aiohttp"})
    for forbidden in ("/health", "http://", "https://"):
        assert forbidden not in source
