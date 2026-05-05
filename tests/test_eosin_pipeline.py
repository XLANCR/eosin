import importlib.util
import sys
import types
import uuid
from pathlib import Path

import pandas as pd
import pytest


ROOT = Path(__file__).resolve().parents[1]
PIPELINE_PATH = ROOT / "eosin" / "backend" / "eosin_pipeline.py"
SERVICE_PATH = ROOT / "eosin" / "backend" / "bank_parser_service.py"
API_PATH = ROOT / "eosin" / "backend" / "bank_parser_api.py"


class DummyLayoutDetector:
    def __init__(self, config):
        self.config = config
        self.batch_size = 1
        self.started = False
        self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True

    def process(self, page_images):
        return [[] for _ in page_images], None


class RaisingLayoutDetector:
    def __init__(self, _config):
        raise TypeError("NoneType is not callable")


class DummyPageLoader:
    def __init__(self, config):
        self.config = config


class DummyOCRClient:
    def __init__(self, config):
        self.config = config
        self._pool_maxsize = 4

    def start(self):
        return None

    def stop(self):
        return None


class DummyDispatcher:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def close(self):
        return None

    def submit(self, *_args, **_kwargs):
        raise AssertionError("submit should not be called in these tests")


class DummyMetricsManager:
    def start_background_samplers(self):
        return None

    def render_metrics(self):
        return types.SimpleNamespace(payload="", media_type="text/plain")


def _load_module(module_name: str, module_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_stub_modules(detector_factory) -> None:
    for name in (
        "eosin.backend.bank_parser_api",
        "eosin.backend.bank_parser_service",
        "eosin.backend.eosin_pipeline",
        "eosin.backend.ocr_pipeline",
        "eosin.backend.metrics",
        "eosin.backend",
        "eosin",
        "glmocr.utils.image_utils",
        "glmocr.utils",
        "glmocr.ocr_client",
        "glmocr.layout",
        "glmocr.dataloader",
        "glmocr.config",
        "glmocr",
    ):
        sys.modules.pop(name, None)

    glmocr_module = types.ModuleType("glmocr")
    glmocr_config = types.ModuleType("glmocr.config")
    glmocr_config.load_config = lambda *_args, **_kwargs: types.SimpleNamespace(
        pipeline=types.SimpleNamespace(
            layout=types.SimpleNamespace(name="layout"),
            page_loader=types.SimpleNamespace(pdf_dpi=150),
            ocr_api=types.SimpleNamespace(name="ocr"),
            max_workers=4,
        )
    )
    glmocr_dataloader = types.ModuleType("glmocr.dataloader")
    glmocr_dataloader.PageLoader = DummyPageLoader
    glmocr_layout = types.ModuleType("glmocr.layout")
    glmocr_layout.PPDocLayoutDetector = detector_factory
    glmocr_ocr_client = types.ModuleType("glmocr.ocr_client")
    glmocr_ocr_client.OCRClient = DummyOCRClient
    glmocr_utils = types.ModuleType("glmocr.utils")
    glmocr_image_utils = types.ModuleType("glmocr.utils.image_utils")
    glmocr_image_utils.crop_image_region = lambda *args, **kwargs: None
    glmocr_image_utils.pdf_to_images_pil = lambda *args, **kwargs: []

    eosin_package = types.ModuleType("eosin")
    eosin_package.__path__ = []  # type: ignore[attr-defined]
    eosin_backend = types.ModuleType("eosin.backend")
    eosin_backend.__path__ = []  # type: ignore[attr-defined]
    eosin_ocr_pipeline = types.ModuleType("eosin.backend.ocr_pipeline")
    eosin_ocr_pipeline.OCRPipelineDispatcher = DummyDispatcher
    eosin_ocr_pipeline.OCRTaskResult = object
    eosin_metrics = types.ModuleType("eosin.backend.metrics")
    eosin_metrics.get_metrics_manager = lambda: DummyMetricsManager()

    torch_module = types.ModuleType("torch")
    torch_module.OutOfMemoryError = RuntimeError
    torch_module.cuda = types.SimpleNamespace(
        is_available=lambda: False,
        empty_cache=lambda: None,
    )

    sys.modules.setdefault("cv2", types.ModuleType("cv2"))
    sys.modules.setdefault("fitz", types.ModuleType("fitz"))
    sys.modules["torch"] = torch_module
    sys.modules["glmocr"] = glmocr_module
    sys.modules["glmocr.config"] = glmocr_config
    sys.modules["glmocr.dataloader"] = glmocr_dataloader
    sys.modules["glmocr.layout"] = glmocr_layout
    sys.modules["glmocr.ocr_client"] = glmocr_ocr_client
    sys.modules["glmocr.utils"] = glmocr_utils
    sys.modules["glmocr.utils.image_utils"] = glmocr_image_utils
    sys.modules["eosin"] = eosin_package
    sys.modules["eosin.backend"] = eosin_backend
    sys.modules["eosin.backend.ocr_pipeline"] = eosin_ocr_pipeline
    sys.modules["eosin.backend.metrics"] = eosin_metrics


def _load_runtime_modules(detector_factory=DummyLayoutDetector):
    _install_stub_modules(detector_factory)
    pipeline_module = _load_module("eosin.backend.eosin_pipeline", PIPELINE_PATH)
    service_module = _load_module("eosin.backend.bank_parser_service", SERVICE_PATH)
    api_module = _load_module(
        f"test_bank_parser_api_module_{uuid.uuid4().hex}",
        API_PATH,
    )
    return pipeline_module, service_module, api_module


def _load_eosin_pipeline_module(detector_factory=DummyLayoutDetector):
    pipeline_module, _, _ = _load_runtime_modules(detector_factory)
    return pipeline_module


def test_layout_auto_keeps_detector_when_available():
    module = _load_eosin_pipeline_module()

    parser = module.BankStatementParser(layout_mode="auto")

    assert parser.layout_mode == "auto"
    assert parser.layout_detector is not None
    parser.close()


def test_layout_auto_falls_back_when_detector_init_fails():
    module = _load_eosin_pipeline_module(RaisingLayoutDetector)

    parser = module.BankStatementParser(layout_mode="auto")

    assert parser.layout_mode == "auto"
    assert parser.layout_detector is None
    parser.close()


def test_layout_required_still_fails_when_detector_init_fails():
    module = _load_eosin_pipeline_module(RaisingLayoutDetector)

    with pytest.raises(RuntimeError, match="failed to initialize layout detector"):
        module.BankStatementParser(layout_mode="required")


def test_select_best_table_candidate_prefers_bank_transaction_table():
    module = _load_eosin_pipeline_module()
    html_tables = [
        """
        <table>
          <tr><th>Account Summary</th><th>Value</th></tr>
          <tr><td>Closing Balance</td><td>1000.00</td></tr>
        </table>
        """,
        """
        <table>
          <tr><th>Date</th><th>Description</th><th>Debit</th><th>Credit</th><th>Balance</th></tr>
          <tr><td>01/01/2024</td><td>ATM Withdrawal</td><td>500.00</td><td></td><td>9500.00</td></tr>
          <tr><td>02/01/2024</td><td>Salary</td><td></td><td>20000.00</td><td>29500.00</td></tr>
        </table>
        """,
    ]

    selected = module.select_best_table_candidate(html_tables)

    assert selected is not None
    index, _, dataframe, headers = selected
    assert index == 1
    assert headers == ["Date", "Description", "Debit", "Credit", "Balance"]
    assert isinstance(dataframe, pd.DataFrame)
    assert len(dataframe) == 2


def test_select_best_table_candidate_prefers_header_match_on_followup_pages():
    module = _load_eosin_pipeline_module()
    html_tables = [
        """
        <table>
          <tr><th>Summary</th><th>Total</th></tr>
          <tr><td>Debits</td><td>500.00</td></tr>
        </table>
        """,
        """
        <table>
          <tr><td>03/01/2024</td><td>UPI Payment</td><td>250.00</td><td></td><td>29250.00</td></tr>
          <tr><td>04/01/2024</td><td>POS Purchase</td><td>120.00</td><td></td><td>29130.00</td></tr>
        </table>
        """,
    ]

    selected = module.select_best_table_candidate(
        html_tables,
        expected_headers=["Date", "Description", "Debit", "Credit", "Balance"],
    )

    assert selected is not None
    index, _, dataframe, _ = selected
    assert index == 1
    assert list(dataframe.columns) == ["Date", "Description", "Debit", "Credit", "Balance"]
    assert len(dataframe) == 2


def test_select_best_table_candidate_rejects_account_summary_only_payload():
    module = _load_eosin_pipeline_module()

    selected = module.select_best_table_candidate(
        [
            """
            <table>
              <tr><th>Deposit Accounts</th><th>Current Balance</th></tr>
              <tr><td>Savings Account</td><td>1200.00</td></tr>
            </table>
            """
        ]
    )

    assert selected is None


def test_select_best_table_candidate_keeps_transaction_table_with_opening_balance_row():
    module = _load_eosin_pipeline_module()
    html_tables = [
        """
        <table>
          <tr><th>Date</th><th>Description</th><th>Debit</th><th>Credit</th><th>Balance</th></tr>
          <tr><td>01/01/2024</td><td>Opening Balance</td><td></td><td></td><td>1000.00</td></tr>
          <tr><td>02/01/2024</td><td>UPI Payment</td><td>250.00</td><td></td><td>750.00</td></tr>
        </table>
        """
    ]

    selected = module.select_best_table_candidate(html_tables)

    assert selected is not None
    _, _, dataframe, headers = selected
    assert headers == ["Date", "Description", "Debit", "Credit", "Balance"]
    assert len(dataframe) == 2


def test_extract_transaction_dataframes_combines_split_pages():
    module = _load_eosin_pipeline_module()
    parser = module.BankStatementParser(layout_mode="disabled")

    all_dfs, expected_headers, selected_pages = parser._extract_transaction_dataframes(
        [
            (
                0,
                """
                <table>
                  <tr><th>Date</th><th>Description</th><th>Debit</th><th>Credit</th><th>Balance</th></tr>
                  <tr><td>01/01/2024</td><td>Salary</td><td></td><td>5000.00</td><td>5000.00</td></tr>
                </table>
                """,
            ),
            (
                1,
                """
                <table>
                  <tr><th>Account Summary</th><th>Total</th></tr>
                  <tr><td>Credits</td><td>5000.00</td></tr>
                </table>
                <table>
                  <tr><td>02/01/2024</td><td>ATM Withdrawal</td><td>500.00</td><td></td><td>4500.00</td></tr>
                </table>
                """,
            ),
        ],
        "fixture.pdf",
    )

    assert expected_headers == ["Date", "Description", "Debit", "Credit", "Balance"]
    assert selected_pages == [0, 1]
    assert len(all_dfs) == 2
    combined = parser._finalize_extracted_tables(all_dfs, expected_headers)
    assert len(combined) == 2
    assert list(combined.columns) == ["Date", "Description", "Debit", "Credit", "Balance"]
    parser.close()
