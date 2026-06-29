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
TRANSACTION_RECONSTRUCTION_PATH = ROOT / "eosin" / "backend" / "transaction_reconstruction.py"


def _is_stubbed_runtime_module(name: str) -> bool:
    return name == "torch" or name.startswith(("eosin", "glmocr"))


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
        "eosin.backend.transaction_reconstruction",
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
    original = {
        name: module
        for name, module in sys.modules.items()
        if _is_stubbed_runtime_module(name)
    }
    try:
        _install_stub_modules(detector_factory)
        _load_module(
            "eosin.backend.transaction_reconstruction",
            TRANSACTION_RECONSTRUCTION_PATH,
        )
        pipeline_module = _load_module("eosin.backend.eosin_pipeline", PIPELINE_PATH)
        service_module = _load_module("eosin.backend.bank_parser_service", SERVICE_PATH)
        api_module = _load_module(
            f"test_bank_parser_api_module_{uuid.uuid4().hex}",
            API_PATH,
        )
    finally:
        for name in [name for name in sys.modules if _is_stubbed_runtime_module(name)]:
            sys.modules.pop(name, None)
        sys.modules.update(original)
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


def test_parse_html_table_tolerates_malformed_colspan_from_model_output():
    module = _load_eosin_pipeline_module()
    html = """
    <table>
      <tr><th>Date</th><th>Description</th><th>Debit</th><th>Credit</th><th>Balance</th></tr>
      <tr>
        <td>01/04/2025</td>
        <td colspan='2>Surcharge on Usage of Debit Card at Petrol Pumps/Railway Ticket Purchase or Hotel Tips</td>
        <td>12.00</td>
        <td></td>
        <td>988.00</td>
      </tr>
    </table>
    """

    headers = module.extract_headers_from_html(html)
    dataframe = module.parse_html_table(html)

    assert headers[:5] == ["Date", "Description", "Debit", "Credit", "Balance"]
    assert isinstance(dataframe, pd.DataFrame)


def test_identify_main_tables_fallback_accepts_parallel_results_with_metrics(monkeypatch):
    module = _load_eosin_pipeline_module()
    parser = module.BankStatementParser.__new__(module.BankStatementParser)
    parser.MIN_TABLE_AREA_RATIO = 0.01

    def fake_ocr_tables_parallel(candidate_images):
        unique_idx, _ = candidate_images[0]
        return [(unique_idx, "<table><tr><th>Date</th><th>Debit</th><th>Balance</th></tr></table>")], {
            "task_count": len(candidate_images)
        }

    monkeypatch.setattr(parser, "_ocr_tables_parallel", fake_ocr_tables_parallel)

    page_images = [module.Image.new("RGB", (1000, 1000), "white")]
    main_tables = parser._identify_main_tables(
        [
            [
                {"label": "table", "bbox_2d": [0, 0, 800, 800]},
                {"label": "table", "bbox_2d": [0, 0, 700, 700]},
            ]
        ],
        page_images,
    )

    assert main_tables == [[0, 0, 800, 800]]


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


def test_parse_html_table_keeps_transaction_thead_as_data_and_infers_axis_columns():
    module = _load_eosin_pipeline_module()

    dataframe = module.parse_html_table(
        """
        <table>
          <thead>
            <tr><th>21-01-2020</th><th></th><th>UPI/P2A/002122279638/BHARATH S/IDBI Bank/UPI</th><th>200.00</th><th></th><th>4801.25</th><th>227</th></tr>
          </thead>
          <tbody>
            <tr><td>22-01-2020</td><td></td><td>UPI/P2M/002210706586/Green Baw/Paytm Pay/UPI</td><td>60.00</td><td></td><td>4741.25</td><td>227</td></tr>
          </tbody>
        </table>
        """
    )

    assert list(dataframe.columns) == [
        "Tran Date",
        "col_1",
        "Particulars",
        "Debit",
        "Credit",
        "Balance",
        "Init. Br",
    ]
    assert dataframe.iloc[0]["Tran Date"] == "21-01-2020"
    assert dataframe.iloc[0]["Particulars"] == "UPI/P2A/002122279638/BHARATH S/IDBI Bank/UPI"
    assert dataframe.iloc[0]["Debit"] == "200.00"
    assert dataframe.iloc[0]["Balance"] == "4801.25"
    assert dataframe.iloc[0]["Init. Br"] == "227"


def test_parse_html_table_infers_hdfc_headerless_transaction_columns():
    module = _load_eosin_pipeline_module()

    dataframe = module.parse_html_table(
        """
        <table>
          <tr><td>02/09/20</td><td>NWD CASH WITHDRAWAL</td><td>000024611012483</td><td>02/09/20</td><td>4,000.00</td><td></td><td>16,158.66</td></tr>
          <tr><td>03/09/20</td><td>SALARY CREDIT</td><td>000000000000002</td><td>03/09/20</td><td></td><td>24,690.00</td><td>40,848.66</td></tr>
        </table>
        """
    )

    assert list(dataframe.columns) == [
        "Transaction Date",
        "Description",
        "Reference",
        "Value Date",
        "Debit",
        "Credit",
        "Balance",
    ]
    assert dataframe.iloc[0]["Description"] == "NWD CASH WITHDRAWAL"
    assert dataframe.iloc[0]["Reference"] == "000024611012483"
    assert dataframe.iloc[0]["Debit"] == "4,000.00"


def test_parse_html_table_infers_hdfc_single_amount_columns():
    module = _load_eosin_pipeline_module()

    dataframe = module.parse_html_table(
        """
        <table>
          <tr><td>09/10/20</td><td>AUTOPAY</td><td>00000029053940</td><td>09/10/20</td><td>2,848.00</td><td>4,373.34</td></tr>
          <tr><td>31/10/20</td><td>SALARY CREDIT</td><td>000000000000002</td><td>31/10/20</td><td>26,342.00</td><td>26,475.34</td></tr>
        </table>
        """,
        expected_headers=[
            "Date",
            "Narration",
            "Chg/Ref.No.",
            "Value Dt",
            "Withdrawal Anti.",
            "Deposit Amt.",
            "Closing Balance",
        ],
    )

    assert list(dataframe.columns) == [
        "Transaction Date",
        "Description",
        "Reference",
        "Value Date",
        "Amount",
        "Balance",
    ]
    assert dataframe.iloc[0]["Amount"] == "2,848.00"
    assert dataframe.iloc[0]["Balance"] == "4,373.34"


def test_parse_html_table_infers_hdfc_clipped_balance_columns():
    module = _load_eosin_pipeline_module()

    dataframe = module.parse_html_table(
        """
        <table>
          <tr><td>25/12/20</td><td>CREDIT PAYMENT</td><td>000000000000002</td><td>25/12/20</td><td></td><td>1.00</td></tr>
          <tr><td>28/12/20</td><td>LOAN DEBIT</td><td>000037814900000</td><td>28/12/20</td><td>1,000.00</td><td></td></tr>
        </table>
        """
    )

    assert list(dataframe.columns) == [
        "Transaction Date",
        "Description",
        "Reference",
        "Value Date",
        "Debit",
        "Credit",
    ]
    assert dataframe.iloc[0]["Credit"] == "1.00"
    assert dataframe.iloc[1]["Debit"] == "1,000.00"


def test_parse_html_table_names_blank_axis_date_header_from_row_shape():
    module = _load_eosin_pipeline_module()

    dataframe = module.parse_html_table(
        """
        <table>
          <thead>
            <tr><th></th><th></th><th>Particulars</th><th>Debit</th><th>Credit</th><th>Balance</th><th>Init. Br</th></tr>
          </thead>
          <tbody>
            <tr><td>02-01-2020</td><td></td><td>POS/SANAT NAGAR,/HYDERABAD/020120/15:40</td><td>1110.00</td><td></td><td>13605.00</td><td>227</td></tr>
          </tbody>
        </table>
        """
    )

    assert list(dataframe.columns) == [
        "Tran Date",
        "col_1",
        "Particulars",
        "Debit",
        "Credit",
        "Balance",
        "Init. Br",
    ]
    assert dataframe.iloc[0]["Tran Date"] == "02-01-2020"
    assert dataframe.iloc[0]["Particulars"] == "POS/SANAT NAGAR,/HYDERABAD/020120/15:40"


def test_parse_html_table_aligns_headerless_continuation_rows_to_expected_blank_reference_column():
    module = _load_eosin_pipeline_module()

    dataframe = module.parse_html_table(
        """
        <table>
          <thead>
            <tr><th>04-03-2020</th><th>POS/SRI SAI SERVICES/HYDERABAD/040320/07:20</th><th>1010.00</th><th></th><th>2146.93</th><th>227</th></tr>
          </thead>
          <tbody>
            <tr><td>05-03-2020</td><td>BY TRANSFER</td><td></td><td>49750.00</td><td>51396.93</td><td>227</td></tr>
          </tbody>
        </table>
        """,
        expected_headers=["Tran Date", "col_1", "Particulars", "Debit", "Credit", "Balance", "Init. Br"],
    )

    assert list(dataframe.columns) == ["Tran Date", "col_1", "Particulars", "Debit", "Credit", "Balance", "Init. Br"]
    assert dataframe.iloc[0].to_dict() == {
        "Tran Date": "04-03-2020",
        "col_1": "",
        "Particulars": "POS/SRI SAI SERVICES/HYDERABAD/040320/07:20",
        "Debit": "1010.00",
        "Credit": "",
        "Balance": "2146.93",
        "Init. Br": "227",
    }
    assert dataframe.iloc[1].to_dict() == {
        "Tran Date": "05-03-2020",
        "col_1": "",
        "Particulars": "BY TRANSFER",
        "Debit": "",
        "Credit": "49750.00",
        "Balance": "51396.93",
        "Init. Br": "227",
    }


def test_parse_html_table_aligns_colspan_description_to_expected_blank_reference_column():
    module = _load_eosin_pipeline_module()

    dataframe = module.parse_html_table(
        """
        <table>
          <tbody>
            <tr><td>05-05-2020</td><td colspan="2">BY SALARY</td><td></td><td>24123.00</td><td>24316.46</td><td>227</td></tr>
          </tbody>
        </table>
        """,
        expected_headers=["Tran Date", "col_1", "Particulars", "Debit", "Credit", "Balance", "Init. Br"],
    )

    assert dataframe.iloc[0].to_dict() == {
        "Tran Date": "05-05-2020",
        "col_1": "",
        "Particulars": "BY SALARY",
        "Debit": "",
        "Credit": "24123.00",
        "Balance": "24316.46",
        "Init. Br": "227",
    }


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


def test_extract_transaction_dataframes_keeps_multiple_tables_from_one_document_response():
    module = _load_eosin_pipeline_module()
    parser = module.BankStatementParser.__new__(module.BankStatementParser)

    combined_html = """
    <table>
      <tr><th>Account Summary</th><th>Value</th></tr>
      <tr><td>Closing Balance</td><td>1000.00</td></tr>
    </table>
    <table>
      <tr><th>Date</th><th>Description</th><th>Debit</th><th>Credit</th><th>Balance</th></tr>
      <tr><td>01/01/2024</td><td>ATM Withdrawal</td><td>500.00</td><td></td><td>9500.00</td></tr>
      <tr><td>02/01/2024</td><td>Salary</td><td></td><td>20000.00</td><td>29500.00</td></tr>
    </table>
    <table>
      <tr><th>Date</th><th>Description</th><th>Debit</th><th>Credit</th><th>Balance</th></tr>
      <tr><td>03/01/2024</td><td>UPI Payment</td><td>250.00</td><td></td><td>29250.00</td></tr>
    </table>
    """

    all_dfs, expected_headers, selected_pages = parser._extract_transaction_dataframes(
        [(0, combined_html)],
        "dummy.pdf",
    )

    assert expected_headers == ["Date", "Description", "Debit", "Credit", "Balance"]
    assert selected_pages == [0]
    assert len(all_dfs) == 1
    assert sum(len(df) for df in all_dfs) == 2


def test_remove_header_rows_drops_fuzzy_repeated_header_rows():
    module = _load_eosin_pipeline_module()
    parser = module.BankStatementParser.__new__(module.BankStatementParser)

    df = pd.DataFrame(
        [
            ["Date", "Description/Narration", "Value date", "Chq/Ref. No.", "Debit(D"],
            ["01/01/2024", "UPI Payment", "01/01/2024", "12345", "250.00"],
        ],
        columns=["Date", "Description/Narration", "Value date", "Chq/Ref. No.", "Debit(Dr.)"],
    )

    cleaned = parser._remove_header_rows(
        df,
        ["Date", "Description/Narration", "Value date", "Chq/Ref. No.", "Debit(Dr.)"],
    )

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["Date"] == "01/01/2024"


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


def test_finalize_extracted_tables_preserves_identical_transaction_rows():
    module = _load_eosin_pipeline_module()
    parser = module.BankStatementParser.__new__(module.BankStatementParser)

    first_page = pd.DataFrame(
        [
            {
                "Date": "01/01/2024",
                "Description": "UPI PAYMENT",
                "Debit": "250.00",
                "Credit": "",
                "Balance": "750.00",
            },
            {
                "Date": "02/01/2024",
                "Description": "ATM Withdrawal",
                "Debit": "100.00",
                "Credit": "",
                "Balance": "650.00",
            },
        ]
    )
    second_page = pd.DataFrame(
        [
            {
                "Date": " 02/01/2024 ",
                "Description": "ATM   Withdrawal",
                "Debit": "100.00",
                "Credit": "",
                "Balance": "650.00",
            },
            {
                "Date": "03/01/2024",
                "Description": "Salary",
                "Debit": "",
                "Credit": "5000.00",
                "Balance": "5650.00",
            },
        ]
    )

    combined = parser._finalize_extracted_tables(
        [first_page, second_page],
        ["Date", "Description", "Debit", "Credit", "Balance"],
    )

    assert len(combined) == 4
    assert list(combined["Date"]) == [
        "01/01/2024",
        "02/01/2024",
        "02/01/2024",
        "03/01/2024",
    ]


def test_materialize_selected_tables_attaches_source_provenance():
    module = _load_eosin_pipeline_module()
    parser = module.BankStatementParser.__new__(module.BankStatementParser)
    page_evaluations = [
        {
            "page_index": 4,
            "selected": True,
            "selected_table_index": 2,
            "table_count": 3,
            "raw_headers": ["Date", "Description", "Debit", "Balance"],
            "dataframe": pd.DataFrame(
                [
                    {"Date": "01/01/2024", "Description": "A", "Debit": "1", "Balance": "9"},
                    {"Date": "02/01/2024", "Description": "B", "Debit": "2", "Balance": "7"},
                ]
            ),
        }
    ]

    frames, _, _ = parser._materialize_selected_tables(page_evaluations)

    assert list(frames[0]["__source_page"]) == [4, 4]
    assert list(frames[0]["__source_row"]) == [0, 1]
    assert list(frames[0]["__source_table"]) == [2, 2]


def test_recovers_transaction_table_from_page_missed_by_layout(monkeypatch):
    module = _load_eosin_pipeline_module()
    parser = module.BankStatementParser.__new__(module.BankStatementParser)
    page_images = [module.Image.new("RGB", (100, 100), "white") for _ in range(3)]
    primary = parser._evaluate_page_ocr_result(
        page_idx=0,
        html_content="""
        <table>
          <tr><th>Date</th><th>Description</th><th>Debit</th><th>Balance</th></tr>
          <tr><td>01/01/2024</td><td>FIRST</td><td>10.00</td><td>90.00</td></tr>
        </table>
        """,
        expected_headers=None,
        pass_label="primary",
    )

    def fake_ocr(images):
        assert [page for page, _ in images] == [1, 2]
        return [
            (
                1,
                """
                <table>
                  <tr><td>02/01/2024</td><td>SECOND PAYMENT</td><td>20.00</td><td>70.00</td></tr>
                </table>
                """,
            ),
            (
                2,
                """
                <table>
                  <tr><td>Statement From</td><td>01/01/2024</td></tr>
                  <tr><td>Account Number</td><td>123456789</td></tr>
                </table>
                """,
            ),
        ], {"task_count": 2.0}

    monkeypatch.setattr(parser, "_normalize_ocr_image", lambda image: image)
    monkeypatch.setattr(parser, "_ocr_tables_parallel", fake_ocr)

    evaluations, metrics = parser._recover_missing_layout_pages(
        page_images=page_images,
        table_bboxes=[[0, 0, 100, 100], None, None],
        page_evaluations=[primary],
    )

    assert metrics["task_count"] == 2.0
    assert [item["page_index"] for item in evaluations] == [0, 1, 2]
    assert evaluations[1]["pass_label"] == "layout_miss_full_page"
    assert evaluations[1]["selected"] is True
    assert evaluations[2]["selected"] is False
    assert "layout_miss_not_transaction_table" in evaluations[2]["reasons"]


def test_parse_pdf_uses_full_page_ocr_when_layout_finds_no_tables(monkeypatch):
    module = _load_eosin_pipeline_module()
    parser = module.BankStatementParser.__new__(module.BankStatementParser)
    parser.layout_detector = object()
    parser.layout_mode = "required"
    pages = [module.Image.new("RGB", (100, 100), "white") for _ in range(2)]
    expected = pd.DataFrame([{"Date": "01/01/2024", "Description": "PAYMENT"}])
    calls = []

    monkeypatch.setattr(module, "pdf_to_images_pil", lambda _path, dpi: pages)
    monkeypatch.setattr(parser, "_effective_pdf_dpi", lambda _path: 200)
    monkeypatch.setattr(parser, "_run_layout_detection", lambda _pages: ([[], []], {}))
    monkeypatch.setattr(parser, "_identify_main_tables", lambda _results, _pages: [None, None])

    def fake_full_page_parse(
        pdf_path,
        page_images,
        effective_dpi,
        started_at,
        timings,
        *,
        layout_attempted=False,
    ):
        calls.append((pdf_path, page_images, effective_dpi, started_at, timings, layout_attempted))
        return expected

    monkeypatch.setattr(parser, "_parse_pdf_without_layout", fake_full_page_parse)

    result = parser.parse_pdf("statement.pdf")

    assert result is expected
    assert len(calls) == 1
    assert calls[0][1] == pages
    assert calls[0][2] == 200
    assert "layout_detection" in calls[0][4]
    assert calls[0][5] is True


def test_finalize_extracted_tables_coalesces_schema_drift_before_classification():
    module = _load_eosin_pipeline_module()
    parser = module.BankStatementParser.__new__(module.BankStatementParser)
    first_page = pd.DataFrame(
        [
            {
                "Date(Value Date)": "01-Jul-23(01-Jul-2023)",
                "Date (Value Date)": "",
                "Narration": "FIRST",
                "Debit": "10.00",
                "Balance": "90.00",
                "__source_page": 0,
                "__source_row": 0,
                "__source_table": 0,
            }
        ]
    )
    second_page = pd.DataFrame(
        [
            {
                "Date(Value Date)": "",
                "Date (Value Date)": "02-Jul-23(02-Jul-2023)",
                "Narration": "SECOND",
                "Debit": "20.00",
                "Balance": "70.00",
                "__source_page": 1,
                "__source_row": 0,
                "__source_table": 0,
            }
        ]
    )

    combined = parser._finalize_extracted_tables([first_page, second_page], ["Date (Value Date)"])

    assert len(combined) == 2
    assert parser._last_transaction_reconstruction["selected_source_rows"] == 2
    assert parser._last_transaction_reconstruction["conservation_ok"] is True
    assert not any(str(column).startswith("__source_") for column in combined.columns)


def test_evaluate_page_ocr_result_marks_repeated_garbage_as_suspicious():
    module = _load_eosin_pipeline_module()
    parser = module.BankStatementParser.__new__(module.BankStatementParser)

    html = """
    <table>
      <tr><th>Date</th><th>Description/Narration</th><th>Debit(Dr.)</th><th>Credit(Cr.)</th><th>Balance</th></tr>
      <tr><td>01 Apr 2023</td><td>IMPS313518140119K C KTRADERS.CBNR000000000000000000000000000000000000000000000000000000000000</td><td>1500</td><td>-</td><td>5528</td></tr>
      <tr><td>01 Apr 2023</td><td>IMPS313518140119K C KTRADERS.CBNR000000000000000000000000000000000000000000000000000000000000</td><td>1500</td><td>-</td><td>5528</td></tr>
      <tr><td>01 Apr 2023</td><td>Date Description/Narration Debit(Dr.) Credit(Cr.) Balance</td><td></td><td></td><td></td></tr>
    </table>
    """

    evaluation = parser._evaluate_page_ocr_result(
        page_idx=0,
        html_content=html,
        expected_headers=None,
        pass_label="primary",
    )

    assert evaluation["selected"] is True
    assert evaluation["suspicious"] is True
    assert evaluation["selected_table_index"] == 0
    assert "long_cell" in evaluation["reasons"]
    assert "duplicate_rows" in evaluation["reasons"]
    assert evaluation["row_count"] == 3


def test_select_best_page_ocr_evaluation_prefers_cleaner_retry_pass():
    module = _load_eosin_pipeline_module()
    parser = module.BankStatementParser.__new__(module.BankStatementParser)

    primary = {
        "page_index": 0,
        "pass_label": "primary",
        "selected": True,
        "suspicious": True,
        "quality_score": 15,
        "reasons": ["duplicate_rows", "long_cell"],
        "raw_html": "<table></table>",
        "selected_table_index": 0,
        "table_count": 1,
        "row_count": 4,
        "columns": ["Date", "Description", "Debit", "Credit", "Balance"],
        "dataframe": pd.DataFrame([{"Date": "01/01/2024"}]),
        "raw_headers": ["Date", "Description", "Debit", "Credit", "Balance"],
    }
    retry = {
        "page_index": 0,
        "pass_label": "retry_high_dpi",
        "selected": True,
        "suspicious": False,
        "quality_score": 78,
        "reasons": [],
        "raw_html": "<table></table>",
        "selected_table_index": 0,
        "table_count": 1,
        "row_count": 3,
        "columns": ["Date", "Description", "Debit", "Credit", "Balance"],
        "dataframe": pd.DataFrame([{"Date": "01/01/2024"}]),
        "raw_headers": ["Date", "Description", "Debit", "Credit", "Balance"],
    }

    selected = parser._select_best_page_ocr_evaluation([primary, retry])

    assert selected["pass_label"] == "retry_high_dpi"
    assert selected["quality_score"] == 78


def test_evaluate_page_ocr_result_flags_column_shift_without_repairing_values():
    module = _load_eosin_pipeline_module()
    parser = module.BankStatementParser.__new__(module.BankStatementParser)

    html = """
    <table>
      <thead>
        <tr>
          <th>Tran Date</th><th>Particulars</th><th>Debit</th><th>Credit</th><th>Balance</th><th>Init. Br</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>05-02-2020</td><td></td><td></td><td>49750.00</td><td></td><td>49771.25</td></tr>
        <tr><td>05-02-2020</td><td>BY TRANSFER</td><td></td><td></td><td>49671.25</td><td>227</td></tr>
      </tbody>
    </table>
    """

    evaluation = parser._evaluate_page_ocr_result(
        page_idx=0,
        html_content=html,
        expected_headers=["Tran Date", "Particulars", "Debit", "Credit", "Balance", "Init. Br"],
        pass_label="primary",
    )

    dataframe = evaluation["dataframe"]
    assert isinstance(dataframe, pd.DataFrame)
    assert dataframe.iloc[0]["Init. Br"] == "49771.25"
    assert "money_in_non_amount_columns" in evaluation["reasons"]
    assert "dated_rows_missing_text" in evaluation["reasons"]
    assert evaluation["suspicious"] is True


def test_finalize_extracted_tables_excludes_summary_but_keeps_dated_ocr_row():
    module = _load_eosin_pipeline_module()
    parser = module.BankStatementParser.__new__(module.BankStatementParser)

    dirty = pd.DataFrame(
        [
            {
                "Date": "01/01/2024",
                "Description": "Opening balance",
                "Debit": "",
                "Credit": "",
                "Balance": "1000.00",
            },
            {
                "Date": "02/01/2024",
                "Description": "IMPS313518140119K C KTRADERS.CBNR000000000000000000000000000000000000000000000000000000000000",
                "Debit": "1500.00",
                "Credit": "",
                "Balance": "5528.00",
            },
            {
                "Date": "03/01/2024",
                "Description": "Salary",
                "Debit": "",
                "Credit": "5000.00",
                "Balance": "6528.00",
            },
            {
                "Date": "04/01/2024",
                "Description": "",
                "Debit": "",
                "Credit": "",
                "Balance": "",
            },
        ]
    )

    cleaned = parser._finalize_extracted_tables(
        [dirty],
        ["Date", "Description", "Debit", "Credit", "Balance"],
    )

    assert list(cleaned["Date"]) == ["02/01/2024", "03/01/2024"]


def test_finalize_extracted_tables_keeps_long_legitimate_descriptions():
    module = _load_eosin_pipeline_module()
    parser = module.BankStatementParser.__new__(module.BankStatementParser)

    long_description = (
        "IMPS/57GS5G3KP1353LW5BINDASS MEDIA ENTERTAINMENT AND WE/"
        "AMAZON SELLER SERVIC/XXX6004/RRN: 201810089747/HSBC BANK"
    )
    dirty = pd.DataFrame(
        [
            {
                "TXN DATE": "18-Jan-2022",
                "DESCRIPTION": long_description,
                "DEBITS": "0",
                "CREDITS": "101.16",
                "BALANCE": "47,989.75",
            }
        ]
    )

    cleaned = parser._finalize_extracted_tables(
        [dirty],
        ["TXN DATE", "DESCRIPTION", "DEBITS", "CREDITS", "BALANCE"],
    )

    assert len(cleaned) == 1
    assert cleaned.iloc[0]["DESCRIPTION"] == long_description


def test_finalize_extracted_tables_merges_semantic_duplicate_columns():
    module = _load_eosin_pipeline_module()
    parser = module.BankStatementParser.__new__(module.BankStatementParser)

    dirty = pd.DataFrame(
        [
            {
                "Date": "01/01/2024",
                "Description": "UPI",
                "Chq./Ref. No.": "",
                "Chq/Ref. No.": "ABC123",
                "Debit": "10.00",
                "Debt/(Dr.)": "",
                "Credit": "",
                "Balance": "990.00",
            },
            {
                "Date": "02/01/2024",
                "Description": "Salary",
                "Chq./Ref. No.": "XYZ789",
                "Chq/Ref. No.": "",
                "Debit": "",
                "Debt/(Dr.)": "20.00",
                "Credit": "1000.00",
                "Balance": "1990.00",
            },
        ]
    )

    cleaned = parser._finalize_extracted_tables(
        [dirty],
        ["Date", "Description", "Chq./Ref. No.", "Debit", "Credit", "Balance"],
    )

    assert "Chq/Ref. No." not in cleaned.columns
    assert "Debt/(Dr.)" not in cleaned.columns
    assert list(cleaned["Chq./Ref. No."]) == ["ABC123", "XYZ789"]
    assert list(cleaned["Debit"]) == ["10.00", "20.00"]


def test_build_document_quality_summary_flags_low_confidence():
    module = _load_eosin_pipeline_module()
    parser = module.BankStatementParser.__new__(module.BankStatementParser)

    df = pd.DataFrame(
        [
            {"Date": "01/01/2024", "Description": "Salary", "Debit": "", "Credit": "5000.00", "Balance": "5000.00"},
            {"Date": "01/01/2024", "Description": "Salary", "Debit": "", "Credit": "5000.00", "Balance": "5000.00"},
        ]
    )
    page_evaluations = [
        {"page_index": 0, "pass_label": "primary", "selected": True, "suspicious": True, "quality_score": 18, "reasons": ["duplicate_rows"], "retried": True},
        {"page_index": 1, "pass_label": "retry_high_dpi", "selected": True, "suspicious": False, "quality_score": 64, "reasons": [], "retried": True},
    ]

    summary = parser._build_document_quality_summary(df, page_evaluations)

    assert summary["low_confidence"] is True
    assert "suspicious_pages_present" in summary["reasons"]
    assert "duplicate_rows_remaining" in summary["reasons"]
    assert summary["retried_pages"] == [1, 2]
