import logging
import os
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import requests
from dotenv import load_dotenv
from eliot import log_call

from app.thesmos.exceptions import ProviderFailedToDig
from app.thesmos.providers.base import Provider

load_dotenv()

logger = logging.getLogger(__name__)


def _payload_to_dataframe(payload: dict[str, Any]) -> pd.DataFrame:
    columns = payload.get("columns", [])
    rows = payload.get("rows", [])
    if columns:
        return pd.DataFrame(rows, columns=columns)
    return pd.DataFrame(rows)


class EosinPDFProvider(Provider):
    """Extract tables through the remote GLM bank-parser service."""

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: int = 600,
        session: requests.Session | None = None,
        bearer_token: str | None = None,
        max_attempts: int = 3,
        retry_backoff_seconds: float = 1.0,
    ) -> None:
        raw_url = (
            base_url
            or os.getenv("EOSIN_PARSER_BASE_URL")
            or os.getenv("EOSIN_MODAL_ENDPOINT")
        )
        if not raw_url or not raw_url.strip():
            raise ValueError(
                "Missing parser base URL. Set EOSIN_PARSER_BASE_URL or EOSIN_MODAL_ENDPOINT."
            )
        raw_url = raw_url.strip().rstrip("/")
        if not raw_url.startswith(("http://", "https://")):
            raw_url = f"https://{raw_url}"

        parsed = urlparse(raw_url)
        if parsed.port is None and parsed.netloc.endswith(".modal.run"):
            self.base_url = raw_url
        elif parsed.port is None:
            port = os.getenv("BANK_PARSER_PORT", "8090")
            raw_url = f"{parsed.scheme}://{parsed.hostname}:{port}{parsed.path}"
            self.base_url = raw_url
        else:
            self.base_url = raw_url

        self.timeout = timeout
        self.session = session if session is not None else requests.Session()
        self.bearer_token = (
            bearer_token
            or os.getenv("EOSIN_PARSER_BEARER_TOKEN")
            or os.getenv("EOSIN_MODAL_BEARER_TOKEN")
        )
        self.modal_proxy_auth_token_id = (
            os.getenv("MODAL_PROXY_AUTH_TOKEN_ID")
            or os.getenv("MODAL_PERSONAL_PROXY_AUTH_TOKEN_ID")
            or os.getenv("EOSIN_MODAL_PROXY_AUTH_TOKEN_ID")
            or os.getenv("EOSIN_PARSER_MODAL_KEY")
        )
        self.modal_proxy_auth_token_secret = (
            os.getenv("MODAL_PROXY_AUTH_TOKEN_SECRET")
            or os.getenv("MODAL_PERSONAL_PROXY_AUTH_TOKEN_SECRET")
            or os.getenv("EOSIN_MODAL_PROXY_AUTH_TOKEN_SECRET")
            or os.getenv("EOSIN_PARSER_MODAL_SECRET")
        )
        self.require_bearer = (
            os.getenv("EOSIN_PARSER_REQUIRE_BEARER", "false").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if self.require_bearer and not self.bearer_token:
            raise ValueError(
                "Bearer auth is required. Set EOSIN_PARSER_BEARER_TOKEN or EOSIN_MODAL_BEARER_TOKEN."
            )
        self.require_proxy_auth = (
            os.getenv("EOSIN_PARSER_REQUIRE_PROXY_AUTH", "false").strip().lower()
            in {"1", "true", "yes", "on"}
        )
        if self.require_proxy_auth and (
            not self.modal_proxy_auth_token_id or not self.modal_proxy_auth_token_secret
        ):
            raise ValueError(
                "Modal proxy auth is required. Set MODAL_PROXY_AUTH_TOKEN_ID and MODAL_PROXY_AUTH_TOKEN_SECRET."
            )
        self.max_attempts = max(1, int(max_attempts))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))

    def _request_headers(self) -> dict[str, str]:
        headers: dict[str, str] = {}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        if self.modal_proxy_auth_token_id and self.modal_proxy_auth_token_secret:
            headers["Modal-Key"] = self.modal_proxy_auth_token_id
            headers["Modal-Secret"] = self.modal_proxy_auth_token_secret
        return headers

    @log_call(include_args=[], include_result=False)
    def extract(self, file_path: Path) -> pd.DataFrame:
        pdf_path = Path(file_path)
        logger.info("EosinPDFProvider: sending %s to %s", pdf_path.name, self.base_url)

        last_exception: Exception | None = None

        for attempt in range(1, self.max_attempts + 1):
            try:
                with pdf_path.open("rb") as handle:
                    response = self.session.post(
                        f"{self.base_url}/parse/bank-statement",
                        files={"file": (pdf_path.name, handle, "application/pdf")},
                        headers=self._request_headers(),
                        timeout=self.timeout,
                    )

                response.raise_for_status()
                payload = response.json()
                dataframe = _payload_to_dataframe(payload)
                if dataframe.empty:
                    raise ProviderFailedToDig(
                        f"EosinPDFProvider: No table data extracted from {pdf_path.name}"
                    )
                return dataframe
            except ProviderFailedToDig:
                raise
            except Exception as exc:
                last_exception = exc
                is_last_attempt = attempt >= self.max_attempts
                logger.warning(
                    "EosinPDFProvider attempt %s/%s failed for %s: %s",
                    attempt,
                    self.max_attempts,
                    pdf_path.name,
                    exc,
                    exc_info=is_last_attempt,
                )
                if is_last_attempt:
                    break
                time.sleep(self.retry_backoff_seconds * attempt)

        assert last_exception is not None
        raise ProviderFailedToDig(
            f"Eosin (remote GLM parser) failed to extract {pdf_path.name}: {last_exception}"
        ) from last_exception
