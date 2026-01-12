# src/swissdox/api.py
from __future__ import annotations

import time
from dataclasses import dataclass
from io import BytesIO
from typing import Any, Dict, Optional

import pandas as pd
import requests
import yaml


@dataclass
class SwissdoxClient:
    api_key: str
    api_secret: str
    base_url: str = "https://swissdox.linguistik.uzh.ch/api"
    timeout: int = 120

    def __post_init__(self) -> None:
        self.query_url = f"{self.base_url}/query"
        self.status_url = f"{self.base_url}/status"
        self.headers = {"X-API-Key": self.api_key, "X-API-Secret": self.api_secret}
        self.session = requests.Session()

    def submit_query(self, yaml_payload: Dict[str, Any], *, name: str, comment: str, expiration_date: str, test: bool = False) -> str:
        yaml_query = yaml.safe_dump(yaml_payload, sort_keys=False, allow_unicode=True)
        data = {
            "query": yaml_query,
            "name": name,
            "comment": comment,
            "expirationDate": expiration_date,
        }
        if test:
            data["test"] = "1"

        r = self.session.post(self.query_url, headers=self.headers, data=data, timeout=self.timeout)
        if r.status_code >= 400:
            # Keep it short but useful for debugging
            raise RuntimeError(f"Swissdox /query failed ({r.status_code}): {r.text[:2000]}")
        r.raise_for_status()

        resp = r.json()
        if resp.get("result") != "ok":
            raise RuntimeError(f"Swissdox non-ok response: {resp}")

        query_id = resp.get("queryId") or resp.get("id")
        if not query_id:
            raise RuntimeError(f"Swissdox missing queryId: {resp}")
        return str(query_id)

    def wait_for_download_url(self, query_id: str, *, max_wait_s: int = 25 * 60, poll_every_s: int = 5) -> str:
        deadline = time.time() + max_wait_s
        while time.time() < deadline:
            rs = self.session.get(self.status_url, headers=self.headers, timeout=self.timeout)
            rs.raise_for_status()
            status_list = rs.json()

            job_info = next((j for j in status_list if str(j.get("id")) == str(query_id)), None)
            if not job_info:
                time.sleep(poll_every_s)
                continue

            if job_info.get("error"):
                raise RuntimeError(f"Swissdox job error: {job_info['error']}")

            download_url = job_info.get("downloadUrl")
            if download_url:
                return self._normalize_download_url(download_url)

            time.sleep(poll_every_s)

        raise TimeoutError(f"Swissdox: no downloadUrl after {max_wait_s}s (query_id={query_id})")

    def download_tsv_xz(self, download_url: str) -> pd.DataFrame:
        r = self.session.get(download_url, headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        return pd.read_csv(BytesIO(r.content), sep="\t", compression="xz")

    def _normalize_download_url(self, download_url: str) -> str:
        if download_url.startswith("http"):
            return download_url
        if download_url.startswith("/"):
            return f"{self.base_url}{download_url}"
        return f"{self.base_url}/download/{download_url}"
