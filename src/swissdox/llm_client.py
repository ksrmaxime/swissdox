from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


@dataclass
class LlamaServerClient:
    base_url: str = "http://127.0.0.1:8080"
    timeout_s: int = 120
    retries_total: int = 5
    backoff_factor: float = 0.5
    pool_connections: int = 50
    pool_maxsize: int = 50

    def __post_init__(self) -> None:
        self.chat_url = f"{self.base_url}/v1/chat/completions"
        self.models_url = f"{self.base_url}/v1/models"

        self.session = requests.Session()
        retry = Retry(
            total=self.retries_total,
            backoff_factor=self.backoff_factor,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "POST"],
            raise_on_status=False,
        )
        adapter = HTTPAdapter(max_retries=retry, pool_connections=self.pool_connections, pool_maxsize=self.pool_maxsize)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    def get_model_id(self) -> str:
        try:
            r = self.session.get(self.models_url, timeout=self.timeout_s)
            r.raise_for_status()
            data = r.json()
            models = data.get("data", [])
            if models and isinstance(models, list):
                mid = models[0].get("id")
                if mid:
                    return str(mid)
        except Exception:
            pass
        return "local-model"

    def chat_completion(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.0,
        top_p: float = 1.0,
        max_tokens: int = 180,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        r = self.session.post(self.chat_url, json=payload, timeout=self.timeout_s)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"]
