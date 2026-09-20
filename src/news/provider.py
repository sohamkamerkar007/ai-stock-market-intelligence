import hashlib
from datetime import UTC, datetime

import requests
from tenacity import retry, stop_after_attempt, wait_exponential


class NewsDataProvider:
    endpoint = "https://newsdata.io/api/1/latest"

    def __init__(self, api_key: str):
        self.api_key = api_key

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20), reraise=True)
    def latest(self, query: str) -> list[dict]:
        response = requests.get(
            self.endpoint,
            params={
                "apikey": self.api_key,
                "q": query,
                "country": "in",
                "language": "en",
                "category": "business",
            },
            timeout=20,
        )
        if response.status_code == 429:
            raise RuntimeError("News provider rate limit reached")
        response.raise_for_status()
        rows = []
        for item in response.json().get("results", []):
            url = item.get("link") or ""
            title = item.get("title") or ""
            external_id = (
                item.get("article_id") or hashlib.sha256(f"{url}|{title}".encode()).hexdigest()
            )
            stamp = item.get("pubDate")
            try:
                published = datetime.fromisoformat(stamp.replace("Z", "+00:00"))
            except (AttributeError, ValueError):
                published = datetime.now(UTC)
            rows.append(
                {
                    "external_id": external_id,
                    "title": title,
                    "description": item.get("description"),
                    "source": item.get("source_name") or item.get("source_id") or "Unknown",
                    "url": url,
                    "published_at": published,
                    "symbols": [],
                }
            )
        return rows
