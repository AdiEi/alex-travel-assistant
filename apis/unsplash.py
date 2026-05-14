import os
import httpx
from dotenv import load_dotenv

load_dotenv()

BASE_URL = "https://api.unsplash.com"


class UnsplashClient:
    def __init__(self):
        self.access_key = os.getenv("UNSPLASH_ACCESS_KEY")

    @property
    def is_configured(self) -> bool:
        return bool(self.access_key)

    def get_photo_url(self, destination: str) -> str | None:
        if not self.access_key:
            return None
        with httpx.Client(timeout=5.0) as client:
            response = client.get(
                f"{BASE_URL}/search/photos",
                params={"query": destination, "per_page": 1, "orientation": "landscape"},
                headers={"Authorization": f"Client-ID {self.access_key}"},
            )
            response.raise_for_status()
            results = response.json().get("results", [])
            if results:
                return results[0]["urls"]["regular"]
        return None
