import hashlib
import json
import re
import time
from datetime import datetime
from pathlib import Path

import requests
from bs4 import BeautifulSoup
from loguru import logger

_CACHE_BASE = Path(__file__).parent.parent / "cache"
_UMP_URL = "https://umpscorecards.com/umpires/"


def _cache_path(date_str: str) -> Path:
    d = _CACHE_BASE / date_str
    d.mkdir(parents=True, exist_ok=True)
    return d / "umpire_data.json"


def _neutral() -> dict:
    return {
        "umpire_name":        None,
        "favor_score":        0.0,
        "called_k_rate_diff": 0.0,
        "zone_score":         0.0,
    }


def get_umpire_tendencies(umpire_name: str, date_str: str = None) -> dict:
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    if not umpire_name:
        return _neutral()

    cp = _cache_path(date_str)
    cached_all = {}
    if cp.exists():
        try:
            cached_all = json.loads(cp.read_text(encoding="utf-8"))
            if umpire_name in cached_all:
                return cached_all[umpire_name]
        except Exception:
            pass

    result = _fetch_umpire(umpire_name)
    cached_all[umpire_name] = result
    try:
        cp.write_text(json.dumps(cached_all), encoding="utf-8")
    except Exception:
        pass
    return result


def _fetch_umpire(umpire_name: str) -> dict:
    try:
        # First try the individual umpire page
        slug = umpire_name.lower().replace(" ", "-").replace(".", "")
        url = f"{_UMP_URL}{slug}/"
        time.sleep(0.5)
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        r = requests.get(url, headers=headers, timeout=15)
        if r.status_code == 200:
            result = _parse_umpire_page(r.text, umpire_name)
            if result:
                return result
    except Exception as e:
        logger.warning(f"Umpire page fetch failed for {umpire_name}: {e}")

    # Try the main umpires listing and scan for this umpire
    try:
        time.sleep(0.5)
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        r = requests.get(_UMP_URL, headers=headers, timeout=15)
        if r.status_code != 200:
            logger.warning(f"UmpScorecards listing returned {r.status_code}")
            return _neutral()
        result = _parse_listing_for_umpire(r.text, umpire_name)
        if result:
            return result
    except Exception as e:
        logger.warning(f"UmpScorecards listing failed: {e}")

    logger.warning(f"Umpire data not found for {umpire_name} — using neutral values")
    neutral = _neutral()
    neutral["umpire_name"] = umpire_name
    return neutral


def _parse_umpire_page(html: str, umpire_name: str) -> dict | None:
    try:
        soup = BeautifulSoup(html, "lxml")
        result = _neutral()
        result["umpire_name"] = umpire_name

        # Look for favor score / zone score in stat cards or tables
        text = soup.get_text(separator=" ", strip=True)

        # Pattern: look for numeric values near keywords
        favor = _extract_number_near(text, ["favor score", "favor_score", "favor"])
        k_rate = _extract_number_near(text, ["called strike", "k rate", "k_rate"])

        if favor is not None:
            result["favor_score"] = favor
        if k_rate is not None:
            result["called_k_rate_diff"] = k_rate

        # zone_score: composite — negative if batter-friendly
        result["zone_score"] = round(-result["favor_score"] * 0.5 + result["called_k_rate_diff"] * 0.3, 4)
        return result
    except Exception:
        return None


def _parse_listing_for_umpire(html: str, umpire_name: str) -> dict | None:
    try:
        soup = BeautifulSoup(html, "lxml")
        # Try JSON embedded in script tags first
        for script in soup.find_all("script"):
            text = script.string or ""
            if umpire_name.split()[-1] in text and "favor" in text.lower():
                numbers = re.findall(r"-?\d+\.\d+", text)
                if numbers:
                    favor = float(numbers[0])
                    neutral = _neutral()
                    neutral["umpire_name"] = umpire_name
                    neutral["favor_score"] = favor
                    neutral["zone_score"] = round(-favor * 0.5, 4)
                    return neutral

        # Look for the umpire in a table row
        name_lower = umpire_name.lower()
        for row in soup.find_all("tr"):
            row_text = row.get_text(separator=" ").lower()
            if name_lower.split()[-1] in row_text:
                cells = row.find_all("td")
                if len(cells) >= 3:
                    try:
                        favor = float(cells[2].get_text(strip=True))
                        neutral = _neutral()
                        neutral["umpire_name"] = umpire_name
                        neutral["favor_score"] = favor
                        neutral["zone_score"] = round(-favor * 0.5, 4)
                        return neutral
                    except (ValueError, IndexError):
                        pass
        return None
    except Exception:
        return None


def _extract_number_near(text: str, keywords: list[str]) -> float | None:
    text_lower = text.lower()
    for kw in keywords:
        idx = text_lower.find(kw)
        if idx == -1:
            continue
        snippet = text[idx:idx + 100]
        nums = re.findall(r"-?\d+\.?\d*", snippet)
        if nums:
            try:
                return float(nums[0])
            except ValueError:
                continue
    return None


if __name__ == "__main__":
    today = datetime.now().strftime("%Y-%m-%d")
    logger.info("Testing umpire.py")
    # Test with a known umpire
    for name in ["Angel Hernandez", "Joe West", "Dan Bellino"]:
        result = get_umpire_tendencies(name, date_str=today)
        logger.info(f"{name}: {result}")
