"""
scripts/download_sparkles.py — SPARKLES Dataset Downloader (NASA Ames / Zenodo)
==============================================================================
Downloads SPARKLES FITS dataset samples from Zenodo (Record: 10935280)
using public Zenodo REST API into data/raw/.
"""

import logging
import os
import sys
from pathlib import Path

import requests

from spherex_streak.config import DATA_DIR, ZENODO_ACCESS_TOKEN, ZENODO_RECORD_ID

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def download_sparkles_dataset(
    record_id: str = ZENODO_RECORD_ID,
    dest_dir: Path = DATA_DIR,
    access_token: str | None = ZENODO_ACCESS_TOKEN,
) -> bool:
    """
    Downloads SPARKLES dataset from Zenodo REST API.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    api_url = f"https://zenodo.org/api/records/{record_id}"

    headers = {}
    if access_token and "your_zenodo_token" not in access_token:
        headers["Authorization"] = f"Bearer {access_token}"

    logger.info("Fetching SPARKLES record metadata from Zenodo (Record ID: %s)...", record_id)

    try:
        resp = requests.get(api_url, headers=headers, timeout=15)
        if resp.status_code == 200:
            data = resp.json()
            files = data.get("files", [])
            logger.info("Found %d files associated with SPARKLES record.", len(files))

            for f_info in files[:5]:  # Download sample files
                f_name = f_info.get("key", f_info.get("filename", "sparkles_sample.fits"))
                download_link = f_info.get("links", {}).get("self", "")
                target_file = dest_dir / f_name

                if target_file.exists():
                    logger.info("File already exists: %s", target_file.name)
                    continue

                if download_link:
                    logger.info("Downloading: %s...", f_name)
                    f_resp = requests.get(download_link, stream=True, timeout=30)
                    if f_resp.status_code == 200:
                        with open(target_file, "wb") as out:
                            for chunk in f_resp.iter_content(chunk_size=65536):
                                out.write(chunk)
                        logger.info("Saved: %s", target_file)
            return True
        else:
            logger.warning("Zenodo API responded with status %d: %s", resp.status_code, resp.text[:100])
            logger.info("Defaulting to built-in synthetic SPARKLES generator for testing.")
            return False

    except Exception as exc:
        logger.warning("Connection to Zenodo failed (%s). Standalone synthetic mode available.", exc)
        return False


if __name__ == "__main__":
    download_sparkles_dataset()
