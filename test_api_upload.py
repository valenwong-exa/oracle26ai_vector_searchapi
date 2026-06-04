import sys
from pathlib import Path

import requests


API_BASE = "http://127.0.0.1:19000"
DEFAULT_FILE = Path(__file__).resolve().parent / "requirements.txt"


def main() -> None:
    target_file = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else DEFAULT_FILE
    if not target_file.exists():
        raise FileNotFoundError(f"Test file not found: {target_file}")

    content_type = "application/pdf" if target_file.suffix.lower() == ".pdf" else "text/plain"
    with target_file.open("rb") as handle:
        files = {
            "file": (target_file.name, handle, content_type),
        }
        data = {
            "chunk_size_tokens": "500",
            "chunk_overlap_tokens": "50",
            "source_file": "blog",
            "device": "cpu",
        }
        response = requests.post(
            f"{API_BASE}/documents/upload",
            files=files,
            data=data,
            timeout=3600,
        )

    print("status_code:", response.status_code)
    print(response.text)


if __name__ == "__main__":
    main()
