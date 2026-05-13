import json

import requests


API_BASE = "http://127.0.0.1:19000"
PROMPT = "What is an AI semantic layer for Select AI?"
TOP_K = 2
RERANKER_SCORE = 0.4


def call_search(return_mode: str) -> None:
    payload = {
        "prompt": PROMPT,
        "top_k": TOP_K,
        "reranker_score": RERANKER_SCORE,
        "return_mode": return_mode,
        "device": "cuda",
        "document_type": "txt",
    }
    response = requests.post(
        f"{API_BASE}/search",
        json=payload,
        timeout=600,
    )
    print(f"mode={return_mode}, status_code={response.status_code}")
    print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    print("-" * 80)


def main() -> None:
    for mode in ("full", "split", "title"):
        call_search(mode)


if __name__ == "__main__":
    main()
