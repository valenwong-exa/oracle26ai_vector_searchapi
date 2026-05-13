import argparse
import json

import requests


DEFAULT_PROMPT = "how to Tuning Network Performance for Data Guard"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test vector search API by port only. 19000=Qwen, 19001=BGE."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=19000,
        help="API port. 19000=Qwen, 19001=BGE.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="API host. Defaults to 127.0.0.1.",
    )
    parser.add_argument(
        "--prompt",
        default=DEFAULT_PROMPT,
        help="Search prompt.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Top K final results.",
    )
    parser.add_argument(
        "--fetch-k",
        type=int,
        default=0,
        help="Optional fetch K before rerank. 0 means server default.",
    )
    parser.add_argument(
        "--reranker-score",
        type=float,
        default=0.4,
        help="Minimum reranker score to keep a hit.",
    )
    parser.add_argument(
        "--return-mode",
        choices=("full", "split", "title"),
        default="full",
        help="Response text mode.",
    )
    parser.add_argument(
        "--document-type",
        default="",
        help="Optional document type filter.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Runtime device.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    api_base = f"http://{args.host}:{args.port}"
    search_url = f"{api_base}/search"

    payload = {
        "prompt": args.prompt,
        "top_k": args.top_k,
        "reranker_score": args.reranker_score,
        "return_mode": args.return_mode,
        "device": args.device,
    }
    if args.fetch_k > 0:
        payload["fetch_k"] = args.fetch_k
    if args.document_type:
        payload["document_type"] = args.document_type

    print(f"Searching: {search_url}")
    print(f"Port meaning: {'Qwen' if args.port == 19000 else 'BGE' if args.port == 19001 else 'custom'}")
    print("payload:")
    print(json.dumps(payload, ensure_ascii=False, indent=2))

    response = requests.post(
        search_url,
        json=payload,
        timeout=600,
    )

    print("status_code:", response.status_code)
    try:
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    except ValueError:
        print(response.text)


if __name__ == "__main__":
    main()
