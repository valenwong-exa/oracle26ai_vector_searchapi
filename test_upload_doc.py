import argparse
import json
from pathlib import Path

import requests


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_FILE = BASE_DIR / "test.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Test document upload API by port only. 19000=Qwen, 19001=BGE."
    )
    parser.add_argument(
        "file",
        nargs="?",
        default=str(DEFAULT_FILE),
        help="Path to the file to upload. Defaults to test.txt in current directory.",
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
        "--chunk-size-tokens",
        type=int,
        default=500,
        help="Chunk size in tokens.",
    )
    parser.add_argument(
        "--chunk-overlap-tokens",
        type=int,
        default=50,
        help="Chunk overlap in tokens.",
    )
    parser.add_argument(
        "--source-file",
        default="test-client",
        help="Source file tag written to Oracle.",
    )
    parser.add_argument(
        "--document-type",
        default="",
        help="Optional document type. Leave empty to let server infer it.",
    )
    parser.add_argument(
        "--instruction",
        default="",
        help="Optional embedding instruction override.",
    )
    parser.add_argument(
        "--device",
        choices=("auto", "cuda", "cpu"),
        default="auto",
        help="Runtime device.",
    )
    parser.add_argument(
        "--use-deepseek-summary",
        choices=("true", "false"),
        default="true",
        help="Whether to append DeepSeek summary before split.",
    )
    return parser.parse_args()


def detect_content_type(file_path: Path) -> str:
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return "application/pdf"
    if suffix == ".txt":
        return "text/plain"
    return "application/octet-stream"


def main() -> None:
    args = parse_args()
    target_file = Path(args.file).expanduser().resolve()
    if not target_file.exists():
        raise FileNotFoundError(f"Test file not found: {target_file}")

    api_base = f"http://{args.host}:{args.port}"
    upload_url = f"{api_base}/documents/upload"
    content_type = detect_content_type(target_file)

    data = {
        "chunk_size_tokens": str(args.chunk_size_tokens),
        "chunk_overlap_tokens": str(args.chunk_overlap_tokens),
        "source_file": args.source_file,
        "device": args.device,
        "use_deepseek_summary": args.use_deepseek_summary,
    }
    if args.document_type:
        data["document_type"] = args.document_type
    if args.instruction:
        data["instruction"] = args.instruction

    print(f"Uploading to: {upload_url}")
    print(f"File: {target_file}")
    print(f"Port meaning: {'Qwen' if args.port == 19000 else 'BGE' if args.port == 19001 else 'custom'}")

    with target_file.open("rb") as handle:
        files = {
            "file": (target_file.name, handle, content_type),
        }
        response = requests.post(
            upload_url,
            files=files,
            data=data,
            timeout=600,
        )

    print("status_code:", response.status_code)
    try:
        print(json.dumps(response.json(), ensure_ascii=False, indent=2))
    except ValueError:
        print(response.text)


if __name__ == "__main__":
    main()
