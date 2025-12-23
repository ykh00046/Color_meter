import argparse
import json
import mimetypes
import os
import time
from pathlib import Path
from urllib import request


def _encode_multipart(fields, files):
    boundary = f"----Boundary{int(time.time() * 1000)}"
    lines = []

    for name, value in fields.items():
        lines.append(f"--{boundary}")
        lines.append(f'Content-Disposition: form-data; name="{name}"')
        lines.append("")
        lines.append(str(value))

    for name, filename, content, content_type in files:
        lines.append(f"--{boundary}")
        lines.append(f'Content-Disposition: form-data; name="{name}"; filename="{filename}"')
        lines.append(f"Content-Type: {content_type}")
        lines.append("")
        lines.append(content)

    lines.append(f"--{boundary}--")
    lines.append("")

    body = b""
    for line in lines:
        if isinstance(line, bytes):
            body += line + b"\r\n"
        else:
            body += line.encode("utf-8") + b"\r\n"

    return boundary, body


def _post_compare(url, sku, ref_path, test_paths):
    files = []
    ref_bytes = Path(ref_path).read_bytes()
    ref_ct = mimetypes.guess_type(ref_path)[0] or "application/octet-stream"
    files.append(("reference_file", Path(ref_path).name, ref_bytes, ref_ct))

    for p in test_paths:
        p = Path(p)
        data = p.read_bytes()
        ct = mimetypes.guess_type(str(p))[0] or "application/octet-stream"
        files.append(("test_files", p.name, data, ct))

    fields = {"sku": sku}
    boundary, body = _encode_multipart(fields, files)

    req = request.Request(url, data=body)
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    req.add_header("Content-Length", str(len(body)))

    with request.urlopen(req) as resp:
        return resp.read()


def main():
    parser = argparse.ArgumentParser(description="Run /compare tests for ink sample folders.")
    parser.add_argument("--base", default="data/samples", help="Base samples directory")
    parser.add_argument("--url", default="http://localhost:8000/compare", help="Compare endpoint URL")
    parser.add_argument("--out", default="results/compare_json", help="Output directory for JSON")
    args = parser.parse_args()

    base = Path(args.base)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = {
        "INK1": "SKU_INK1",
        "INK2": "SKU_INK2",
        "INK3": "SKU_INK3",
    }

    for folder, sku in mapping.items():
        folder_path = base / folder
        if not folder_path.exists():
            print(f"Skip {folder}: not found")
            continue

        files = sorted([p for p in folder_path.iterdir() if p.is_file()])
        if len(files) < 2:
            print(f"Skip {folder}: need at least 2 images")
            continue

        ref = files[0]
        tests = files[1:]
        print(f"Running {folder} with {sku}: ref={ref.name}, tests={len(tests)}")

        response = _post_compare(args.url, sku, ref, tests)
        out_name = f"compare_{folder}_{ref.stem}_{int(time.time())}.json"
        (out_dir / out_name).write_bytes(response)
        print(f"Saved {out_name}")


if __name__ == "__main__":
    main()
