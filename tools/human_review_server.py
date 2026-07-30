#!/usr/bin/env python3
"""Serve the owner-review UI on localhost and persist scorer-compatible CSV."""

from __future__ import annotations

import argparse
import json
import mimetypes
import sys
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from human_review_app import load_review_state, save_annotations  # noqa: E402


ASSETS = ROOT / "review"
FULL_CAMPAIGNS = {
    "memgym": {
        "label": "Réponses MemGym",
        "pack": ROOT
        / ".cache"
        / "human-calibration"
        / "memgym-judge"
        / "review-pack.jsonl",
        "output": ROOT
        / ".cache"
        / "human-calibration"
        / "memgym-judge"
        / "annotator-owner.csv",
    },
    "context-shards": {
        "label": "Context Shards",
        "pack": ROOT
        / ".cache"
        / "human-calibration"
        / "context-shards-24"
        / "review-pack.jsonl",
        "output": ROOT
        / ".cache"
        / "human-calibration"
        / "context-shards-24"
        / "annotator-owner.csv",
    },
}
MINI_ROOT = ROOT / ".cache" / "human-calibration" / "owner-mini-10"
MINI_CAMPAIGNS = {
    "memgym": {
        "label": "Réponses MemGym · sélection courte",
        "pack": MINI_ROOT / "memgym" / "review-pack.jsonl",
        "output": MINI_ROOT / "memgym" / "annotator-owner.csv",
    },
    "context-shards": {
        "label": "Context Shards · sélection courte",
        "pack": MINI_ROOT / "context-shards" / "review-pack.jsonl",
        "output": MINI_ROOT / "context-shards" / "annotator-owner.csv",
    },
}
CAMPAIGNS = MINI_CAMPAIGNS


def campaign_state(campaign_id: str) -> dict[str, Any]:
    campaign = CAMPAIGNS[campaign_id]
    return load_review_state(
        campaign_id=campaign_id,
        label=str(campaign["label"]),
        pack_path=Path(campaign["pack"]),
        output_path=Path(campaign["output"]),
    )


class ReviewHandler(BaseHTTPRequestHandler):
    server_version = "AgentMemoryReview/1"

    def send_json(
        self,
        payload: dict[str, Any] | list[dict[str, Any]],
        status: HTTPStatus = HTTPStatus.OK,
    ) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def send_error_json(
        self,
        message: str,
        status: HTTPStatus,
    ) -> None:
        self.send_json({"error": message}, status)

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/state":
            try:
                self.send_json(
                    [campaign_state(campaign_id) for campaign_id in CAMPAIGNS]
                )
            except (OSError, ValueError, KeyError) as error:
                self.send_error_json(
                    str(error),
                    HTTPStatus.INTERNAL_SERVER_ERROR,
                )
            return
        relative = "index.html" if path == "/" else path.lstrip("/")
        target = (ASSETS / relative).resolve()
        if ASSETS.resolve() not in target.parents and target != ASSETS.resolve():
            self.send_error_json("invalid path", HTTPStatus.BAD_REQUEST)
            return
        if not target.is_file():
            self.send_error_json("not found", HTTPStatus.NOT_FOUND)
            return
        body = target.read_bytes()
        content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{content_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; connect-src 'self'; img-src 'none'; "
            "font-src 'self'; object-src 'none'; base-uri 'none'; "
            "frame-ancestors 'none'; form-action 'none'",
        )
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        prefix = "/api/save/"
        if not path.startswith(prefix):
            self.send_error_json("not found", HTTPStatus.NOT_FOUND)
            return
        campaign_id = path.removeprefix(prefix)
        if campaign_id not in CAMPAIGNS:
            self.send_error_json("unknown campaign", HTTPStatus.NOT_FOUND)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if length <= 0 or length > 2_000_000:
                raise ValueError("invalid request size")
            payload = json.loads(self.rfile.read(length))
            annotations = payload.get("annotations")
            if not isinstance(annotations, dict):
                raise ValueError("annotations must be an object")
            campaign = CAMPAIGNS[campaign_id]
            save_annotations(
                Path(campaign["pack"]),
                Path(campaign["output"]),
                annotations,
            )
            self.send_json(campaign_state(campaign_id))
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
            self.send_error_json(str(error), HTTPStatus.BAD_REQUEST)

    def log_message(self, format: str, *args: object) -> None:
        print(f"[review] {format % args}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8766)
    parser.add_argument("--no-browser", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument(
        "--full",
        action="store_true",
        help="review the original 64-item campaign instead of the mini sample",
    )
    return parser.parse_args()


def check_inputs() -> None:
    missing = [
        str(path)
        for campaign in CAMPAIGNS.values()
        for path in (Path(campaign["pack"]),)
        if not path.is_file()
    ]
    missing.extend(
        str(path)
        for path in (ASSETS / "index.html", ASSETS / "styles.css", ASSETS / "app.js")
        if not path.is_file()
    )
    if missing:
        raise FileNotFoundError(f"missing review files: {', '.join(missing)}")
    for campaign_id in CAMPAIGNS:
        campaign_state(campaign_id)


def main() -> int:
    global CAMPAIGNS
    args = parse_args()
    CAMPAIGNS = FULL_CAMPAIGNS if args.full else MINI_CAMPAIGNS
    check_inputs()
    if args.check:
        print("human review UI inputs are ready")
        return 0
    address = ("127.0.0.1", args.port)
    url = f"http://{address[0]}:{address[1]}"
    server = ThreadingHTTPServer(address, ReviewHandler)
    print(f"Human review UI: {url}")
    print("Press Ctrl+C to stop.")
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
