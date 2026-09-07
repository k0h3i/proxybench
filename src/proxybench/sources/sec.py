"""Preserve a small, explicitly selected set of SEC source documents."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")


class SecClient:
    """Use one sequential client per workspace. Do not retry access denials."""

    def __init__(self, root, identity, *, opener=urlopen, interval=1.0):
        if not re.search(r"\S+@\S+\.\S+", identity) or "\n" in identity or "\r" in identity:
            raise ValueError("Supply a real SEC client name and contact email.")
        if len(identity.split()) < 2:
            raise ValueError("Supply a client name as well as the contact email.")
        self.root = Path(root)
        self.identity = identity
        self.opener = opener
        self.interval = max(1.0, interval)
        self.next_request = 0.0
        self.stopped = False

    def fetch(self, accession, url):
        if not re.fullmatch(r"\d{10}-\d{2}-\d{6}", accession):
            raise ValueError("Invalid accession.")
        parts = urlsplit(url)
        if (parts.scheme != "https" or parts.netloc != "www.sec.gov"
                or not parts.path.startswith("/Archives/edgar/data/")
                or parts.query or parts.fragment or "%" in parts.path
                or ".." in parts.path.split("/")):
            raise ValueError("Use a direct SEC archive document URL.")
        if accession.replace("-", "") not in parts.path.split("/"):
            raise ValueError("URL does not belong to this accession directory.")
        destination = self.root / "data/raw" / accession / Path(parts.path).name
        metadata = destination.with_name(destination.name + ".retrieval.json")
        event = {"accession": accession, "url": url, "attempted_at": utc_now()}
        log = self.root / "data/manifests/retrieval-log.jsonl"
        if destination.exists():
            digest = hashlib.sha256(destination.read_bytes()).hexdigest()
            saved = json.loads(metadata.read_text()) if metadata.exists() else {}
            if saved.get("sha256") != digest or saved.get("url") != url:
                raise ValueError("Cache metadata or hash mismatch. Preserve and inspect the source.")
            event.update(outcome="cache_hit", sha256=digest, path=str(destination.relative_to(self.root)))
            append_jsonl(log, event)
            return event
        if self.stopped:
            event.update(outcome="deferred_after_access_denial")
            append_jsonl(log, event)
            return event
        time.sleep(max(0.0, self.next_request - time.monotonic()))
        self.next_request = time.monotonic() + self.interval
        request = Request(url, headers={"User-Agent": self.identity, "Accept-Encoding": "identity"})
        try:
            with self.opener(request, timeout=45) as response:
                content = response.read()
                status = response.status
                content_type = response.headers.get("Content-Type", "")
                final_url = response.geturl()
            if final_url != url or status != 200:
                raise ValueError("Unexpected redirect or response status.")
            if not content or any(marker in content[:20000].lower() for marker in (
                    b"your request originates from an undeclared automated tool",
                    b"request rate threshold exceeded", b"sec.gov | request rate")):
                self.stopped = True
                event.update(outcome="rejected_response", http_status=status)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                with destination.open("xb") as stream:
                    stream.write(content)
                event.update(outcome="downloaded", retrieved_at=utc_now(), http_status=status,
                             content_type=content_type, bytes=len(content),
                             sha256=hashlib.sha256(content).hexdigest(),
                             path=str(destination.relative_to(self.root)))
                metadata.write_text(json.dumps(event, indent=2) + "\n", encoding="utf-8")
        except HTTPError as error:
            event.update(outcome="http_error", http_status=error.code,
                         retry_after=error.headers.get("Retry-After") if error.headers else None)
            if error.code in (403, 429):
                self.stopped = True
            error.close()
        except (URLError, TimeoutError, OSError) as error:
            event.update(outcome="network_error", error=str(error))
        except ValueError as error:
            event.update(outcome="rejected_response", error=str(error))
        append_jsonl(log, event)
        return event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("accession")
    parser.add_argument("url")
    parser.add_argument("--identity-file", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    identity = args.identity_file.read_text(encoding="utf-8").strip()
    client = SecClient(args.root, identity)
    result = client.fetch(args.accession, args.url)
    print(json.dumps(result))
    return 0 if result["outcome"] in ("downloaded", "cache_hit") else 1


if __name__ == "__main__":
    raise SystemExit(main())
