#!/usr/bin/env python3
"""Build a sanitized, deterministic archive of the retained experiment traces."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
DATA = ROOT / "data"
OUTPUT = DATA / "swarm-runs.tar.gz"
CHECKSUM = DATA / "swarm-runs.sha256"

ROOT_ARTIFACTS = {
    "agent-results.json",
    "annotations.jsonl",
    "checkpoint.json",
    "complete.json",
    "config.yaml",
    "events.jsonl",
    "experiment-state.json",
    "inference.jsonl",
    "manifest.json",
    "messageboard.jsonl",
    "submissions.jsonl",
}


def _included(relative: Path) -> bool:
    if len(relative.parts) == 1:
        return relative.name in ROOT_ARTIFACTS or (
            relative.name.startswith("messageboard-agent-")
            and relative.suffix == ".jsonl"
        )
    return (
        len(relative.parts) == 2
        and relative.parts[0] == "messages"
        and relative.suffix == ".jsonl"
    )


def _sanitize(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _sanitize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize(item) for item in value]
    if isinstance(value, str) and value.startswith("/Users/") and value.endswith(".gguf"):
        return Path(value).name
    return value


def _artifact_bytes(path: Path) -> bytes:
    if path.suffix == ".json":
        value = _sanitize(json.loads(path.read_text(encoding="utf-8")))
        return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()
    if path.suffix == ".jsonl":
        records = []
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(
                    json.dumps(
                        _sanitize(json.loads(line)),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                )
        return (("\n".join(records) + "\n") if records else "").encode()
    data = path.read_bytes()
    if b"/Users/" in data:
        raise ValueError(f"machine-specific path remains in {path}")
    return data


def _add_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name)
    info.size = len(data)
    info.mode = 0o644
    info.mtime = 0
    info.uid = 0
    info.gid = 0
    info.uname = ""
    info.gname = ""
    archive.addfile(info, io.BytesIO(data))


def main() -> None:
    run_dirs = sorted(path for path in RUNS.iterdir() if path.is_dir())
    if not run_dirs:
        raise SystemExit("No retained runs found")
    agent_rollouts = sum(
        len(json.loads((run_dir / "agent-results.json").read_text(encoding="utf-8")))
        for run_dir in run_dirs
        if (run_dir / "agent-results.json").exists()
    )

    DATA.mkdir(exist_ok=True)
    with OUTPUT.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
            with tarfile.open(fileobj=compressed, mode="w") as archive:
                for run_dir in run_dirs:
                    for path in sorted(item for item in run_dir.rglob("*") if item.is_file()):
                        relative = path.relative_to(run_dir)
                        if not _included(relative):
                            continue
                        _add_bytes(
                            archive,
                            str(Path("swarm-runs") / run_dir.name / relative),
                            _artifact_bytes(path),
                        )

    digest = hashlib.sha256(OUTPUT.read_bytes()).hexdigest()
    CHECKSUM.write_text(f"{digest}  {OUTPUT.name}\n", encoding="utf-8")
    print(
        f"Wrote {OUTPUT} with {len(run_dirs)} sessions / "
        f"{agent_rollouts} agent rollouts ({OUTPUT.stat().st_size:,} bytes)"
    )
    print(f"SHA-256: {digest}")


if __name__ == "__main__":
    main()
