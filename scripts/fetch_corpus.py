"""Download the three handbook corpora.

The handbooks are ~90MB and are not vendored into this repo — they belong to
their authors and they change. This script makes the corpus reproducible
instead: anyone cloning the project runs it and gets the same sources.

Uses a sparse, blobless, depth-1 clone. Without that, GitLab's handbook repo
pulls its full history and every image, which is gigabytes for a few thousand
markdown files we actually want.

Run:  uv run python scripts/fetch_corpus.py
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"

SOURCES = [
    {
        "slug": "gitlab",
        "url": "https://gitlab.com/gitlab-com/content-sites/handbook.git",
        "sparse": "content/handbook",
    },
    {
        "slug": "sourcegraph",
        "url": "https://github.com/sourcegraph/handbook.git",
        "sparse": "content",
    },
    {
        "slug": "posthog",
        "url": "https://github.com/PostHog/posthog.com.git",
        "sparse": "contents/handbook",
    },
]


def run(cmd: list[str], cwd: pathlib.Path | None = None) -> None:
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)}\n{result.stderr.strip()}")


def fetch(source: dict, force: bool) -> str:
    dest = RAW / source["slug"]
    if dest.exists():
        if not force:
            return "already present"
        shutil.rmtree(dest)

    run([
        "git", "clone", "--quiet", "--depth", "1",
        "--filter=blob:none", "--sparse", source["url"], str(dest),
    ])
    run(["git", "sparse-checkout", "set", source["sparse"]], cwd=dest)

    count = sum(1 for _ in dest.rglob("*.md"))
    return f"{count} markdown files"


def main() -> int:
    force = "--force" in sys.argv
    RAW.mkdir(parents=True, exist_ok=True)

    print()
    for source in SOURCES:
        print(f"  {source['slug']:<14} ", end="", flush=True)
        try:
            print(fetch(source, force))
        except RuntimeError as exc:
            print(f"FAILED\n    {exc}")
            return 1

    print("\nNext:  uv run python scripts/seed.py\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
