"""Download the three handbook corpora, pinned to exact commits.

The handbooks are ~90MB and are not vendored into this repo — they belong to
their authors and they change. This script makes the corpus reproducible
instead: anyone cloning the project runs it and gets the same sources.

THE PIN IS THE POINT. These are live repositories that change daily. An
earlier version cloned each one at whatever HEAD happened to be, which meant
the corpus quietly differed between machines and between runs. The golden
evaluation dataset is generated from this corpus, so an unpinned corpus makes
the dataset unreproducible — and CI caught exactly that, with a document
sampled locally being absent by the time CI ran hours later.

Every phase's results are compared against every other phase's. That is only
meaningful if they were all measured on the same documents.

To deliberately move to newer content: update the commit SHAs here, re-run
this script, regenerate the goldens, and commit both together so the corpus
change and the dataset change land in the same reviewable diff.

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
        "commit": "37f0a1088dd8606ec05134b31b241e2923aac80c",
        "sparse": "content/handbook",
    },
    {
        "slug": "sourcegraph",
        "url": "https://github.com/sourcegraph/handbook.git",
        "commit": "5752928b213a3ddbc7cee4459524f404a5b96fb4",
        "sparse": "content",
    },
    {
        "slug": "posthog",
        "url": "https://github.com/PostHog/posthog.com.git",
        "commit": "5a5e941ea3dacdd4c75fae63ff66cbab2014d983",
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
        current = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=dest, capture_output=True, text=True, check=False,
        ).stdout.strip()
        if current == source["commit"] and not force:
            return "already at pinned commit"
        shutil.rmtree(dest)

    dest.mkdir(parents=True)

    # `git clone` cannot target an arbitrary commit, so fetch the specific
    # SHA directly. --depth 1 keeps it to a single commit's worth of history
    # and --filter=blob:none defers file contents until sparse-checkout says
    # which ones are actually wanted. Cloning GitLab's handbook without both
    # pulls gigabytes of history and images for a few thousand markdown files.
    run(["git", "init", "--quiet"], cwd=dest)
    run(["git", "remote", "add", "origin", source["url"]], cwd=dest)
    run(["git", "config", "core.sparseCheckout", "true"], cwd=dest)
    run(["git", "fetch", "--quiet", "--depth", "1", "--filter=blob:none",
         "origin", source["commit"]], cwd=dest)
    run(["git", "sparse-checkout", "set", source["sparse"]], cwd=dest)
    run(["git", "checkout", "--quiet", "FETCH_HEAD"], cwd=dest)

    count = sum(1 for _ in dest.rglob("*.md"))
    return f"{count} markdown files at {source['commit'][:8]}"


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
