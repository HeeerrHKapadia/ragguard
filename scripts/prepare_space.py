"""Assemble a ready-to-push Hugging Face Space directory.

A Space is its own git repository with the Dockerfile at its root, so it
cannot simply point at this one. This copies exactly what the image build
needs and nothing else — no corpus, no test suite, no benchmarks — into
build/space/, which is then a working tree that can be pushed as-is.

The layout deliberately mirrors this repository rather than flattening it,
so `deploy/space/Dockerfile` builds identically from either context. A
flattened copy would need its COPY paths rewritten, and a Dockerfile that
differs between what is tested and what is deployed is how deployments
diverge from the thing that was verified.

Run:  uv run python scripts/prepare_space.py
"""

from __future__ import annotations

import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ragguard.config import PROJECT_ROOT

OUT = PROJECT_ROOT / "build" / "space"

# Everything the build touches. The corpus is fetched during the build, so
# data/ is absent by design — copying 90MB of handbooks into a git push would
# be slow and pointless.
TREES = ["src", "static", "config", "scripts", "db", "deploy"]
FILES = ["pyproject.toml", "uv.lock"]

# Excluded from the copied trees: nothing here runs inside the image, and a
# Space repository is public.
SKIP_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".venv"}


def ignore(_dir: str, names: list[str]) -> set[str]:
    return {n for n in names if n in SKIP_DIRS or n.endswith(".pyc")}


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    for tree in TREES:
        source = PROJECT_ROOT / tree
        if not source.exists():
            print(f"  missing: {tree}")
            return 1
        shutil.copytree(source, OUT / tree, ignore=ignore)

    for name in FILES:
        shutil.copy2(PROJECT_ROOT / name, OUT / name)

    # A Space expects its Dockerfile and README at the root. The README
    # carries the frontmatter that tells Spaces this is a Docker app on port
    # 7860 — without it the Space builds and then serves nothing.
    shutil.copy2(PROJECT_ROOT / "deploy" / "space" / "Dockerfile", OUT / "Dockerfile")
    shutil.copy2(PROJECT_ROOT / "deploy" / "space" / "README.md", OUT / "README.md")

    (OUT / ".gitattributes").write_text("* text=auto eol=lf\n", encoding="utf-8")

    files = sum(1 for p in OUT.rglob("*") if p.is_file())
    size = sum(p.stat().st_size for p in OUT.rglob("*") if p.is_file())

    header = (OUT / "README.md").read_text(encoding="utf-8").splitlines()[:8]
    if not header or header[0].strip() != "---":
        print("  README is missing Space frontmatter — the Space will not start")
        return 1

    print(f"\n  {OUT.relative_to(PROJECT_ROOT)}")
    print(f"  {files} files, {size / 1024:.0f} KB")
    print("\n  frontmatter:")
    for line in header:
        print(f"    {line}")

    print(
        "\n  Push it:\n"
        "    cd build/space\n"
        "    git init && git add -A && git commit -m 'ragguard'\n"
        "    git remote add origin https://huggingface.co/spaces/<user>/ragguard\n"
        "    git push -u origin main\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
