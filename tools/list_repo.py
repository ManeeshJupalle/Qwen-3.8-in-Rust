"""List every file in a Hugging Face repo with its size in bytes.

Usage: python tools/list_repo.py <repo_id> [out_path]
Uses HF_TOKEN from the environment if set (anonymous otherwise).
"""
import os
import sys

from huggingface_hub import HfApi


def main() -> None:
    repo_id = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else None
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    entries = list(api.list_repo_tree(repo_id, recursive=True))
    lines = []
    total = 0
    for e in sorted(entries, key=lambda x: x.path):
        size = getattr(e, "size", None)
        if size is None:
            lines.append(f"{'<dir>':>16}  {e.path}")
        else:
            total += size
            lines.append(f"{size:>16}  {e.path}")
    lines.append(f"{'':>16}  ---")
    lines.append(f"{total:>16}  TOTAL bytes ({total / 2**30:.2f} GiB), {len(entries)} entries")
    text = f"# repo: {repo_id}\n# size_bytes  path\n" + "\n".join(lines) + "\n"
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(text)
    print(text)


if __name__ == "__main__":
    main()
