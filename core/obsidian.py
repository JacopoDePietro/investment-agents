"""Writes Markdown notes to the Obsidian vault (YAML front matter + wikilinks)."""
from datetime import datetime
from pathlib import Path

def write_note(vault: Path, folder: str, name: str, meta: dict, body: str):
    path = vault / folder / f"{name}.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    fm = "\n".join(f"{k}: {v}" for k, v in meta.items())
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    path.write_text(f"---\n{fm}\nupdated: {stamp}\n---\n\n{body}\n")
    return path
