"""Remove the handover blocks from README.md and AGENTS.md before submission.

Usage (from the repository root):  python handover/strip_handover_blocks.py
Then delete the handover/ folder.
"""
import re
from pathlib import Path

PATTERN = re.compile(r"\n?<!-- HANDOVER-START -->.*?<!-- HANDOVER-END -->\n?", re.S)

for name in ("README.md", "AGENTS.md"):
    path = Path(name)
    if not path.is_file():
        continue
    text = path.read_text(encoding="utf-8")
    stripped, n = PATTERN.subn("", text)
    path.write_text(stripped, encoding="utf-8")
    print(f"{name}: removed {n} handover block(s)")
