"""提取 PDF 文本到 UTF-8 文本文件。

用法:
    python scripts/extract_pdf.py <input.pdf> [output.txt]
"""

import sys
from pathlib import Path

from pypdf import PdfReader


def main() -> None:
    src = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else None
    reader = PdfReader(str(src))
    chunks = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        chunks.append(f"===== Page {i} =====\n{text}")
    full = "\n\n".join(chunks)
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(full, encoding="utf-8")
        print(f"wrote {out} ({len(full)} chars, {len(reader.pages)} pages)")
    else:
        print(full)


if __name__ == "__main__":
    main()
