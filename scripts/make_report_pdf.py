from __future__ import annotations

import textwrap
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "architecture_report.md"
TARGET = ROOT / "architecture_report.pdf"


def escape_pdf_text(text: str) -> str:
    ascii_text = text.encode("latin-1", errors="replace").decode("latin-1")
    return ascii_text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def markdown_to_lines(markdown: str) -> list[str]:
    lines: list[str] = []
    for raw in markdown.splitlines():
        line = raw.strip()
        if not line:
            lines.append("")
            continue
        line = line.replace("#", "").replace("`", "")
        for wrapped in textwrap.wrap(line, width=92) or [""]:
            lines.append(wrapped)
    return lines


def build_pdf(lines: list[str]) -> bytes:
    per_page = 58
    pages = [lines[index : index + per_page] for index in range(0, len(lines), per_page)] or [[]]
    objects: list[bytes] = []
    font_id = 3 + len(pages) * 2
    kids: list[str] = []

    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objects.append(b"")
    object_id = 3
    for page in pages:
        page_id = object_id
        content_id = object_id + 1
        kids.append(f"{page_id} 0 R")
        page_object = (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 {font_id} 0 R >> >> /Contents {content_id} 0 R >>"
        ).encode("ascii")
        text_ops = ["BT /F1 10 Tf 50 760 Td 12 TL"]
        for line in page:
            text_ops.append(f"({escape_pdf_text(line)}) Tj T*")
        text_ops.append("ET")
        stream = "\n".join(text_ops).encode("latin-1")
        content_object = b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream"
        objects.append(page_object)
        objects.append(content_object)
        object_id += 2
    objects[1] = f"<< /Type /Pages /Kids [{' '.join(kids)}] /Count {len(pages)} >>".encode("ascii")
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    output = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for index, obj in enumerate(objects, start=1):
        offsets.append(len(output))
        output.extend(f"{index} 0 obj\n".encode("ascii"))
        output.extend(obj)
        output.extend(b"\nendobj\n")
    xref_at = len(output)
    output.extend(f"xref\n0 {len(objects) + 1}\n".encode("ascii"))
    output.extend(b"0000000000 65535 f \n")
    for offset in offsets[1:]:
        output.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    output.extend(
        (
            f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_at}\n%%EOF\n"
        ).encode("ascii")
    )
    return bytes(output)


def main() -> None:
    lines = markdown_to_lines(SOURCE.read_text(encoding="utf-8"))
    TARGET.write_bytes(build_pdf(lines))
    print(f"wrote {TARGET}")


if __name__ == "__main__":
    main()
