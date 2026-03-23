from __future__ import annotations

import argparse
import html
import re
import struct
import zipfile
from dataclasses import dataclass
from pathlib import Path

NS = {
    "w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    "wp": "http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing",
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
}

EMU_PER_INCH = 914400
PX_PER_INCH = 96


@dataclass
class Paragraph:
    text: str
    style: str = "Normal"
    preserve_space: bool = False


@dataclass
class ImageBlock:
    alt: str
    path: Path


Block = Paragraph | ImageBlock


def xml_escape(text: str) -> str:
    return html.escape(text, quote=False)


def strip_markdown_links(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("`", "")
    return text


def parse_markdown(markdown_text: str, base_dir: Path) -> list[Block]:
    lines = markdown_text.splitlines()
    blocks: list[Block] = []
    paragraph_buffer: list[str] = []
    in_code = False
    code_lines: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph_buffer:
            return
        text = "\n".join(paragraph_buffer).strip()
        paragraph_buffer.clear()
        if text:
            blocks.append(Paragraph(strip_markdown_links(text)))

    def flush_code() -> None:
        nonlocal code_lines
        if not code_lines:
            return
        while code_lines and not code_lines[0].strip():
            code_lines.pop(0)
        while code_lines and not code_lines[-1].strip():
            code_lines.pop()
        for line in code_lines:
            blocks.append(Paragraph(line, style="Code", preserve_space=True))
        code_lines = []

    for raw_line in lines:
        line = raw_line.rstrip("\n")

        if line.startswith("```"):
            flush_paragraph()
            if in_code:
                flush_code()
                in_code = False
            else:
                in_code = True
            continue

        if in_code:
            code_lines.append(line)
            continue

        image_match = re.match(r"!\[(.*?)\]\((.*?)\)", line.strip())
        if image_match:
            flush_paragraph()
            img_path = (base_dir / image_match.group(2)).resolve()
            blocks.append(ImageBlock(image_match.group(1), img_path))
            continue

        if not line.strip():
            flush_paragraph()
            continue

        if re.match(r"^图\s*\d", line.strip()):
            flush_paragraph()
            blocks.append(Paragraph(strip_markdown_links(line.strip()), style="Caption"))
            continue

        if line.startswith("# "):
            flush_paragraph()
            blocks.append(Paragraph(strip_markdown_links(line[2:].strip()), style="Heading1"))
            continue

        if line.startswith("## "):
            flush_paragraph()
            blocks.append(Paragraph(strip_markdown_links(line[3:].strip()), style="Heading2"))
            continue

        if line.startswith("### "):
            flush_paragraph()
            blocks.append(Paragraph(strip_markdown_links(line[4:].strip()), style="Heading3"))
            continue

        if line.lstrip().startswith("- "):
            flush_paragraph()
            blocks.append(Paragraph(strip_markdown_links("• " + line.lstrip()[2:].strip()), style="NormalNoIndent"))
            continue

        paragraph_buffer.append(line)

    flush_paragraph()
    flush_code()
    return blocks


def png_dimensions(path: Path) -> tuple[int, int]:
    with path.open("rb") as fh:
        signature = fh.read(8)
        if signature != b"\x89PNG\r\n\x1a\n":
            raise ValueError(f"Unsupported image format for {path}")
        length = struct.unpack(">I", fh.read(4))[0]
        chunk_type = fh.read(4)
        if chunk_type != b"IHDR" or length < 8:
            raise ValueError(f"Invalid PNG header for {path}")
        width, height = struct.unpack(">II", fh.read(8))
        return width, height


def image_size_emu(path: Path, max_width_inches: float = 6.0) -> tuple[int, int]:
    width_px, height_px = png_dimensions(path)
    width_inches = width_px / PX_PER_INCH
    height_inches = height_px / PX_PER_INCH
    scale = min(1.0, max_width_inches / width_inches)
    return int(width_inches * scale * EMU_PER_INCH), int(height_inches * scale * EMU_PER_INCH)


def paragraph_xml(text: str, style: str = "Normal", preserve_space: bool = False) -> str:
    escaped = xml_escape(text)
    style_xml = f'<w:pPr><w:pStyle w:val="{style}"/></w:pPr>' if style != "Normal" else ""
    space_attr = ' xml:space="preserve"' if preserve_space or text.startswith(" ") or text.endswith(" ") else ""
    if style == "Code":
        run_props = (
            "<w:rPr>"
            '<w:rFonts w:ascii="Courier New" w:hAnsi="Courier New" w:eastAsia="Courier New"/>'
            '<w:sz w:val="20"/>'
            "</w:rPr>"
        )
    else:
        run_props = ""
    return f"<w:p>{style_xml}<w:r>{run_props}<w:t{space_attr}>{escaped}</w:t></w:r></w:p>"


def image_paragraph_xml(rel_id: str, name: str, cx: int, cy: int, doc_pr_id: int) -> str:
    return f"""
<w:p>
  <w:r>
    <w:drawing>
      <wp:inline distT="0" distB="0" distL="0" distR="0">
        <wp:extent cx="{cx}" cy="{cy}"/>
        <wp:docPr id="{doc_pr_id}" name="{xml_escape(name)}"/>
        <wp:cNvGraphicFramePr>
          <a:graphicFrameLocks noChangeAspect="1"/>
        </wp:cNvGraphicFramePr>
        <a:graphic>
          <a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">
            <pic:pic>
              <pic:nvPicPr>
                <pic:cNvPr id="{doc_pr_id}" name="{xml_escape(name)}"/>
                <pic:cNvPicPr/>
              </pic:nvPicPr>
              <pic:blipFill>
                <a:blip r:embed="{rel_id}"/>
                <a:stretch><a:fillRect/></a:stretch>
              </pic:blipFill>
              <pic:spPr>
                <a:xfrm>
                  <a:off x="0" y="0"/>
                  <a:ext cx="{cx}" cy="{cy}"/>
                </a:xfrm>
                <a:prstGeom prst="rect"><a:avLst/></a:prstGeom>
              </pic:spPr>
            </pic:pic>
          </a:graphicData>
        </a:graphic>
      </wp:inline>
    </w:drawing>
  </w:r>
</w:p>
""".strip()


def styles_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:styles xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:docDefaults>
    <w:rPrDefault>
      <w:rPr>
        <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体"/>
        <w:sz w:val="24"/>
      </w:rPr>
    </w:rPrDefault>
  </w:docDefaults>
  <w:style w:type="paragraph" w:default="1" w:styleId="Normal">
    <w:name w:val="Normal"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:line="360" w:lineRule="auto" w:before="0" w:after="120"/>
      <w:ind w:firstLine="480"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体"/>
      <w:sz w:val="24"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="NormalNoIndent">
    <w:name w:val="NormalNoIndent"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:spacing w:line="360" w:lineRule="auto" w:before="0" w:after="120"/>
      <w:ind w:firstLine="0"/>
    </w:pPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading1">
    <w:name w:val="heading 1"/>
    <w:basedOn w:val="Normal"/>
    <w:uiPriority w:val="9"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:before="240" w:after="120"/>
      <w:ind w:firstLine="0"/>
    </w:pPr>
    <w:rPr>
      <w:b/>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体"/>
      <w:sz w:val="32"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading2">
    <w:name w:val="heading 2"/>
    <w:basedOn w:val="Normal"/>
    <w:uiPriority w:val="9"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:before="200" w:after="100"/>
      <w:ind w:firstLine="0"/>
    </w:pPr>
    <w:rPr>
      <w:b/>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体"/>
      <w:sz w:val="28"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Heading3">
    <w:name w:val="heading 3"/>
    <w:basedOn w:val="Normal"/>
    <w:uiPriority w:val="9"/>
    <w:qFormat/>
    <w:pPr>
      <w:spacing w:before="160" w:after="80"/>
      <w:ind w:firstLine="0"/>
    </w:pPr>
    <w:rPr>
      <w:b/>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="黑体"/>
      <w:sz w:val="26"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Code">
    <w:name w:val="Code"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:spacing w:before="0" w:after="0" w:line="240" w:lineRule="auto"/>
      <w:ind w:firstLine="0" w:left="240"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Courier New" w:hAnsi="Courier New" w:eastAsia="Courier New"/>
      <w:sz w:val="20"/>
    </w:rPr>
  </w:style>
  <w:style w:type="paragraph" w:styleId="Caption">
    <w:name w:val="Caption"/>
    <w:basedOn w:val="Normal"/>
    <w:pPr>
      <w:jc w:val="center"/>
      <w:spacing w:before="60" w:after="60" w:line="360" w:lineRule="auto"/>
      <w:ind w:firstLine="0"/>
    </w:pPr>
    <w:rPr>
      <w:rFonts w:ascii="Times New Roman" w:hAnsi="Times New Roman" w:eastAsia="宋体"/>
      <w:sz w:val="22"/>
    </w:rPr>
  </w:style>
</w:styles>
"""


def content_types_xml() -> str:
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        '<Default Extension="png" ContentType="image/png"/>'
        '<Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        '<Override PartName="/word/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.styles+xml"/>'
        '</Types>'
    )


def package_rels_xml() -> str:
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
"""


def document_rels_xml(image_targets: list[str]) -> str:
    rels = [
        '<Relationship Id="rIdStyles" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
    ]
    for idx, target in enumerate(image_targets, start=1):
        rels.append(
            f'<Relationship Id="rIdImg{idx}" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" Target="{target}"/>'
        )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        + "".join(rels)
        + '</Relationships>'
    )


def document_xml(blocks: list[Block], image_map: dict[Path, tuple[str, int, int, int]]) -> str:
    body_parts: list[str] = []
    for block in blocks:
        if isinstance(block, Paragraph):
            body_parts.append(paragraph_xml(block.text, block.style, block.preserve_space))
        else:
            rel_id, cx, cy, doc_pr_id = image_map[block.path]
            body_parts.append(image_paragraph_xml(rel_id, block.alt or block.path.name, cx, cy, doc_pr_id))

    section = (
        '<w:sectPr>'
        '<w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:top="1440" w:right="1800" w:bottom="1440" w:left="1800" w:header="851" w:footer="992" w:gutter="0"/>'
        '</w:sectPr>'
    )
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        f'<w:document xmlns:w="{NS["w"]}" xmlns:r="{NS["r"]}" xmlns:wp="{NS["wp"]}" xmlns:a="{NS["a"]}" xmlns:pic="{NS["pic"]}">'
        f'<w:body>{"".join(body_parts)}{section}</w:body></w:document>'
    )


def build_docx(markdown_path: Path, output_path: Path) -> None:
    base_dir = markdown_path.parent
    blocks = parse_markdown(markdown_path.read_text(encoding="utf-8"), base_dir)

    images: list[Path] = []
    for block in blocks:
        if isinstance(block, ImageBlock) and block.path not in images:
            if not block.path.exists():
                raise FileNotFoundError(f"Image not found: {block.path}")
            images.append(block.path)

    image_map: dict[Path, tuple[str, int, int, int]] = {}
    image_targets: list[str] = []
    for idx, image_path in enumerate(images, start=1):
        cx, cy = image_size_emu(image_path)
        image_map[image_path] = (f"rIdImg{idx}", cx, cy, idx)
        image_targets.append(f"media/image{idx}.png")

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types_xml())
        zf.writestr("_rels/.rels", package_rels_xml())
        zf.writestr("word/document.xml", document_xml(blocks, image_map))
        zf.writestr("word/styles.xml", styles_xml())
        zf.writestr("word/_rels/document.xml.rels", document_rels_xml(image_targets))
        for idx, image_path in enumerate(images, start=1):
            zf.write(image_path, f"word/media/image{idx}.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a DOCX file from the project markdown report.")
    parser.add_argument("--input", default="PYTHON_DAE_REPORT.md")
    parser.add_argument("--output", default="PYTHON_DAE_REPORT.docx")
    args = parser.parse_args()
    build_docx(Path(args.input).resolve(), Path(args.output).resolve())


if __name__ == "__main__":
    main()
