"""Render REPORT.md to a formatted PDF (reportlab platypus)."""
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, Preformatted)

SRC = "/Users/arjunsingla/Desktop/SAM2_Pipeline/REPORT.md"
OUT = "/Users/arjunsingla/Desktop/SAM2_Pipeline/REPORT.pdf"

REPL = {"→": "->", "×": "x", "≤": "<=", "≥": ">=", "λ": "lambda", "β": "beta",
        "Σ": "sum", "∈": " in ", "∪": " U ", "ᵀ": "^T", "·": "*", "≈": "~",
        "—": " -- ", "–": "-", "✓": "[OK]", "✗": "[X]", "²": "^2", "‖": "||"}


def clean(s):
    for a, b in REPL.items():
        s = s.replace(a, b)
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"\*(.+?)\*", r"<i>\1</i>", s)
    s = re.sub(r"`(.+?)`", r'<font face="Courier" size="8">\1</font>', s)
    return s


styles = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=styles["Title"], fontSize=17, spaceAfter=6)
H2 = ParagraphStyle("H2", parent=styles["Heading1"], fontSize=13, spaceBefore=12, spaceAfter=4,
                    textColor=colors.HexColor("#1a4d7a"))
BODY = ParagraphStyle("B", parent=styles["Normal"], fontSize=9.2, leading=12.5, spaceAfter=4)
CELL = ParagraphStyle("C", parent=styles["Normal"], fontSize=7.6, leading=9.5)
CODE = ParagraphStyle("Code", parent=styles["Code"], fontSize=7.5, leading=9.5,
                      backColor=colors.HexColor("#f2f2f2"), borderPadding=4)

story = []
lines = open(SRC).read().split("\n")
i = 0
while i < len(lines):
    ln = lines[i]
    if ln.startswith("```"):
        block = []
        i += 1
        while i < len(lines) and not lines[i].startswith("```"):
            block.append(lines[i]); i += 1
        txt = "\n".join(block)
        for a, b in REPL.items():
            txt = txt.replace(a, b)
        story.append(Preformatted(txt, CODE))
        story.append(Spacer(1, 4))
    elif ln.startswith("# "):
        story.append(Paragraph(clean(ln[2:]), H1))
    elif ln.startswith("## "):
        story.append(Paragraph(clean(ln[3:]), H2))
    elif ln.startswith("|"):
        rows = []
        while i < len(lines) and lines[i].startswith("|"):
            cells = [c.strip() for c in lines[i].strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-+:?", c) for c in cells):
                rows.append([Paragraph(clean(c), CELL) for c in cells])
            i += 1
        i -= 1
        ncol = len(rows[0])
        width = 180 * mm
        if ncol == 4 and rows and "Method" in str(rows[0][1].text if hasattr(rows[0][1],'text') else ""):
            colw = [8*mm, 78*mm, 48*mm, 46*mm]
        elif ncol == 4:
            colw = [10*mm, 80*mm, 45*mm, 45*mm]
        elif ncol == 5:
            colw = [34*mm, 26*mm, 30*mm, 40*mm, 50*mm]
        else:
            colw = [width / ncol] * ncol
        t = Table(rows, colWidths=colw, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1a4d7a")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#bbbbbb")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#eef3f8")]),
            ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3),
            ("TOPPADDING", (0, 0), (-1, -1), 2), ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
        ]))
        story.append(t)
        story.append(Spacer(1, 6))
    elif ln.startswith("---"):
        story.append(Spacer(1, 4))
    elif ln.strip().startswith(("-", "1.", "2.", "3.", "4.", "5.", "6.", "7.", "8.")):
        story.append(Paragraph(clean(ln.strip()), BODY))
    elif ln.strip():
        story.append(Paragraph(clean(ln.strip()), BODY))
    i += 1

doc = SimpleDocTemplate(OUT, pagesize=A4, leftMargin=15*mm, rightMargin=15*mm,
                        topMargin=14*mm, bottomMargin=14*mm)
doc.build(story)
print("saved", OUT)
