"""Generate docs/rust_num_lib.pdf from the .pyi stubs. Requires: pip install reportlab"""
import ast, re, pathlib
from xml.sax.saxutils import escape
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import mm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Preformatted, PageBreak, HRFlowable

ROOT = pathlib.Path(__file__).resolve().parent.parent
STUBS = ROOT / "python" / "rust_num_lib"
ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", ss["Title"], fontSize=24, spaceAfter=6)
H2 = ParagraphStyle("H2", ss["Heading1"], fontSize=18, spaceBefore=6, spaceAfter=6)
H3 = ParagraphStyle("H3", ss["Heading2"], fontSize=13, spaceBefore=4, spaceAfter=2)
BODY = ParagraphStyle("B", ss["BodyText"], fontSize=10, leading=14)
PARAM = ParagraphStyle("P", BODY, leftIndent=14, fontSize=9.5)
SIG = ParagraphStyle("S", ss["Code"], fontName="Courier", fontSize=8.5, leading=11,
                     backColor=colors.HexColor("#f2f2f5"), borderPadding=4, spaceBefore=4, spaceAfter=4)

def sig(fn, in_class):
    a = ast.unparse(fn.args)
    if in_class and any(isinstance(d, ast.Name) and d.id == "staticmethod" for d in fn.decorator_list):
        pre = "@staticmethod\n"
    elif in_class and any(isinstance(d, ast.Name) and d.id == "property" for d in fn.decorator_list):
        pre = "@property\n"
    else:
        pre = ""
    r = f" -> {ast.unparse(fn.returns)}" if fn.returns else ""
    return f"{pre}def {fn.name}({a}){r}"

def doc_flow(doc):
    out = []
    for para in (doc or "").strip().split("\n\n"):
        lines = para.split("\n")
        text = [l for l in lines if not l.strip().startswith(":")]
        if text:
            out.append(Paragraph(escape(" ".join(l.strip() for l in text)), BODY))
        for l in lines:
            m = re.match(r"\s*:(\w+)(?: (\w+))?:\s*(.*)", l)
            if m:
                label = m.group(2) or m.group(1)
                out.append(Paragraph(f"<b>{escape(label)}</b> — {escape(m.group(3))}", PARAM))
    return out

def func_flow(fn, in_class=False):
    return [Paragraph(escape(fn.name), H3), Preformatted(sig(fn, in_class), SIG)] + doc_flow(ast.get_docstring(fn))

story = [Paragraph("rust_num_lib", H1),
         Paragraph("A scientific computing library written in Rust, with Python bindings.", BODY),
         HRFlowable(width="100%", color=colors.lightgrey, spaceBefore=6, spaceAfter=6),
         Paragraph("Modules: <b>linalg</b> (matrices, factorizations, solving) and "
                   "<b>stats</b> (descriptive statistics, sampling).", BODY),
         Preformatted("from rust_num_lib import linalg, stats\n\n"
                      "a = linalg.matrix([[2, 1], [1, 3]])\nb = linalg.vector([1, 2])\n"
                      "x = linalg.solve(a, b)\nprint(stats.mean(x))", SIG)]

for mod in ("linalg", "stats"):
    tree = ast.parse((STUBS / f"{mod}.pyi").read_text())
    story += [Spacer(1, 10)] + ([PageBreak()] if mod != "linalg" else []) + [Paragraph(f"rust_num_lib.{mod}", H2)]
    story += doc_flow(ast.get_docstring(tree))
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            story += [Paragraph(f"class {node.name}", H2)] + doc_flow(ast.get_docstring(node))
            for m in node.body:
                if isinstance(m, ast.FunctionDef):
                    story += func_flow(m, True)
        elif isinstance(node, ast.FunctionDef):
            story += func_flow(node)

out = ROOT / "docs" / "rust_num_lib.pdf"
SimpleDocTemplate(str(out), pagesize=A4, leftMargin=20*mm, rightMargin=20*mm,
                  topMargin=20*mm, bottomMargin=20*mm, title="rust_num_lib documentation").build(story)
print("wrote", out)
