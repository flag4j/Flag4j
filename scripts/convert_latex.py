#
#   MIT License
#
#   Copyright (c) 2025. Jacob Watters
#
#   Permission is hereby granted, free of charge, to any person obtaining a copy
#   of this software and associated documentation files (the "Software"), to deal
#   in the Software without restriction, including without limitation the rights
#   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#   copies of the Software, and to permit persons to whom the Software is
#   furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
#   copies or substantial portions of the Software.
#
#   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WArANTY OF ANY KIND, EXPRESS OR
#   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WArANTIES OF MERCHANTABILITY,
#   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#   SOFTWARE.
#

import glob
import re
import sys


# Mapping between common html characters and strings to equivalent latex.
# The order that these are applied in matters and must match the order of insertion.
# Must be using pyton 3.7+ to guarantee a dict maintains insertion order.
html2latex = {
    # Braces
    r"\{": r"\\left\{", r"\}": r"\\right\}",

    # Simple fractions
    r"\([ ]*([^()<>]+)[ ]*\)[ ]*/[ ]*\([ ]*([^()<>]+)[ ]*\)": r"\\cfrac{\g<1>}{\g<2>}",

    # Parens, brackets
    r"\(": r"\\left(", r"\)": r"\\right)", r"\[": r"\\left[", r"]": r"\\right]",

    # Sums and products
    r"&Sigma;<sub>(.*?)</sub><sup>(.*?)</sup>": r"\\sum_{\g<1>}^{\g<2>}",
    r"&Pi;<sub>(.*?)</sub><sup>(.*?)</sup>": r"\\prod_{\g<1>}^{\g<2>}",

    # Lowercase greek
    r"&alpha;": r"\\alpha ", r"&beta;": r"\\beta ", r"&gamma;": r"\\gamma ", r"&delta;": r"\\delta ", r"&epsilon;": r"\\varepsilon ",
    r"&zeta;": r"\\zeta ", r"&eta;": r"\\eta ", r"&theta;": r"\\theta ", r"&iota;": r"\\iota ", r"&kappa;": r"\\kappa ",
    r"&lambda;": r"\\lambda ", r"&mu;": r"\\mu ", r"&nu;": r"\\nu ", r"&xi;": r"\\xi ", r"&omicron;": r"\\omicron ", r"&pi;": r"\\pi ",
    r"&rho;": r"\\rho ", r"&sigma;": r"\\sigma ", r"&tau;": r"\\tau ", r"&upsilon;": r"\\upsilon ", r"&phi;": r"\\phi ",
    r"&chi;": r"\\chi ", r"&psi;": r"\\psi ", r"&omega;": r"\\omega ",

    # Uppercase greek
    r"&Alpha;": r"\\Alpha ", r"&Beta;": r"\\Beta ", r"&Gamma;": r"\\Gamma ", r"&Delta;": r"\\Delta ", r"&Epsilon;": r"\\Epsilon ",
    r"&Zeta;": r"\\Zeta ", r"&Eta;": r"\\Eta ", r"&Theta;": r"\\Theta ", r"&Iota;": r"\\Iota ", r"&Kappa;": r"\\Kappa ",
    r"&Lambda;": r"\\Lambda ", r"&Mu;": r"\\Mu ", r"&Nu;": r"\\Nu ", r"&Xi;": r"\\Xi ", r"&Omicron;": r"\\Omicron ", r"&Pi;": r"\\Pi ",
    r"&Rho;": r"\\Rho ", r"&Sigma;": r"\\Sigma ", r"&Tau;": r"\\Tau ", r"&Upsilon;": r"\\Upsilon ", r"&Phi;": r"\\Phi ",
    r"&Chi;": r"\\Chi ", r"&Psi;": r"\\Psi ", r"&Omega;": r"\\Omega ",

    # Sub/superscripts
    r"<sub>": r"_{", "</sub>": "}", r"<sup>": r"^{", "</sup>": r"}",

    # Boldface
    r"<b>(.*?)</b>": r"\\mathbf{\g<1>}", r"<strong>(.*?)</strong>": r"\\mathbf{\g<1>}",

    # Common sets.
    r"ℝ": r"\\mathbb{R}", r"&#8477": r"\\mathbb{R}", r"ℚ": r"\\mathbb{Q}", r"ℂ": r"\\mathbb{C}",
    r"ℤ": r"\\mathbb{Z}", r"ℕ": r"\\mathbb{N}",

    # Operators
    r"&lt;": r"<", r"&gt;": r">", r"&le;": r"\\leq ", r"&ge;": r"\\geq ", r"&ne;": r"\\neq ", r"&plusmn;": r"\\pm ",
    r"&isin;": r"\\in ", r"&notin;": r"\\notin ", r"&radic;\((.*)?\)": r"\\sqrt{\g<1>}", r"&asymp;": r"\\approx ",
    r"&oplus;": r"\\oplus ", r"&Implies;": r"\\implies", r"&times;": r"\\times", r"&middot;": r"\\cdot",

    # Other symbols.
    r"&infin;": r"\\inf ", r"\.\.\.": r"\\cdots ", r"&ell;": r"\\ell ", r"&deg;": r"^{\\circ}", r"exp": r"\\exp",
    r"log": r"\\log", r"ln": r"\\ln", r"sin": r"\\sin", r"cos": r"\\cos", r"tan": r"\\tan",
    r"cot": r"\\cot", r"sec": r"\\sec", r"csc": r"\\csc", r"min": r"\\min", r"max": r"\\max",
    r"Re": r"\\Re", r"Im": r"\\Im",
}

mathjax_script = '<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>'

def inject_mathjax(content: str) -> str:
    """
    Injects the MathJax script into the header of a html file.
    :param content: The html content.
    :return: The original html content with the MathJax script injected.
    """
    head_match = re.search(r'<head.*?>', content, re.IGNORECASE)
    if head_match:
        head_end = head_match.end()
        content = content[:head_end] + "\n    " + mathjax_script + content[head_end:]
        print(f"[LATEX CONVERT] --      - injecting MathJax script")

    return content


def convert_simple_inline(match: re.Match) -> str:
    """
    Converts matched content within a latex-inline tag to inline latex.
    :param match: The matched content.
    :return: The matched content formatted as inline latex.
    """
    return convert_simple(match, "\\(", "\\)")


def convert_simple_display(match: re.Match) -> str:
    """
    Converts matched content within a latex-display tag to display mode latex.
    :param match: The matched content.
    :return: The matched content formatted as display mode latex.
    """
    return convert_simple(match, "\\[", "\\]")


def convert_simple(match: re.Match, opened: str, closed: str) -> str:
    """
    Converts matched content within a latex-simple-(inline)|(display) tag to latex.
    :param match: The matched content.
    :param opened: Opening string for math block. Must be one of ["\\(", "\\[", "$$"]
    :param closed: Closing string for math block. Must be one of ["\\)", "\\]", "$$"]
    :return: The matched content formatted as latex.
    """
    content = (match.group(1)
               .replace("<pre>", "")
               .replace("</pre>", "")
               .replace("<blockquote>", "")
               .replace("</blockquote>", "")
               .strip())
    for pattern, replacement in html2latex.items():
        content = re.sub(pattern, replacement, content)
    return f"{opened} {content} {closed}"


def convert_simple_eq_aligned(match: re.Match) -> str:
    """
    Converts matched text from a latex-simple-aligned tag to latex in an 'align' block. Each line will be aligned by the equals
    or implies symbol. Assumes that there is only one equals symbol per line.
    :param match: Matched content.
    :return: The matched content within a latex 'align' block such that each line is aligned by an equals' character.
    """
    return convert_simple_aligned(match, "=", "=")


def convert_simple_impl_aligned(match: re.Match) -> str:
    """
    Converts matched text from a latex-simple-aligned tag to latex in an 'align' block. Each line will be aligned by the equals
    or implies symbol. Assumes that there is only one equals symbol per line.
    :param match: Matched content.
    :return: The matched content within a latex 'align' block such that each line is aligned by an equals' character.
    """
    return convert_simple_aligned(match, "&Implies;", "\\implies")


def convert_simple_aligned(match: re.Match, align_token, align_replacement) -> str:
    content = (match.group(1)
               .replace("<pre>", "")
               .replace("</pre>", "")
               .replace("<blockquote>", "")
               .replace("</blockquote>", "")
               .replace(align_token, f"&{align_replacement}")
               .strip()
               .replace("\n", " \\\\ \n"))
    for pattern, replacement in html2latex.items():
        content = re.sub(pattern, replacement, content)
    return f"\\[ \\begin{{align*}} {content} \\end{{align*}} \\]"


def replace_match(match: re.Match) -> str:
    """
    Extracts latex from html comment for generic latex-replaceable tags.
    :param match: Matched content.
    :return: The latex content within the html comment.
    """
    latex_block = match.group(3)  # The LaTeX content inside the comment.
    return f"{latex_block}"


def process_javadoc(file_path) -> None:
    """
    Processes a single html Javadoc file to replace specified html formatted equations with latex.
    :param file_path: Path to the Javadoc file.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        content = f.read()
        print(f"[LATEX CONVERT] -- parsing file {file_path}")

    content = inject_mathjax(content)

    print(f"[LATEX CONVERT] --      - performing simple LaTeX replacements")
    # Handle simply replaceable html.

    content = re.sub(r'<span class="latex-inline">\s*(.*?)\s*</span>',
                     convert_simple_inline, content, flags=re.DOTALL)
    content = re.sub(r'<span class="latex-display">\s*(.*?)\s*</span>',
                     convert_simple_display, content, flags=re.DOTALL)
    content = re.sub(r'<span class="latex-eq-align">\s*(.*?)\s*</span>',
                     convert_simple_eq_aligned, content, flags=re.DOTALL)
    content = re.sub(r'<span class="latex-impl-align">\s*(.*?)\s*</span>',
                     convert_simple_impl_aligned, content, flags=re.DOTALL)

    # Handel custom replacements.
    pattern = re.compile(
        r'(<span class="latex-replace">.*?</span>)\s*<!--\s*LATEX:\s*(\{@literal)?\s*(.*?)\s*(\})?\s*-->',
        re.DOTALL
    )

    # Apply the replacement.
    content = pattern.sub(replace_match, content)

    print(f"[LATEX CONVERT] --      - performing custom LaTeX replacement")
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content)


# Process all Javadoc HTML files in
base_dir = sys.argv[1] if len(sys.argv) > 1 else "target/reports/apidocs"
for file in glob.glob(base_dir + "/**/*.html", recursive=True):
    process_javadoc(file)

print("[LATEX CONVERT] -- complete")
