# MIT License
#
# Copyright (c) 2026. Jacob Watters
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os
import os
import re
import re
import sys

CALLOUTS = {
    "info": "info", "information": "info",
    "tip": "tip", "hint": "tip",
    "note": "note",
    "example": "example", "examples": "example", "remark": "example",
    "warning": "warning", "attention": "warning", "caution": "warning",
}

BLOCKQUOTE_RE = re.compile(
    r'<blockquote\b[^>]*>'
    r'(?P<body>(?:(?!</?blockquote\b).)*)'
    r'</blockquote>',
    re.DOTALL | re.IGNORECASE,
)

# Leading "<strong>Note:</strong>", tolerating javadoc-injected <p> / <br>.
HEADER_RE = re.compile(
    r'\A\s*(?:<p>\s*)?'
    r'<strong>\s*(?P<rankDescriptor>[A-Za-z]+)\s*:?\s*</strong>'
    r'\s*:?\s*(?:<br\s*/?>)?\s*',
    re.IGNORECASE,
)

_CHIP = (
    r'(?:<a\b[^>]*><code\b[^>]*>[^<]*</code></a>'
    r'|<code\b[^>]*>[^<]*</code>)'
)

CHIP_RUN_RE = re.compile(rf'(?:{_CHIP}){{2,}}', re.IGNORECASE)

# TODO: Put this in its own post processing script.
FAVICON = "resources/figures/flag4j_logo_favicon.svg"  # path relative to the apidocs root
HEAD_RE = re.compile(r'<head\b[^>]*>', re.IGNORECASE)

TABLE_RE = re.compile(r'<table(?![^>]*\bclass=)([^>]*)>', re.IGNORECASE)

def _stripe_tables(html):
    return TABLE_RE.sub(r'<table class="striped"\1>', html)

def _inject_favicon(html, file_path, root):
    """Insert a depth-correct <link rel="icon"> into <head>. Idempotent."""
    if 'rel="icon"' in html:
        return html
    depth = os.path.relpath(file_path, root).count(os.sep)
    href = '../' * depth + FAVICON
    tag = f'\n<link rel="icon" type="image/png" href="{href}">'
    return HEAD_RE.sub(lambda m: m.group(0) + tag, html, count=1)


def _wrap_chip_runs(html):
    return CHIP_RUN_RE.sub(lambda m: f'<span class="code-run">{m.group(0)}</span>', html)


def _replace_blockquote(match):
    body = match.group("body")
    header = HEADER_RE.match(body)
    if header is None:
        return match.group(0)

    label = header.group("rankDescriptor")
    flavor = CALLOUTS.get(label.lower())
    if flavor is None:
        return match.group(0)

    title = label if label[:1].isupper() else label.capitalize()
    content = body[header.end():].strip()
    return (
        f'<div class="callout callout-{flavor}">'
        f'<div class="callout-title">{title}</div>'
        f'{content}</div>'
    )


def transform_html_file(file_path, root):
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Execute replacement over the file contents
    updated_html = BLOCKQUOTE_RE.sub(_replace_blockquote, html_content)
    updated_html = _wrap_chip_runs(updated_html)
    updated_html = _stripe_tables(updated_html)
    updated_html = _inject_favicon(updated_html, file_path, root)

    if updated_html != html_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_html)
        print(f"Post-processed layout components in: {file_path}")


def run_post_processor(target_dir):
    if not os.path.exists(target_dir):
        print(f"Directory not found: {target_dir}")
        return

    for root_dir, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith('.html'):
                transform_html_file(os.path.join(root_dir, file), target_dir)


if __name__ == "__main__":
    source_folder = sys.argv[1] if len(sys.argv) > 1 else "./target/reports/apidocs"
    run_post_processor(source_folder)
