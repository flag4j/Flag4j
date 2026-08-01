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


def transform_html_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Flexible pattern handling variations with or without implicit <p> tags injected by javadoc tool
    pattern_flexible = r'<blockquote\s+.*?;[^"]*">.*?<strong>\s*(info|example|warning):?\s*</strong>(.*?)</blockquote>'

    def replacer(match):
        raw_header = match.group(1).lower()
        raw_body = match.group(2)

        # 1. Map type variants dynamically based on header text
        if "warning" in raw_header or "attention" in raw_header:
            flavor = "warning"
        elif "example" in raw_header:
            flavor = "example"
        else:
            flavor = "info"

        # 2. Scrub residual formatting tags out of the internal body content
        # (e.g., removing any dangling <br> tags or empty trailing blocks)
        clean_body = re.sub(r'^\s*<br\s*/?>', '', raw_body).strip()
        clean_body = re.sub(r'^:\s*', '', clean_body).strip()  # Cleans up optional colons

        # 3. Assemble the updated production HTML structure
        return f'<div class="callout callout-{flavor}">{clean_body}</div>'

    # Execute replacement over the file contents
    updated_html = re.sub(pattern_flexible, replacer, html_content, flags=re.DOTALL | re.IGNORECASE)

    if updated_html != html_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(updated_html)
        print(f"Post-processed layout components in: {file_path}")


def run_post_processor(target_dir):
    if not os.path.exists(target_dir):
        print(f"Directory not found: {target_dir}")
        return

    for root, _, files in os.walk(target_dir):
        for file in files:
            if file.endswith('.html'):
                transform_html_file(os.path.join(root, file))


if __name__ == "__main__":
    # Points to standard Maven / Gradle Java source layouts
    source_folder = sys.argv[1] if len(sys.argv) > 1 else "./target/reports/apidocs"
    run_post_processor(source_folder)
