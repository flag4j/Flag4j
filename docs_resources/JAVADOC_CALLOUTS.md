# Custom Callout Blocks In Flag4j

> Note: the quality of rendering these callouts in an IDE has only been tested in IntelliJ.

Flag4j uses callouts in the documentation. Due to the limitations of Javadoc, the easiest way to do this is with HTML. This document
serves to hold references to the common callouts seen in the Flag4j documentation. When generating Javadocs, running
`convert_blockquotes.py` will convert these callouts to slightly different formatting that looks better on the web. This callout format
is used so that IDEs can render the callouts correctly.

One thing to note is that text inside the `<blockquote>` block must be html/javadoc tags in general to render properly in the browser.
IntelliJ seems to render Markdown within these blocks fine but this does not seem to be true for browsers in general. That is to say,
use things such as:

```html
<code>int x = 1;</code> or {@code int x = 1;}
<em>this is important</em>
<strong>this is bold</strong>
```

instead of:

```markdown
`int x = 1;`
*this is important*
**this is bold**
```

An exception to this is code blocks. They may be included in the
`<blockquote>`:

````
```java
float x = 1.0f;
double y = Math.pow(x, -3.3);
```
````

## Note Callout

HTML code:

```html
<blockquote style="color: #b0bbd9; background-color: #1e3a5f; border-left: 5px solid #4b82bd; padding: 10px;">
    <strong>Note:</strong> nD-arrays may be non-contiguous in memory in some cases.
</blockquote>
```

Preview:
<blockquote style="color: #b0bbd9; background-color: #1e3a5f; border-left: 5px solid #4b82bd; padding: 10px;">
    <strong>Note:</strong> nD-arrays may be non-contiguous in memory in some cases.
</blockquote>

## Info Callout

HTML code:

```html
<blockquote style="color: #b0bbd9; background-color: #1e3a5f; border-left: 5px solid #4b82bd; padding: 10px;">
    <strong>Info:</strong> nD-arrays may be non-contiguous in memory in some cases.
</blockquote>
```

Preview:
<blockquote style="color: #b0bbd9; background-color: #1e3a5f; border-left: 5px solid #4b82bd; padding: 10px;">
    <strong>Info:</strong> nD-arrays may be non-contiguous in memory in some cases.
</blockquote>

## Tip Callout

HTML code:

```html
<blockquote style="color: #a9cfc7; background-color: #10362d; border-left: 5px solid #20a88b; padding: 10px;">
    <strong>Tip:</strong> Using array views can save memory.
</blockquote>
```

Preview:
<blockquote style="color: #a9cfc7; background-color: #10362d; border-left: 5px solid #20a88b; padding: 10px;">
    <strong>Tip:</strong> Using array views can save memory.
</blockquote>

## Example Callout

HTML code:

```html
<blockquote style="color: #cdb8e0; background-color: #372445; border-left: 5px solid #9836f4; padding: 10px;">
    <strong>Example:</strong> To create a complex number: <code> var z = new Complex128(1, 1)}</code>.
</blockquote>
```

Preview:
<blockquote style="color: #cdb8e0; background-color: #372445; border-left: 5px solid #9836f4; padding: 10px;">
    <strong>Example:</strong> To create a complex number: <code> var z = new Complex128(1, 1)}</code>.
</blockquote>

## Warning Callout

HTML code:

```html
<blockquote style="color: #d4aeae; background-color: #571f1f; border-left: 5px solid #f44336; padding: 10px;">
    <strong>Warning:</strong> Modifying this parameter may lead to undefined states.
</blockquote>
```

Preview:
<blockquote style="color: #d4aeae; background-color: #571f1f; border-left: 5px solid #f44336; padding: 10px;">
    <strong>Warning:</strong> Modifying this parameter may lead to undefined states.
</blockquote>

