---
title: Sample Markdown Document
tags:
  - example
  - fixture
author: WorkBuddy
date: 2026-08-06
---

# Sample Markdown Document

This is a **sample** document used as a test fixture for the Markdown loader.

## Introduction

The loader copies local images and inserts `[IMAGE: {id}]` placeholders,
mirroring the PDF loader's multimodal flow.

![Sample diagram](sample_image.png)

## Code Example

```python
def greet(name: str) -> str:
    return f"Hello, {name}!"
```

## Heading Outline

A nested structure to exercise the heading outline extractor:

### Subsection A

Some text.

### Subsection B

More text with a reference image:

![ref image][diagram]

[diagram]: sample_image.png

## Remote Image

Remote images are preserved verbatim:

![remote](https://example.com/banner.png)
