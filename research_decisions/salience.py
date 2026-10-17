"""Context Salience and Zoning Preprocessor for Decision Models.

Handles arbitrarily large inputs (e.g. 50k+ char code dumps, stack traces, diffs)
by extracting high-salience instruction zones (Head & Tail) and replacing raw
payload blocks with dense semantic tokens. Runs in sub-millisecond Python
and microseconds in Go.
"""

import re
from typing import Tuple, List

# Common language markers in fenced code blocks
LANG_REGEX = re.compile(r"```([a-zA-Z0-9_\-\+]+)?\n([\s\S]*?)```")
STACKTRACE_PATTERNS = [
    re.compile(r"(panic:\s*.+)"),
    re.compile(r"(Traceback \(most recent call last\):[\s\S]+)"),
    re.compile(r"(\s+at\s+[\w\.\$/]+:\d+:\d+)"),
    re.compile(r"(goroutine \d+ \[\w+\]:)"),
]
DIFF_HEADER_REGEX = re.compile(r"diff --git a/.* b/.*")


def extract_salient_context(
    text: str,
    head_chars: int = 512,
    tail_chars: int = 512,
) -> Tuple[str, List[str]]:
    """Extract salient instruction text and structural markers from large input.
    
    Returns:
        (compressed_text, structural_markers)
    """
    if not text or len(text.strip()) == 0:
        return "", []

    markers: List[str] = []

    # 1. Detect and compress fenced code blocks
    code_blocks = LANG_REGEX.findall(text)
    if code_blocks:
        markers.append("__HAS_CODE_BLOCK__")
        for lang, _ in code_blocks:
            if lang:
                markers.append(f"__LANG_{lang.upper()}__")

    # 2. Detect stack traces
    has_stack = any(p.search(text) for p in STACKTRACE_PATTERNS)
    if has_stack:
        markers.append("__HAS_STACKTRACE__")

    # 3. Detect git diffs
    if DIFF_HEADER_REGEX.search(text):
        markers.append("__HAS_GIT_DIFF__")

    # Replace code blocks with summary tokens to prevent trigram hash flooding
    clean_text = LANG_REGEX.sub(r" __CODE_BLOCK_\1__ ", text)

    # 4. Head & Tail windowing if input is large
    clean_text = clean_text.strip()
    if len(clean_text) > (head_chars + tail_chars + 128):
        head = clean_text[:head_chars].strip()
        tail = clean_text[-tail_chars:].strip()
        compressed = f"{head}\n... [PAYLOAD_TRUNCATED] ...\n{tail}"
    else:
        compressed = clean_text

    # Append structural markers as pseudo-words for feature extraction
    if markers:
        compressed = f"{compressed} {' '.join(markers)}"

    return compressed, markers
