"""Pandu-Micro Domain Tokenizer (2,048 Vocab Budget).

Provides fast subword/word-level tokenization for developer and long-running agent prompts:
- Compact vocabulary of 2,048 tokens strictly bounded for ~1.2MB model footprints
- Session control markers: [pad], [unk], [cls], [sep], [mask], [goal], [turn], [fail], [intent]
- Character-level fallback for arbitrary identifier out-of-vocabulary terms
"""

import re
from typing import Dict, List, Optional
from datasets.rakitkode_domain import get_rakitkode_vocabulary, RAKITKODE_DATASET
from datasets.long_running_trajectories import generate_long_running_trajectories

SPECIAL_TOKENS = ["[pad]", "[unk]", "[cls]", "[sep]", "[mask]", "[goal]", "[turn]", "[fail]", "[intent]"]
MAX_VOCAB_SIZE = 2048


def build_default_domain_vocab() -> List[str]:
    """Harvest comprehensive domain vocabulary for software engineering agent sessions."""
    vocab_list = list(SPECIAL_TOKENS)

    # 1. Add single ASCII printable characters for fallback
    for ch in range(32, 127):
        char_str = chr(ch).lower()
        if char_str not in vocab_list:
            vocab_list.append(char_str)

    # 2. Extract words from datasets & trajectories
    text_corpus: List[str] = []
    for prompt, _, _, _, _, _ in RAKITKODE_DATASET:
        text_corpus.append(prompt)
    for t in generate_long_running_trajectories():
        text_corpus.append(t.goal_prompt)
        text_corpus.append(t.current_intent)

    # Common benchmark phrases
    benchmark_phrases = [
        "StringReverse utility function to rakitkode utils go add unit tests and update docs",
        "Inspect existing utils go and package declaration",
        "Write StringReverse implementation and create utils_test go",
        "Run go test -v to verify unit tests pass",
        "Check git status and commit changes",
        "Implement new Decision File Adapter with config toggle in rakitkode decisions",
        "Compile new decisions adapter go with go build",
        "Read compiler error log missing field in DecisionAdapter struct",
        "Fix missing struct field in adapter go",
        "Migrate agent harness to pure Go micro policy engine and isolate search via subagents",
        "Diagnose and resolve high load deadlock in concurrent agent session worker pool",
        "Tracing goroutine worker pool channels and lock contention points",
        "Cyclic deadlock persisting on worker pool channel retrying patch attempt",
        "Alarm triggered pivoting strategy to inspect runtime stack dump carefully",
        "Running go test -race -count=50 to verify deadlock is completely resolved",
        "All benchmarks pass releasing v0 3 0",
    ]
    text_corpus.extend(benchmark_phrases)

    for text in text_corpus:
        words = re.findall(r"[\w]+", text.lower())
        for w in words:
            if w not in vocab_list and len(vocab_list) < MAX_VOCAB_SIZE:
                vocab_list.append(w)

    # 3. High-frequency developer, Go runtime, and agent terms
    dev_words = [
        "view_file", "replace_file_content", "write_to_file", "run_command", "manage_task",
        "send_message", "schedule", "ask_question", "read_url_content", "search_web",
        "compile", "build", "test", "race", "bench", "benchmark", "benchmarks", "deadlock",
        "mutex", "channel", "channels", "goroutine", "goroutines", "worker", "pool", "sync",
        "atomic", "context", "timeout", "cancel", "defer", "select", "struct", "interface",
        "slice", "map", "pointer", "nil", "error", "errors", "panic", "recover", "log",
        "debug", "trace", "dump", "stack", "traceback", "git", "commit", "push", "pull",
        "fetch", "branch", "checkout", "merge", "rebase", "diff", "patch", "status", "tag",
        "release", "repo", "clone", "subagent", "subagents", "delegate", "delegation", "spawn",
        "kill", "task", "session", "loop", "thrash", "thrashing", "cycle", "cycles", "stuck",
        "alarm", "pivot", "isolate", "research", "coding", "planning", "architect", "explore",
        "read", "modify", "execute", "wrapup", "discovery", "implementation", "verification",
        "debugging", "not", "dont", "never", "only", "except", "without", "prevent",
        "failure", "failures", "failed", "consecutive", "third", "fourth", "attempt",
        "persisting", "dump", "resolved", "inspecting", "passing", "verified", "stringreverse",
        "__has_code_block__", "__has_stacktrace__", "__has_git_diff__",
    ]
    for w in dev_words:
        if w not in vocab_list and len(vocab_list) < MAX_VOCAB_SIZE:
            vocab_list.append(w)

    return vocab_list


class PanduMicroTokenizer:
    """Domain tokenizer for Pandu-Micro (MicroBERT) with a 2,048 token budget."""

    def __init__(self, vocab: Optional[List[str]] = None):
        self.special_tokens = list(SPECIAL_TOKENS)
        self.pad_token_id = 0
        self.unk_token_id = 1
        self.cls_token_id = 2
        self.sep_token_id = 3
        self.mask_token_id = 4
        self.goal_token_id = 5
        self.turn_token_id = 6
        self.fail_token_id = 7
        self.intent_token_id = 8

        vocab_list = vocab or build_default_domain_vocab()
        self.id_to_token = vocab_list
        self.token_to_id = {tok: idx for idx, tok in enumerate(vocab_list)}
        self.vocab_size = len(vocab_list)

    def tokenize(self, text: str) -> List[str]:
        """Split text into words, special markers, and sub-tokens."""
        raw_tokens = re.findall(r"\[[\w]+\]|__[\w]+__|[\w]+|[^\w\s]", text.lower())
        result = []
        for tok in raw_tokens:
            if tok in self.token_to_id:
                result.append(tok)
            else:
                for ch in tok:
                    result.append(ch if ch in self.token_to_id else "[unk]")
        return result

    def encode(self, text: str, max_length: int = 384, add_special_tokens: bool = True) -> List[int]:
        """Encode text into token IDs."""
        tokens = self.tokenize(text)
        if add_special_tokens:
            token_ids = [self.cls_token_id] + [self.token_to_id.get(t, self.unk_token_id) for t in tokens] + [self.sep_token_id]
        else:
            token_ids = [self.token_to_id.get(t, self.unk_token_id) for t in tokens]

        if len(token_ids) > max_length:
            token_ids = token_ids[:max_length-1] + [self.sep_token_id]
        return token_ids

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs back to a readable string."""
        tokens = [self.id_to_token[i] if 0 <= i < self.vocab_size else "[unk]" for i in token_ids]
        tokens = [t for t in tokens if t not in ["[pad]", "[cls]", "[sep]"]]
        return " ".join(tokens)
