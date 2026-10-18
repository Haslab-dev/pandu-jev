"""Reusable TypeSafe Jev Teacher Client for Knowledge Distillation.

Queries the live TypeSafe Jev (Cloud RLCD) API as the Teacher, harvesting
soft probability distributions and typed question telemetry for student models.
"""

import os
import time
from typing import Any, Dict, List, Optional
import requests

from datasets.salience import extract_salient_context

DEFAULT_JEV_KEY = os.getenv("TYPESAFE_API_KEY", "")
DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"


class JevTeacherClient:
    """Client for querying TypeSafe Jev API to obtain soft distributions."""

    def __init__(
        self,
        api_key: str = DEFAULT_JEV_KEY,
        endpoint: str = DEFAULT_ENDPOINT,
        model: str = DEFAULT_MODEL,
        timeout: float = 6.0,
    ):
        self.api_key = api_key
        self.endpoint = endpoint
        self.model = model
        self.timeout = timeout
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        })

    def query_category_distribution(
        self,
        prompt: str,
        category_criteria: Optional[Dict[str, str]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Query Jev for full categorical soft probability distribution over tool categories."""
        salient_prompt, _ = extract_salient_context(prompt, head_chars=768, tail_chars=768)

        criteria = category_criteria or {
            "explore": "Search, find files, directory tree, grep",
            "read": "Read, view, inspect, examine file contents",
            "modify": "Edit, replace, refactor, write, fix bugs",
            "execute": "Run bash, terminal, compile, build, test",
            "git": "Git operations, commit, diff, branch, status",
            "web": "Fetch URLs, browse web, HTTP requests",
            "skills": "Skills management, workflows",
            "mcp": "Model context protocol, database queries",
            "none": "General conversation, questions, explanations",
        }

        payload = {
            "model": self.model,
            "state": f"User intent and context:\n{salient_prompt}",
            "questions": {
                "category": {
                    "type": "choice",
                    "instructions": "Classify the user intent into the appropriate tool category",
                    "criteria": criteria,
                },
                "needs_tools": {
                    "type": "noul",
                    "instructions": "Does this request require searching, reading, modifying, or executing tools on the codebase?",
                },
            },
        }

        try:
            resp = self.session.post(self.endpoint, json=payload, timeout=self.timeout)
            if resp.status_code == 200:
                data = resp.json()
                cat_ans = data.get("answers", {}).get("category", {})
                tool_ans = data.get("answers", {}).get("needs_tools", {})
                return {
                    "choice": cat_ans.get("choice"),
                    "confidence": cat_ans.get("confidence", 0.0),
                    "probabilities": cat_ans.get("probabilities", {}),
                    "needs_tools": tool_ans.get("noul", 1.0),
                }
            else:
                print(f"⚠️ Jev API error {resp.status_code}: {resp.text[:120]}")
        except Exception as e:
            print(f"⚠️ Jev Teacher query failed: {e}")
        return None
