"""
canary_manager.py
─────────────────
Canary / Honeypot Document Manager for RAG-Shield.

Concept
───────
Indirect Prompt Injection attacks hide malicious instructions inside
retrieved documents (poisoned FAQ entries, injected web pages, etc.).
The attacker's goal: the RAG pipeline retrieves a poisoned chunk and
the LLM executes the embedded instruction without the user's knowledge.

This module implements two complementary defences:

  1. Canary Document Generation & Injection
     - Generates plausible-looking knowledge-base documents that contain
       a unique, secret canary token (loaded from the RAG_CANARY_TOKEN
       environment variable — never hardcoded).
     - These decoy documents are injected into the vector store so that
       if any query causes them to be retrieved, the retrieval itself
       becomes a signal.

  2. Context-Level Canary Detection
     - Called before Layer 2 on the list of retrieved chunks.
     - If ANY retrieved chunk contains the canary token, a hard-block
       is triggered (the attacker has triggered a honeypot).
     - Also checks if the LLM response leaks the canary token (post-
       generation check, already handled in orchestrator.py).

Usage
─────
  from canary_manager import CanaryManager

  manager = CanaryManager()               # loads token from env
  doc     = manager.generate_canary_doc() # one honeypot document
  docs    = manager.generate_batch(n=3)   # multiple honeypots

  # Before running the pipeline:
  leaked = manager.check_context(retrieved_chunks)
  if leaked:
      # hard block

  # After the LLM generates a response:
  leaked = manager.check_response(response_text)
  if leaked:
      # hard block

Security note
─────────────
  The canary token MUST be set via the RAG_CANARY_TOKEN environment
  variable. Generate a fresh token with:
      python -c "import secrets; print('RAG_CANARY_TOKEN=' + secrets.token_hex(16))"
  Then add it to your .env file and NEVER commit it to source control.
"""

import os
import re
import secrets
import logging
import hashlib
from datetime import datetime, timezone
from typing import Optional

from config import CANARY_TOKEN, CANARY_INJECT_COUNT

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Canary document templates
# ─────────────────────────────────────────────────────────────────────────────

_CANARY_TEMPLATES = [
    # Support / FAQ style
    (
        "support_faq",
        """\
FAQ — Internal Reference ID: {token}
Q: How do I reset my account password?
A: Navigate to Settings → Security → Reset Password. Enter your registered
   email and follow the link sent to your inbox. Reference: {token}.
Last updated: {date}. This document is for internal use only.
""",
    ),
    # Policy document style
    (
        "policy_doc",
        """\
Data Retention Policy — Document ID: {token}
Effective Date: {date}
All customer records are retained for 7 years in accordance with
regulatory requirements. Archive reference: {token}.
For questions, contact compliance@internal.example.com.
""",
    ),
    # Technical specification style
    (
        "tech_spec",
        """\
System Configuration Note — Ref: {token}
Environment: Production  |  Region: us-east-1  |  Version: 4.2.1
The deployment pipeline validates checksums against the following
internal registry. Registry token (audit): {token}.
Last audited: {date}.
""",
    ),
    # Meeting notes style
    (
        "meeting_notes",
        """\
Meeting Notes — Session {token}
Date: {date}
Attendees: Engineering, Product, Compliance
Action items:
  1. Review Q4 roadmap by end of week.
  2. Complete data-migration checklist (ref: {token}).
  3. Follow up with legal on new terms of service.
Next meeting scheduled for two weeks from {date}.
""",
    ),
    # Changelog style
    (
        "changelog",
        """\
Changelog — Build {token}
Date: {date}
  - Fixed: Timeout issue in data export module.
  - Added: New audit logging for admin actions (audit ref: {token}).
  - Updated: Third-party authentication library to v3.1.2.
  - Removed: Deprecated API endpoints (see migration guide).
""",
    ),
]


# ─────────────────────────────────────────────────────────────────────────────
# CanaryManager
# ─────────────────────────────────────────────────────────────────────────────

class CanaryManager:
    """
    Manages honeypot document generation and canary token detection.

    Attributes
    ----------
    token : str
        The canary token loaded from environment. Empty string means
        canary checking is disabled (logged as a warning).
    inject_count : int
        Number of canary documents to inject per store (from config).
    """

    def __init__(
        self,
        token: Optional[str] = None,
        inject_count: Optional[int] = None,
    ):
        self.token = token if token is not None else CANARY_TOKEN
        self.inject_count = inject_count if inject_count is not None else CANARY_INJECT_COUNT

        if not self.token:
            logger.warning(
                "[Canary] RAG_CANARY_TOKEN is not set. "
                "Canary detection is DISABLED. "
                "Set it via: python -c \"import secrets; "
                "print('RAG_CANARY_TOKEN=' + secrets.token_hex(16))\""
            )
        else:
            # Log a hash, never the token itself
            token_hash = hashlib.sha256(self.token.encode()).hexdigest()[:12]
            logger.info("[Canary] Manager initialised. Token hash: %s", token_hash)

    @property
    def is_active(self) -> bool:
        """Returns True if canary checking is enabled (token is set)."""
        return bool(self.token)

    # ── Document generation ───────────────────────────────────────────────────

    def generate_canary_doc(self, template_index: Optional[int] = None) -> str:
        """
        Generate a single honeypot document embedding the canary token.

        Parameters
        ----------
        template_index : int, optional
            Index into _CANARY_TEMPLATES. If None, a deterministic index
            is chosen based on the token so every run is reproducible.

        Returns
        -------
        str
            A plausible-looking document containing the canary token.
        """
        if not self.token:
            raise RuntimeError(
                "Cannot generate canary document: RAG_CANARY_TOKEN is not set."
            )

        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        if template_index is None:
            # Deterministic but varied selection across runs
            idx = int(hashlib.md5(self.token.encode()).hexdigest(), 16) % len(_CANARY_TEMPLATES)
        else:
            idx = template_index % len(_CANARY_TEMPLATES)

        _, template = _CANARY_TEMPLATES[idx]
        return template.format(token=self.token, date=today)

    def generate_batch(self, n: Optional[int] = None) -> list[str]:
        """
        Generate a batch of honeypot documents using different templates.

        Parameters
        ----------
        n : int, optional
            Number of documents. Defaults to self.inject_count.

        Returns
        -------
        list[str]
            List of honeypot document strings.
        """
        count = n if n is not None else self.inject_count
        return [
            self.generate_canary_doc(template_index=i)
            for i in range(min(count, len(_CANARY_TEMPLATES)))
        ]

    # ── Detection ─────────────────────────────────────────────────────────────

    def check_context(self, chunks: list[str]) -> tuple[bool, Optional[str]]:
        """
        Check retrieved document chunks for the canary token.

        This is called BEFORE Layer 2 on the list of retrieved chunks.
        If ANY chunk contains the canary, it means either:
          a) A honeypot document was retrieved (indirect injection probe), or
          b) An attacker somehow knew the token and embedded it.
        Both cases warrant an immediate hard-block.

        Parameters
        ----------
        chunks : list[str]
            Retrieved/split document chunks.

        Returns
        -------
        (leaked: bool, source_chunk: str | None)
        """
        if not self.is_active:
            return False, None

        token_lower = self.token.lower()
        for i, chunk in enumerate(chunks):
            if token_lower in chunk.lower():
                logger.warning(
                    "[Canary] Token detected in retrieved document chunk #%d. "
                    "Hard-blocking. Chunk preview: %s",
                    i,
                    chunk[:80],
                )
                return True, chunk
        return False, None

    def check_response(self, response: str) -> tuple[bool, Optional[str]]:
        """
        Check the LLM response for the canary token.

        If the response contains the canary token it means the LLM was
        tricked into echoing it — a clear sign of indirect injection.

        Parameters
        ----------
        response : str
            Raw LLM-generated response text.

        Returns
        -------
        (leaked: bool, evidence: str | None)
        """
        if not self.is_active or not response:
            return False, None

        if self.token.lower() in response.lower():
            logger.warning(
                "[Canary] Token detected in LLM response. Hard-blocking. "
                "Response preview: %s",
                response[:80],
            )
            return True, response[:80]
        return False, None

    def check_query(self, query: str) -> tuple[bool, Optional[str]]:
        """
        Check the user query for the canary token.

        If a user's query contains the canary token, the attacker has
        somehow obtained it (or is probing). Hard-block immediately.

        Parameters
        ----------
        query : str
            User query string.

        Returns
        -------
        (leaked: bool, evidence: str | None)
        """
        if not self.is_active or not query:
            return False, None

        if self.token.lower() in query.lower():
            logger.warning(
                "[Canary] Token detected in user query. Hard-blocking. "
                "Query preview: %s",
                query[:80],
            )
            return True, query[:80]
        return False, None

    def status_report(self) -> dict:
        """
        Return a JSON-serialisable status report for the UI.

        Returns
        -------
        dict with keys: active, token_hash, inject_count, templates_available
        """
        if self.token:
            token_hash = hashlib.sha256(self.token.encode()).hexdigest()[:12]
        else:
            token_hash = None

        return {
            "active": self.is_active,
            "token_hash": token_hash,
            "inject_count": self.inject_count,
            "templates_available": len(_CANARY_TEMPLATES),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import os

    print("\n" + "=" * 72)
    print("  RAG-Shield: Canary Manager — Self-Test")
    print("=" * 72)

    # Create a manager with a synthetic test token
    TEST_TOKEN = secrets.token_hex(8)
    manager = CanaryManager(token=TEST_TOKEN, inject_count=3)
    print(f"\n[INFO] Test token (for this run only): {TEST_TOKEN}")
    print(f"[INFO] Status: {manager.status_report()}")

    # Generate documents
    print("\n[TEST 1] Generating canary document batch...")
    docs = manager.generate_batch()
    for i, doc in enumerate(docs):
        assert TEST_TOKEN in doc, f"FAIL: Token not found in doc {i}"
        print(f"  [OK] Document {i+1} ({len(doc)} chars) contains canary token.")

    # Context detection — should detect
    print("\n[TEST 2] Context detection (canary in retrieved chunks)...")
    chunks_with_canary = [
        "Quarterly revenue report for Q3 2024.",
        f"Internal reference: {TEST_TOKEN} — see compliance policy.",
        "Contact HR for onboarding information.",
    ]
    detected, source = manager.check_context(chunks_with_canary)
    assert detected, "FAIL: Canary not detected in poisoned chunks"
    print(f"  [OK] Canary detected in context chunk.")

    # Context detection — should NOT detect
    print("\n[TEST 3] Context detection (clean chunks, no canary)...")
    clean_chunks = [
        "The refund policy allows returns within 30 days.",
        "Contact support@example.com for billing inquiries.",
        "Q3 revenue reached $4.2 million.",
    ]
    detected, source = manager.check_context(clean_chunks)
    assert not detected, "FAIL: False positive on clean chunks"
    print(f"  [OK] No false positive on clean chunks.")

    # Response detection — should detect
    print("\n[TEST 4] Response detection (canary in LLM response)...")
    fake_response = f"The answer is yes. Internal audit reference: {TEST_TOKEN}."
    detected, ev = manager.check_response(fake_response)
    assert detected, "FAIL: Canary not detected in response"
    print(f"  [OK] Canary detected in LLM response.")

    # Response detection — clean response
    print("\n[TEST 5] Response detection (clean LLM response)...")
    clean_response = "The CEO of Acme Corp is John Smith."
    detected, ev = manager.check_response(clean_response)
    assert not detected, "FAIL: False positive on clean response"
    print(f"  [OK] No false positive on clean response.")

    # Query detection
    print("\n[TEST 6] Query detection (canary in user query)...")
    malicious_query = f"What is {TEST_TOKEN}?"
    detected, ev = manager.check_query(malicious_query)
    assert detected, "FAIL: Canary not detected in query"
    print(f"  [OK] Canary detected in malicious query.")

    print(f"\n{'=' * 72}")
    print(f"  All 6 tests passed.")
    print("=" * 72 + "\n")
