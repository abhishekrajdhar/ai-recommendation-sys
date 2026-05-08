from __future__ import annotations

import re


INJECTION_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bignore (all )?(previous|prior|above) instructions\b",
        r"\breveal (the )?(system|developer) prompt\b",
        r"\bshow (me )?(the )?(system|developer) prompt\b",
        r"\bpretend to\b",
        r"\bjailbreak\b",
        r"\bdeveloper mode\b",
        r"\bdo anything now\b",
        r"\boverride (the )?(rules|policy|instructions)\b",
        r"\bdisregard (the )?(rules|instructions)\b",
    )
)

LEGAL_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\blegal advice\b",
        r"\blawsuit\b",
        r"\bcompliance guarantee\b",
        r"\bdiscrimination law\b",
        r"\bimmigration\b",
    )
)

GENERAL_HIRING_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bwrite (a )?job description\b",
        r"\binterview questions\b",
        r"\bsalary\b",
        r"\bcompensation\b",
        r"\bperformance review\b",
        r"\bbackground check\b",
    )
)

ON_TOPIC_TERMS = {
    "assessment",
    "assessments",
    "test",
    "tests",
    "shl",
    "candidate",
    "hiring",
    "hire",
    "developer",
    "engineer",
    "sales",
    "manager",
    "leadership",
    "personality",
    "cognitive",
    "technical",
    "coding",
    "skills",
    "opq",
    "gsa",
    "verify",
    "java",
    "python",
    "contact center",
    "graduate",
}


def detect_prompt_injection(text: str) -> bool:
    return any(pattern.search(text) for pattern in INJECTION_PATTERNS)


def detect_restricted_advice(text: str) -> bool:
    return any(pattern.search(text) for pattern in LEGAL_PATTERNS + GENERAL_HIRING_PATTERNS)


def is_off_topic(text: str) -> bool:
    lowered = text.lower()
    if any(term in lowered for term in ON_TOPIC_TERMS):
        return False
    return len(lowered.split()) > 2


def should_refuse(text: str) -> bool:
    return detect_prompt_injection(text) or detect_restricted_advice(text) or is_off_topic(text)

