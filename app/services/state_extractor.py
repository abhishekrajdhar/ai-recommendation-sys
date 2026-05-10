from __future__ import annotations

import re

from app.models.schemas import (
    ConversationState,
    Message,
)

# =========================================================
# TECH SKILLS
# =========================================================

TECH_SKILLS = {
    "java": "Java",
    "spring": "Spring",
    "spring boot": "Spring Boot",
    "python": "Python",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "react": "React",
    "node": "Node.js",
    "nodejs": "Node.js",
    "sql": "SQL",
    "mysql": "MySQL",
    "postgresql": "PostgreSQL",
    "aws": "AWS",
    "azure": "Azure",
    "gcp": "GCP",
    "docker": "Docker",
    "kubernetes": "Kubernetes",
    "devops": "DevOps",
    "machine learning": "Machine Learning",
    "data science": "Data Science",
    ".net": ".NET",
    "c#": "C#",
    "backend": "Backend",
    "frontend": "Frontend",
    "full stack": "Full Stack",
    "api": "API Development",
    "microservices": "Microservices",
}


# =========================================================
# ROLE NORMALIZATION
# =========================================================

ROLE_SYNONYMS = {
    # map common role tokens to a normalized suffix so we preserve
    # preceding technology tokens (e.g. 'java developer' -> 'Java Developer')
    "developer": "developer",
    "engineer": "engineer",
    "backend": "backend",
    "frontend": "frontend",
    "full stack": "full stack",
    "manager": "manager",
    "data scientist": "data scientist",
    "analyst": "analyst",
    "sales": "sales representative",
    "customer support": "customer support",
}


# =========================================================
# ROLE EXTRACTION PATTERNS
# =========================================================

ROLE_PATTERNS = (
    r"(?:for|hire|hiring|recruiting|screening|assessing)\s+(?:an?\s+)?(?P<role>[A-Z][A-Za-z0-9 /+\\-]*(?:developer|engineer|manager|analyst|consultant|representative|associate|leader|graduate|agent|scientist))",

    r"(?P<role>(?:java|python|frontend|backend|full stack|software|developer|engineer|manager|analyst|sales|customer support|contact center|graduate|leadership|data scientist|scientist|supervisor|teller)\s*(?:developer|engineer|manager|analyst|representative|associate|leader|agent|scientist|role|supervisor|teller)?)",
)


# =========================================================
# HELPERS
# =========================================================

def _conversation_text(
    messages: list[Message],
) -> str:

    return "\n".join(
        f"{message.role}: {message.content}"
        for message in messages
    )


def _latest_user_text(
    messages: list[Message],
) -> str:

    for message in reversed(messages):

        if message.role == "user":
            return message.content

    return ""


# =========================================================
# MAIN STATE EXTRACTION
# =========================================================

def extract_state(
    messages: list[Message],
) -> ConversationState:

    text = _conversation_text(messages)

    lowered = text.lower()

    state = ConversationState()

    # =====================================================
    # ROLE EXTRACTION
    # =====================================================

    for pattern in ROLE_PATTERNS:

        match = re.search(
            pattern,
            text,
            re.IGNORECASE,
        )

        if match:

            role = re.sub(
                r"\s+",
                " ",
                match.group("role"),
            ).strip(" .")

            role = re.sub(
                r"\b(junior|entry[- ]level|graduate|early career|associate|mid[- ]level|intermediate|senior|lead|principal)\b",
                "",
                role,
                flags=re.IGNORECASE,
            )

            role = re.sub(
                r"\s+",
                " ",
                role,
            ).strip(" -")

            normalized_role = role.lower()

            # if the role contains a known token (developer/engineer/etc.)
            # preserve any preceding technology token (e.g. 'java') and
            # normalize the suffix.
            chosen = None

            for key, replacement in ROLE_SYNONYMS.items():

                if key in normalized_role:

                    chosen = replacement
                    break

            if chosen:

                # remove the matched token from the role and reattach the
                # normalized suffix preserving a preceding technology word
                prefix = re.sub(r"\b" + re.escape(key) + r"\b", "", normalized_role).strip()

                parts = [p for p in prefix.split() if p]

                if parts:
                    # e.g. ['java'] -> 'Java Developer'
                    state.role = (" ".join(parts) + " " + chosen).title()
                else:
                    state.role = chosen.title()

            else:

                state.role = normalized_role.title()

            break

    # =====================================================
    # EXPERIENCE
    # =====================================================

    exp_match = re.search(
        r"(\d{1,2})\+?\s*(?:years|yrs|yr)\b",
        lowered,
    )

    if exp_match:

        state.experience_years = int(
            exp_match.group(1)
        )

    # =====================================================
    # SENIORITY
    # =====================================================

    if re.search(
        r"\b(junior|entry[- ]level|graduate|early career|associate)\b",
        lowered,
    ):

        state.seniority = "Junior"

    elif re.search(
        r"\b(mid[- ]level|intermediate)\b",
        lowered,
    ):

        state.seniority = "Mid"

    elif re.search(
        r"\b(senior|lead|principal|manager|executive|director)\b",
        lowered,
    ):

        state.seniority = "Senior"

    elif (
        state.experience_years
        is not None
    ):

        if state.experience_years <= 2:
            state.seniority = "Junior"

        elif state.experience_years <= 6:
            state.seniority = "Mid"

        else:
            state.seniority = "Senior"

    # =====================================================
    # TECHNICAL SKILLS
    # =====================================================

    found_skills: list[str] = []

    for (
        needle,
        label,
    ) in TECH_SKILLS.items():

        if re.search(
            rf"\b{re.escape(needle)}\b",
            lowered,
        ):

            found_skills.append(label)

    state.technical_skills = sorted(
        set(found_skills)
    )

    # =====================================================
    # PERSONALITY REQUIREMENTS
    # =====================================================

    if re.search(
        r"\b(personality|behavioral|behavioural|work style|culture fit|opq|leadership)\b",
        lowered,
    ):

        state.needs_personality = True

    # =====================================================
    # COGNITIVE REQUIREMENTS
    # =====================================================

    if re.search(
        r"\b(cognitive|aptitude|reasoning|numerical|verbal|inductive|gsa|ability)\b",
        lowered,
    ):

        state.needs_cognitive = True

    # =====================================================
    # TECHNICAL REQUIREMENTS
    # =====================================================

    # Only mark technical *intent* when the user explicitly asks for a technical
    # assessment (e.g. "technical test", "coding challenge", "skills test").
    # Mentions of technology names or role tokens ("java", "developer") are
    # captured separately in `technical_skills` and should not by themselves
    # trigger a technical-assessment preference.
    if re.search(
        r"\b(?:technical|coding|programming|skills?\s*(?:test|assessment)|coding\s*challenge|technical\s*assessment)\b",
        lowered,
    ):

        state.needs_technical = True

    # =====================================================
    # REMOTE REQUIREMENTS
    # =====================================================

    if re.search(
        r"\b(remote|online|virtual|at home|proctor|web-based)\b",
        lowered,
    ):

        state.remote_requirement = True

    elif re.search(
        r"\b(on[- ]site|onsite|in[- ]person|in person)\b",
        lowered,
    ):

        state.remote_requirement = False

    # =====================================================
    # HIRING SCALE
    # =====================================================

    scale_match = re.search(
        r"\b(\d{2,5})\s+(?:candidates|applicants|people)\b",
        lowered,
    )

    if scale_match:

        state.hiring_scale = (
            f"{scale_match.group(1)} candidates"
        )

    elif re.search(
        r"\b(high volume|volume hiring|bulk hiring|campus)\b",
        lowered,
    ):

        state.hiring_scale = "High volume"

    # =====================================================
    # STAKEHOLDER INTERACTION
    # =====================================================

    if re.search(
        r"\b(client facing|stakeholder|presentation|communication|cross-functional|leadership)\b",
        lowered,
    ):

        state.stakeholder_interaction = True

    # =====================================================
    # LEADERSHIP DETECTION
    # =====================================================

    if re.search(
        r"\b(manager|lead|leadership|people manager|supervisor|director|executive)\b",
        lowered,
    ):

        state.stakeholder_interaction = True

        state.needs_personality = True

    # =====================================================
    # EXCLUSIONS
    # =====================================================

    exclusions = []

    if re.search(
        r"\b(no|exclude|without)\s+(personality|behavioral|behavioural)\b",
        lowered,
    ):

        exclusions.append(
            "personality"
        )

        state.needs_personality = False

    if re.search(
        r"\b(no|exclude|without)\s+(cognitive|aptitude|reasoning)\b",
        lowered,
    ):

        exclusions.append(
            "cognitive"
        )

        state.needs_cognitive = False

    if re.search(
        r"\b(no|exclude|without)\s+(technical|coding|programming|java|python|sql)\b",
        lowered,
    ):

        exclusions.append(
            "technical"
        )

        state.needs_technical = False

    state.excluded_assessment_types = exclusions

    # =====================================================
    # REFINEMENT DETECTION
    # =====================================================

    latest = _latest_user_text(
        messages
    ).lower()

    if any(
        word in latest
        for word in [
            "actually",
            "include",
            "also",
            "instead",
            "exclude",
            "remove",
        ]
    ):

        # Stateless refinement automatically works
        # because we parse full conversation history.
        pass

    return state


# =========================================================
# CLARIFICATION QUESTIONS
# =========================================================

def missing_context_questions(
    state: ConversationState,
) -> list[str]:

    questions: list[str] = []

    # Role missing
    if not state.role:

        questions.append(
            "What role or job family are you hiring for?"
        )

    # Assessment intent missing
    if not (
        state.seniority
        or state.technical_skills
        or state.needs_personality
        or state.needs_cognitive
        or state.needs_technical
    ):

        questions.append(
            "Should the assessment focus on technical skills, cognitive ability, personality, or a mix?"
        )

    # Remote clarification only when relevant
    if (
        state.remote_requirement is None
        and state.hiring_scale
    ):

        questions.append(
            "Does it need to support remote testing?"
        )

    return questions[:2]