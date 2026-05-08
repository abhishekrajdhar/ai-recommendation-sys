from __future__ import annotations

import re

from app.models.schemas import ConversationState, Message


TECH_SKILLS = {
    "java": "Java",
    "spring": "Spring",
    "python": "Python",
    "javascript": "JavaScript",
    "typescript": "TypeScript",
    "react": "React",
    "sql": "SQL",
    "aws": "AWS",
    "azure": "Azure",
    "devops": "DevOps",
    "data science": "Data Science",
    "machine learning": "Machine Learning",
    "salesforce": "Salesforce",
}

ROLE_PATTERNS = (
    r"(?:for|hire|hiring|recruiting|screening|assessing)\s+(?:an?\s+)?(?P<role>[A-Z][A-Za-z0-9 /+\-]*(?:developer|engineer|manager|analyst|consultant|representative|associate|leader|graduate|agent))",
    r"(?P<role>(?:java|python|frontend|front end|backend|back end|full stack|sales|customer service|contact center|graduate|manager|leadership|data)\s+(?:developer|engineer|manager|analyst|representative|associate|leader|agent|role))",
)


def _conversation_text(messages: list[Message]) -> str:
    return "\n".join(f"{message.role}: {message.content}" for message in messages)


def _latest_user_text(messages: list[Message]) -> str:
    for message in reversed(messages):
        if message.role == "user":
            return message.content
    return ""


def extract_state(messages: list[Message]) -> ConversationState:
    text = _conversation_text(messages)
    lowered = text.lower()
    state = ConversationState()

    for pattern in ROLE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            role = re.sub(r"\s+", " ", match.group("role")).strip(" .")
            role = re.sub(r"\b(junior|entry[- ]level|graduate|early career|associate|mid[- ]level|intermediate|senior|lead|principal)\b", "", role, flags=re.IGNORECASE)
            role = re.sub(r"\s+", " ", role).strip(" -")
            state.role = role.title()
            break

    exp_match = re.search(r"(\d{1,2})\+?\s*(?:years|yrs|yr)\b", lowered)
    if exp_match:
        state.experience_years = int(exp_match.group(1))

    if re.search(r"\b(junior|entry[- ]level|graduate|early career|associate)\b", lowered):
        state.seniority = "Junior"
    elif re.search(r"\b(mid[- ]level|intermediate)\b", lowered):
        state.seniority = "Mid"
    elif re.search(r"\b(senior|lead|principal|manager|executive|director)\b", lowered):
        state.seniority = "Senior"
    elif state.experience_years is not None:
        if state.experience_years <= 2:
            state.seniority = "Junior"
        elif state.experience_years <= 6:
            state.seniority = "Mid"
        else:
            state.seniority = "Senior"

    found_skills: list[str] = []
    for needle, label in TECH_SKILLS.items():
        if re.search(rf"\b{re.escape(needle)}\b", lowered):
            found_skills.append(label)
    state.technical_skills = sorted(set(found_skills))

    if re.search(r"\b(personality|behavioral|behavioural|work style|culture fit|opq)\b", lowered):
        state.needs_personality = True
    if re.search(r"\b(cognitive|aptitude|reasoning|numerical|verbal|inductive|gsa|ability)\b", lowered):
        state.needs_cognitive = True
    if re.search(r"\b(technical|coding|programming|skills? test|java|python|sql|developer)\b", lowered):
        state.needs_technical = True

    if re.search(r"\b(remote|online|virtual|at home|proctor)\b", lowered):
        state.remote_requirement = True

    scale_match = re.search(r"\b(\d{2,5})\s+(?:candidates|applicants|people)\b", lowered)
    if scale_match:
        state.hiring_scale = f"{scale_match.group(1)} candidates"
    elif re.search(r"\b(high volume|volume hiring|bulk hiring|campus)\b", lowered):
        state.hiring_scale = "High volume"

    if re.search(r"\b(client facing|stakeholder|presentation|communication)\b", lowered):
        state.stakeholder_interaction = True

    exclusions = []
    if re.search(r"\b(no|exclude|without)\s+(personality|behavioral|behavioural)\b", lowered):
        exclusions.append("personality")
        state.needs_personality = False
    if re.search(r"\b(no|exclude|without)\s+(cognitive|aptitude|reasoning)\b", lowered):
        exclusions.append("cognitive")
        state.needs_cognitive = False
    state.excluded_assessment_types = exclusions

    latest = _latest_user_text(messages).lower()
    if any(word in latest for word in ("actually", "include", "also", "instead", "exclude")):
        # Refinements are already reflected because extraction uses full stateless history.
        pass

    return state


def missing_context_questions(state: ConversationState) -> list[str]:
    questions: list[str] = []
    if not state.role:
        questions.append("What role or job family are you hiring for?")
    if not (state.seniority or state.technical_skills or state.needs_personality or state.needs_cognitive or state.needs_technical):
        questions.append("Should the assessment focus on technical skills, cognitive ability, personality, or a mix?")
    if state.remote_requirement is None:
        questions.append("Does it need to support remote testing?")
    return questions[:2]
