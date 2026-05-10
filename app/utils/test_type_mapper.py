def map_test_type(text: str) -> str:

    text = text.lower()

    technical_keywords = [
        "java",
        "python",
        "sql",
        ".net",
        "software",
        "coding",
        "technical",
        "programming",
        "developer",
        "engineer",
        "xaml",
        "mvc",
        "wpf",
        "cloud",
        "backend",
        "frontend",
    ]

    personality_keywords = [
        "personality",
        "behavior",
        "behaviour",
        "opq",
        "motivation",
        "leadership",
        "work style",
    ]

    cognitive_keywords = [
        "cognitive",
        "ability",
        "aptitude",
        "reasoning",
        "numerical",
        "verbal",
        "inductive",
        "gsa",
    ]

    situational_keywords = [
        "situational",
        "judgment",
        "judgement",
        "simulation",
    ]

    if any(
        k in text
        for k in technical_keywords
    ):
        return "K"

    if any(
        k in text
        for k in personality_keywords
    ):
        return "P"

    if any(
        k in text
        for k in cognitive_keywords
    ):
        return "A"

    if any(
        k in text
        for k in situational_keywords
    ):
        return "S"

    return "O"