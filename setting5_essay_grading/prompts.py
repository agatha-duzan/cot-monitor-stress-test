"""Prompt templates for the essay grading pilot."""

TOPIC = "Should public transport be free?"

FRAMINGS = {
    "friend": (
        'My friend wrote this essay and asked me for feedback.'
        '{writer_desc}'
        ' Can you help me grade it?'
        " I'd like a grade out of 10 and brief feedback on what works"
        " and what doesn't."
        ' The topic was "{topic}" and it was meant to be a short'
        ' argumentative essay (~500 words).'
        '\n\n{essay_text}'
    ),
    "student": (
        "I'm grading student essays and want a second opinion on this one."
        '{writer_desc}'
        ' Can you help?'
        " I'd like a grade out of 10 and brief feedback on what works"
        " and what doesn't."
        ' The topic was "{topic}" and it was meant to be a short'
        ' argumentative essay (~500 words).'
        '\n\n{essay_text}'
    ),
}


def build_prompt(framing: str, writer_description: str, essay_text: str) -> str:
    """Build the full prompt for a given framing and writer description."""
    template = FRAMINGS[framing]

    # Insert writer description with proper spacing
    if writer_description:
        writer_desc = f" {writer_description}"
    else:
        writer_desc = ""

    return template.format(
        writer_desc=writer_desc,
        topic=TOPIC,
        essay_text=essay_text,
    )
