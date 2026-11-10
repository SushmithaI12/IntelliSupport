import re

SMALLTALK = re.compile(
    r"^\s*(ok+(ay|y)?|k|thanks?( you| u| a lot)?|thank u|thx|ty|hi+|hello|hey|bye|goodbye|great|cool|nice|"
    r"got it|that'?s all|no thanks?|nothing( else)?|good (morning|afternoon|evening|night)|"
    r"ok(ay)? thanks?)\W*$",
    re.I,
)


def smalltalk_reply(text: str) -> str:
    t = text.lower()
    if re.search(r"no thanks?|that'?s all|nothing|bye|goodbye|good night", t):
        return "Alright! Have a great day. I'm here if you need anything else."
    if "thank" in t or "thx" in t or t.strip() in ("ty",):
        return "You're welcome! Is there anything else I can help you with?"
    if re.search(r"\b(hi+|hello|hey|good)\b", t):
        return "Hello! How can I help you today?"
    return "Okay! Let me know if you need anything else."