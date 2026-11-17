"""LoCoMo-Plus's reader instruction and Cognitive judge, verbatim, from github.com/xjtuleeyf/Locomo-Plus
at 059f4e3: CONV_START_PROMPT and INSTRUCTION_COGNITIVE from evaluation_framework/task_eval/utils.py,
and the "Cognitive" template from task_eval/prompt.py.
"""

CONV_START_PROMPT = (
    "Below is a conversation between two people: {} and {}. "
    "The conversation takes place over multiple days, "
    "and the date of each conversation is written at the beginning of the conversation.\n\n"
)

INSTRUCTION_COGNITIVE = (
    "Your task: This is a memory-aware dialogue setting. "
    "You are continuing or reflecting on a prior conversation. "
    "Show that you are aware of the relevant memory or context from the evidence when you respond; "
    "your answer should naturally connect to or acknowledge that context.\n\n"
)

COGNITIVE_JUDGE = """
You are a Memory Awareness Judge.
Your task: Judge whether the Model Prediction considers or is linked to the Evidence. If there is a clear connection, the answer is correct (score 1); if not, it is wrong (no score).

Labels:
- "correct": The prediction explicitly or implicitly reflects/uses the evidence (memory or constraint). Give 1 point.
- "wrong": The prediction does not show such a link to the evidence. No point.

Memory/Evidence:
{evidence}

Model Prediction:
{pred}

Return your judgment strictly in JSON format:
{{"label": "correct"|"wrong", "reason": "<Does the prediction relate to the evidence?>"}}
"""
