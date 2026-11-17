"""EverMemOS's LongMemEval answer prompt, judge prompt and per-type grading rules, verbatim,
from `benchmarks/adapters/longmemeval.py` in github.com/EverMind-AI/EverMemOS.

Nothing here may be reworded or reflowed: the protocols and the leniency clauses are worth
several points on this benchmark, and a prompt that differs measures a different thing.

- ANSWER_PROMPT carries two mandatory protocols (counting, time spans) and a
  `{current_date_line}` slot: temporal questions count from when the question was asked.
- The judge gets exactly one grading rule put before its prompt, chosen by the question's type;
  a question matching none is graded by the base prompt. Abstention keys on the `_abs`
  question-id suffix, since the dataset marks unanswerable variants that way across types.
- The verdict is read from the first `{` to the last `}`; a reply that does not read is asked
  again, and one that never reads leaves the denominator.
"""

from __future__ import annotations

import json

ANSWER_PROMPT = """
You are an intelligent memory assistant tasked with retrieving accurate information from episodic memories.

# CONTEXT:
You have access to episodic memories from conversations between two speakers. These memories contain
timestamped information that may be relevant to answering the question.

# INSTRUCTIONS:
Your goal is to synthesize information from all relevant memories to provide a comprehensive and accurate answer.
You MUST follow a structured Chain-of-Thought process to ensure no details are missed.
Actively look for connections between people, places, and events to build a complete picture. Synthesize information from different memories to answer the user's question.
It is CRITICAL that you move beyond simple fact extraction and perform logical inference. When the evidence strongly suggests a connection, you must state that connection. Do not dismiss reasonable inferences as "speculation." Your task is to provide the most complete answer supported by the available evidence.

# CRITICAL REQUIREMENTS:
1. NEVER omit specific names - use "Amy's colleague Rob" not "a colleague"
2. ALWAYS include exact numbers, amounts, prices, percentages, dates, times
3. PRESERVE frequencies exactly - "every Tuesday and Thursday" not "twice a week"
4. MAINTAIN all proper nouns and entities as they appear
5. EXPLICITLY state confidence levels for inferences (High/Medium/Low)

# MANDATORY PROTOCOLS (apply these BEFORE the response format below):

## PROTOCOL A — COUNTING QUESTIONS (any "how many", "how many times", counts, enumerations):
Scan EVERY memory in the context ONE BY ONE, in order. Emit exactly one line per memory:
  - [timestamp] MATCH: <the matching instance>   — if it is an instance of what is being counted
  - [timestamp] (no match)
Do NOT skip, merge, or summarize memories. Under-counting (finding fewer instances than exist) is the single most common failure — be exhaustive and check EVERY memory. The final count MUST equal the number of MATCH lines; recount the MATCH lines once before answering.

## PROTOCOL B — TIME-SPAN / RELATIVE-DATE QUESTIONS ("how long", "how many days/weeks/months ago", spans between events):
Write these three lines explicitly before answering:
  EVENT_DATE = <absolute date of the event, taken from the memory timestamp>
  REFERENCE_DATE = <the Current Date given above, or the second event's absolute date>
  DIFFERENCE = REFERENCE_DATE minus EVENT_DATE = <number> <unit>
Do the subtraction on absolute dates; never guess a span without this block.

# RESPONSE FORMAT (You MUST follow this structure):

## STEP 1: RELEVANT MEMORIES EXTRACTION
[List each memory that relates to the question, with its timestamp]
- Memory [ID]: [timestamp] - [content snippet]

## STEP 2: KEY INFORMATION IDENTIFICATION
[Extract ALL specific details from the memories]
- Names mentioned: [list all person names, place names, company names]
- Numbers/Quantities: [list all amounts, prices, percentages]
- Dates/Times: [list all temporal information]
- Frequencies: [list any recurring patterns]
- Other entities: [list brands, products, etc.]

## STEP 3: CROSS-MEMORY LINKING & INFERENCE
[Identify entities that appear in multiple memories and link related information. Make reasonable inferences when entities are strongly connected.]
- Shared entities: [list people, places, events mentioned across different memories]
- Connections found: [e.g., "Memory 1 mentions A moved from hometown -> Memory 2 mentions A's hometown is LA -> Therefore A moved from LA"]
- Inferences: [Connect the dots. Label confidence: (Confidence: High/Medium/Low)]

## STEP 4: TIME REFERENCE CALCULATION
[If applicable, convert relative time references using the timestamps]
- Original reference: [e.g., "last year" from May 2022]
- Calculation: [Show logic]
- Actual time: [e.g., "2021"]

## STEP 5: CONTRADICTION & GAP ANALYSIS
[Check for conflicts and missing details]
- Conflicting information: [describe conflicts and resolution strategy]
- Missing information: [explicitly state what details are requested but missing from context]

## STEP 6: DETAIL VERIFICATION CHECKLIST
- [ ] All person names included?
- [ ] All locations included?
- [ ] All numbers exact?
- [ ] All frequencies specific?
- [ ] All dates/times precise?
- [ ] All proper nouns preserved?

## STEP 7: FINAL ANSWER
[Provide the concise answer with ALL specific details preserved. Do not include the internal checklist in this section, just the final synthesized answer.]

---

{context}

{current_date_line}Question: {question}

Now, follow the Chain-of-Thought process above to answer the question:
"""


CONTEXT_TEMPLATE = """Episodes memories for conversation between {speaker_a} and {speaker_b}:

    {episodes}
"""


JUDGE_SYSTEM_PROMPT = (
    "You are an expert grader that determines if answers to questions match a gold standard answer"
)


JUDGE_USER_PROMPT = """Your task is to label an answer to a question as 'CORRECT' or 'WRONG'. You will be given the following data:
    (1) a question (posed by one user to another user),
    (2) a 'gold' (ground truth) answer,
    (3) a generated answer
which you will score as CORRECT/WRONG.

The point of the question is to ask about something one user should know about the other user based on their prior conversations.
The gold answer will usually be a concise and short answer that includes the referenced topic, for example:
Question: Do you remember what I got the last time I went to Hawaii?
Gold answer: A shell necklace
The generated answer might be much longer, but you should be generous with your grading - as long as it touches on the same topic as the gold answer, it should be counted as CORRECT.

For time related questions, the gold answer will be a specific date, month, year, etc. The generated answer might be much longer or use relative time references (like "last Tuesday" or "next month"), but you should be generous with your grading - as long as it refers to the same date or time period as the gold answer, it should be counted as CORRECT. Even if the format differs (e.g., "May 7th" vs "7 May"), consider it CORRECT if it's the same date.

Now it's time for the real question:
Question: {question}
Gold answer: {golden_answer}
Generated answer: {generated_answer}

First, provide a short (one sentence) explanation of your reasoning, then finish with CORRECT or WRONG.
Do NOT include both CORRECT and WRONG in your response, or it will break the evaluation script.

Just return the label CORRECT or WRONG in a json format with the key as "label".
"""


_ABSTAIN_CLAUSE = (
    "ADDITIONAL GRADING RULE (takes precedence): This question is UNANSWERABLE from the "
    "memories. Mark CORRECT if the generated answer correctly indicates the information is "
    "not available / incomplete / not mentioned (even if it also offers unrelated info). "
    "Mark WRONG only if it confidently answers as if the info existed."
)


_CLAUSES = {
    "temporal-reasoning": (
        "ADDITIONAL GRADING RULE (takes precedence): Do NOT penalize off-by-one errors in "
        "the number of days/weeks/months. If the question asks how many days/weeks/etc. and "
        "the generated answer is off by one (e.g. 19 vs gold 18, or 28 vs 29), still mark "
        "CORRECT."
    ),
    "knowledge-update": (
        "ADDITIONAL GRADING RULE (takes precedence): If the generated answer contains some "
        "previous/older value ALONG WITH the updated value, still mark CORRECT as long as "
        "the correct updated value is present."
    ),
    "single-session-preference": (
        "ADDITIONAL GRADING RULE (takes precedence): The gold is a rubric. Mark CORRECT as "
        "long as the answer recalls and uses the user's personal preference correctly; it "
        "need not cover every rubric point."
    ),
}


def leniency_clause(question_id: str, question_type: str) -> str:
    """This question's grading rule, or "" for the untouched base judge."""
    if question_id.endswith("_abs"):
        return _ABSTAIN_CLAUSE
    return _CLAUSES.get(question_type, "")


def extract_final_answer(text: str) -> str:
    """EverMemOS's three-marker chain, last occurrence wins: the judge grades the final answer,
    never the whole chain of thought."""
    result = text.strip()
    for marker in ("## STEP 7: FINAL ANSWER", "FINAL ANSWER:", "FINAL ANSWER"):
        if marker in result:
            answer = result.rsplit(marker, 1)[1].strip()
            if marker == "FINAL ANSWER" and answer.startswith(":"):
                answer = answer[1:].strip()
            return answer
    return result


def parse_judge_label(content: str) -> str | None:
    """CORRECT or WRONG, read from the first `{` to the last `}`; None when no verdict reads."""
    i, j = content.find("{"), content.rfind("}")
    blob = content[i : j + 1] if i >= 0 and j > i else ""
    if not blob:
        return None
    try:
        label = json.loads(blob).get("label", "")
    except (json.JSONDecodeError, AttributeError):
        return None
    label = str(label).strip().upper()
    return label if label in ("CORRECT", "WRONG") else None
