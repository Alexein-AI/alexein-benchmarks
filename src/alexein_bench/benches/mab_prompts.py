"""MemoryAgentBench's reader prompts, verbatim, from github.com/HUST-AI-HYZ/MemoryAgentBench
(MIT licence): `utils/templates.py` (BASE_TEMPLATES, `rag_agent` queries) for the prompts and
each family's data config (`generation_max_length`) for the answer lengths, as read on
2026-10-05.

The odd spacing and typos are theirs and are kept: a prompt that differs measures a different
thing. Only the three families this harness runs are copied. Where their RAG agents put
retrieved "Memory i:" blocks ahead of the query, Alexein's search result is given in its place.
"""

SYSTEM = (
    "You are a helpful assistant that can read the context and memorize it for future retrieval."
)
RAG_QUERY = {
    "factconsolidation": "Pretend you are a knowledge management system. Each fact in the knowledge pool is provided with a serial number at the beginning, and the newer fact has larger serial number. \n You need to solve the conflicts of facts in the knowledge pool by finding the newest fact with larger serial number. You need to answer a question based on this rule. You should give a very concise answer without saying other words for the question **only** from the knowledge pool you have memorized rather than the real facts in real world. \n\nFor example:\n\n [Knowledge Pool] \n\n Question: Based on the provided Knowledge Pool, what is the name of the current president of Russia? \nAnswer: Donald Trump \n\n Now Answer the Question: Based on the provided Knowledge Pool, {question} \nAnswer:",
    "ruler_qa": "Answer the question based on the memorized documents. Only give me the answer and do not output any other words. \n\n Now Answer the Question: {question}",
    "detective_qa": "Based on the context you memorized, answer the question below. You are required to answer the question based on the strict output format.\n\n {question} \n\n",
}
# `generation_max_length` from each family's data config: the reader's max_tokens.
MAX_TOKENS = {
    "factconsolidation": 10,
    "ruler_qa": 50,
    "detective_qa": 2000,
}
# The agent configs of the paper's main table sample at 0.7.
TEMPERATURE = 0.7
