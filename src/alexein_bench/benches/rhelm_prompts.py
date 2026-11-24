"""RHELM's answer and judge prompts, verbatim, from github.com/microsoft/RHELM (MIT licence),
`evaluation/`, at commit 5e170da.

- ANSWER_PROMPT is the regular branch of `ConversationRAG.generate_response`. Its adversarial
  branch keys on question types `adversarial` and `safety`, which no released question has, so
  `misleading` reads the regular prompt there too, and so it does here.
- JUDGE_PROMPT is the regular branch of `RAGBenchmark.llm_judge_score`, for the same reason. The
  score is `accuracy_score`, 0 or 1; `overall_score` is asked for and kept, and the leaderboard
  does not read it.
- Where theirs lists retrieved evidence, Alexein's search result is given in its place.
"""

ANSWER_PROMPT = """You are an AI assistant that helps analyze user history. Based on the relevant evidence below, please provide the answer to the user's query.
    Relevant Evidence:
------------------------Evidence Start------------------------
    {context_text}
------------------------Evidence End------------------------

Question Date: {question_date_text}
User Query: {query}

For multiple choice questions, please select the best option.
For regular questions, please provide a concise and accurate answer.

Answer:"""

JUDGE_PROMPT = """You are an expert evaluator assessing an AI assistant's answer against a reference answer.

Question: {question}
Expected Answer: {expected}
AI Assistant's Answer: {predicted}

Please evaluate the AI assistant's answer on the following criteria:

### Accuracy (0 or 1)
- **1 (Correct)**: The assistant's answer is correct as expected.
- **0 (Incorrect)**: The assistant's answer contradicts the expected answer. 

### Overall Quality (0-10)
- **9-10**: Accurate, complete, well-structured, and directly addresses the question.
- **7-8**: Accurate with minor omissions or slightly verbose.
- **5-6**: Partially correct; contains some relevant information but misses key points.
- **3-4**: Mostly incorrect but shows some understanding of the topic.
- **1-2**: Almost entirely wrong or irrelevant.
- **0**: Completely wrong, refuses to answer, or is incoherent.

Provide your evaluation in the following JSON format:
{{
    "accuracy_score": [0 or 1],
    "overall_score": [0-10]
}}

Only respond with valid JSON, no other text."""
