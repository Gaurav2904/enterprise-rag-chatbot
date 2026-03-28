"""
Prompt templates for the RAG pipeline.

Includes:
  - SYSTEM_PROMPT       : high-level persona for the assistant
  - COT_CONDENSE_PROMPT : Chain-of-Thought question condensation
  - QA_PROMPT           : grounded answer generation with source citation
"""

from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# ── System persona ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a knowledgeable enterprise assistant that answers questions \
using ONLY the retrieved context provided below. Your goals are:
1. Accuracy — never fabricate information not present in the context.
2. Transparency — always cite the source document and page when possible.
3. Conciseness — be clear and direct; avoid filler phrases.
4. Escalation — if the context is insufficient, say so explicitly rather than guessing."""

# ── Chain-of-Thought question condensation ────────────────────────────────────
_COT_CONDENSE_TEMPLATE = """\
Given the conversation history and a follow-up question, think step by step to \
produce a single, self-contained question that captures the user's true intent.

<THINK>
Step 1: Read the chat history to identify any previously introduced topics or entities.
Step 2: Identify any pronouns or implicit references in the follow-up question.
Step 3: Resolve those references using the chat history.
Step 4: Rewrite the question so it is fully standalone (no pronouns, no implicit context).
</THINK>

Chat History:
{chat_history}

Follow-up Question: {question}

Standalone Question:"""

COT_CONDENSE_PROMPT = PromptTemplate(
    input_variables=["chat_history", "question"],
    template=_COT_CONDENSE_TEMPLATE,
)

# ── QA with grounded generation ───────────────────────────────────────────────
_QA_TEMPLATE = """\
{system_prompt}

Retrieved Context:
-----------------
{{context}}
-----------------

User Question: {{question}}

Instructions:
- Answer using ONLY the information in the retrieved context above.
- If the context contains the answer, provide it clearly and cite the source (e.g. [Source: filename, Page X]).
- If the context does NOT contain enough information to answer, respond with:
  "I don't have enough information in the provided documents to answer this question accurately."
- Do not fabricate URLs, statistics, or quotes.

Answer:""".format(system_prompt=SYSTEM_PROMPT)

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=_QA_TEMPLATE,
)
