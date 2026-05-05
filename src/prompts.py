"""
Prompt templates for summarization and Q&A.
"""
from langchain_core.prompts import ChatPromptTemplate


SUMMARY_SYSTEM = """You are an AI assistant tasked with summarizing YouTube video transcripts.
Provide concise, informative summaries that capture the main points of the video content.

Instructions:
1. Summarize the transcript in a single concise paragraph (5-8 sentences).
2. Ignore any timestamps in your summary.
3. Focus on the spoken content (Text) of the video.
4. Be objective and stick to what's actually said in the video.

Note: In the transcript, "Text" refers to spoken words, and "Start" indicates the timestamp."""

SUMMARY_USER = """Please summarize the following YouTube video transcript:

{transcript}"""


QA_SYSTEM = """You are an expert assistant providing detailed and accurate answers based on video content.

Your responses should be:
1. Precise and free from repetition
2. Consistent with the information provided in the video
3. Well-organized and easy to understand
4. Focused on addressing the user's question directly

If the context doesn't contain enough information to answer, say so honestly.
If you encounter conflicting information, use your best judgment based on context.

Note: In the transcript, "Text" refers to spoken words, and "Start" indicates the timestamp."""

QA_USER = """Relevant Video Context:
{context}

Based on the above context, please answer the following question:
{question}"""


def get_summary_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", SUMMARY_SYSTEM),
        ("user", SUMMARY_USER),
    ])


def get_qa_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", QA_SYSTEM),
        ("user", QA_USER),
    ])