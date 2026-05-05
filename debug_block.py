"""
Debug why AgentRouter is blocking our requests.
Tests progressively from simple to complex.
"""
import uuid
import requests
import json
from config import AGENT_ROUTER_API_KEY, AGENT_ROUTER_BASE_URL, LLM_MODEL_NAME
from src.pipeline import YouTubeBot


URL = f"{AGENT_ROUTER_BASE_URL}/chat/completions"

HEADERS = {
    "Authorization": f"Bearer {AGENT_ROUTER_API_KEY}",
    "Content-Type": "application/json",
    "User-Agent": "codex_cli_rs/0.21.0 (Linux; x86_64) Cursor/1.0.0",
    "OpenAI-Beta": "responses=experimental",
    "originator": "codex_cli_rs",
    "session_id": str(uuid.uuid4()),
    "version": "0.21.0",
}


def call_llm(messages: list, label: str):
    """Make a raw call and print result."""
    print(f"\n{'='*70}")
    print(f"TEST: {label}")
    print(f"{'='*70}")
    
    payload = {
        "model": LLM_MODEL_NAME,
        "messages": messages,
        "max_tokens": 500,
    }
    
    try:
        response = requests.post(URL, headers=HEADERS, json=payload, timeout=60)
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            content = data["choices"][0]["message"].get("content", "")
            print(f"✅ Response: {content[:300]}")
        else:
            print(f"❌ Response: {response.text[:300]}")
    except Exception as e:
        print(f"❌ Exception: {e}")


# ===== TEST 1: Simple English =====
call_llm(
    [{"role": "user", "content": "Say hello in 5 words."}],
    "Simple English greeting"
)

# ===== TEST 2: Simple Arabic =====
call_llm(
    [{"role": "user", "content": "قل مرحبا في 5 كلمات"}],
    "Simple Arabic greeting"
)

# ===== TEST 3: Religious topic in English =====
call_llm(
    [{"role": "user", "content": "Who was Prophet Muhammad?"}],
    "Religious question (English)"
)

# ===== TEST 4: Religious topic in Arabic =====
call_llm(
    [{"role": "user", "content": "من هو النبي محمد؟"}],
    "Religious question (Arabic)"
)

# ===== TEST 5: Specific Khalid question =====
call_llm(
    [{"role": "user", "content": "من هو خالد بن الوليد؟"}],
    "Khalid ibn al-Walid question"
)

# ===== TEST 6: With actual chunks from the video =====
print(f"\n{'='*70}")
print("TEST 6: Loading actual video and checking chunks")
print(f"{'='*70}")

# Replace with the URL you've been testing
VIDEO_URL = input("\n📺 Enter the YouTube URL you've been testing: ").strip()

if VIDEO_URL:
    bot = YouTubeBot()
    print("\n⏳ Loading video...")
    status = bot.load_video(VIDEO_URL)
    print(status)
    
    question = "من هي مرضعة محمد؟"
    
    # Get chunks
    docs = bot.faiss_index.similarity_search(question, k=5)
    print(f"\n📦 Retrieved {len(docs)} chunks for: '{question}'\n")
    
    for i, doc in enumerate(docs, 1):
        print(f"--- Chunk {i} ---")
        print(doc.page_content[:300])
        print()
    
    # Try to call LLM with these chunks
    context = "\n\n".join(doc.page_content for doc in docs)
    
    call_llm(
        [
            {"role": "system", "content": "Answer based on the context."},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        ],
        "Real Q&A with retrieved chunks"
    )