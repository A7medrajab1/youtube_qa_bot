# test_llm.py
from src.llm_setup import get_llm

llm = get_llm()

print("Testing LLM via langchain-openai...")
try:
    response = llm.invoke("Say hi in exactly 5 words.")
    print("✅ SUCCESS")
    print(f"Content: {response.content}")
except Exception as e:
    print(f"❌ FAILED: {e}")