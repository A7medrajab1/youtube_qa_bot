# test_openrouter.py
from src.llm_setup import get_llm

print("Testing OpenRouter with glm-4.5-air:free...")

llm = get_llm("openrouter", "z-ai/glm-4.5-air:free")

print("\n--- English ---")
print(llm.invoke("Say hello in 5 words.").content)

print("\n--- Arabic ---")
print(llm.invoke("قل مرحبا في 5 كلمات").content)

print("\n--- Arabic Religious/Historical ---")
print(llm.invoke("من هو خالد بن الوليد؟").content[:300])