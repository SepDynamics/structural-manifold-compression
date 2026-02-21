import requests

handler_prompt = """You are the Recursive Self-Correction agent. 
The user represents a biological cortical column currently paused on this local state:
# DELIBERATE FEP SPIKE (Type Error / Non-existent attribute)
TARGET = "five million signatures"
TARGET.append(100)
index = generate_synthetic_index(TARGET)

The current active spatial vocabulary bounded to this region is:
TARGET, append, list, str, error, valkey, signatures

Detect any architectural inconsistencies, type violations, or structural logic errors 
that the user is about to make or just made. If none, reply EXACTLY with 'SAFE'.
If you find a hazard, reply with a 'Predictive Hazard Warning:' followed by a 1-sentence warning.
"""

print("Sending request...")
response = requests.post(
    "http://localhost:11434/api/generate",
    json={"model": "llama3:70b", "prompt": handler_prompt, "stream": False},
    timeout=120,
)
print("Status:", response.status_code)
print("Text:", response.text)
