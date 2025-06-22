import dspy

print("✅ dspy version:", dspy.__version__)
print("🧪 Available symbols:")
print(" - Run:", hasattr(dspy, "Run"))
print(" - Signature:", hasattr(dspy, "Signature"))
print(" - InputField:", hasattr(dspy, "InputField"))
print(" - OutputField:", hasattr(dspy, "OutputField"))
print(" - Example:", hasattr(dspy, "Example"))
