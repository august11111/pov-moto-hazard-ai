import subprocess

def ask_llama(prompt: str) -> str:
    process = subprocess.Popen(
        ["ollama", "run", "llama3.2"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    stdout, stderr = process.communicate(prompt)
    if process.returncode != 0:
        print("Erreur LLaMA :", stderr)
        return "Erreur LLaMA"
    return stdout.strip()
