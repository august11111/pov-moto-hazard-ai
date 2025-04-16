import subprocess

def ask_llama(prompt: str) -> str:
    """
    Envoie un prompt à LLaMA 3.2 via Ollama et récupère la réponse.
    """
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
