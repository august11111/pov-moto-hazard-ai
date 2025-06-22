#!/usr/bin/env python3
"""
optimize_prompts.py

Lit le fichier `results/results.jsonl` généré par votre pipeline
et génère :
  - des suggestions de prompt pour Pixtrale via dspy.LM
  - des suggestions de prompt pour Llama via HTTP Ollama

Usage :
    python optimize_prompts.py

Sortie :
  - dspy_exports/prompt_suggestions.json
"""
import os
import json
from pathlib import Path
from dotenv import load_dotenv

import dspy
from dspy.adapters import ChatAdapter
from dspy import LM
import requests


def load_results(path: Path) -> list[dict]:
    records = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def generate_with_pixtrale(lm: LM, examples: list[dict]) -> list[str]:
    """
    Utilise dspy.LM pour Pixtrale afin de générer des templates de prompt.
    """
    # instruction système et user
    system = (
        "Vous êtes un prompt engineer expert pour Pixtrale. "
        "Proposez des gabarits de prompt (inputs) contenant les placeholders "
        "DESCRIPTION_SCÈNE et LISTE_OBJETS."
    )
    user = (
        "Exemples (JSON) :\n"
        + json.dumps(examples, ensure_ascii=False, indent=2)
        + "\n\nRépondez UNIQUE MENT par une liste JSON de chaînes, sans explications."
    )

    # on force l’adapter chat
    dspy.configure(adapter=ChatAdapter())
    # appel LLM
    responses = lm(messages=[
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ])
    if not responses:
        return []

    raw = responses[0]
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
    except json.JSONDecodeError:
        pass

    # fallback ligne par ligne
    return [l.strip() for l in raw.splitlines() if l.strip()]


def generate_with_ollama(api_base: str, model: str, examples: list[dict]) -> list[str]:
    """
    Appel HTTP direct à Ollama pour générer des templates de prompt Llama.
    """
    url = api_base.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Vous êtes un prompt engineer expert pour Llama. "
                    "Proposez des gabarits de prompt (inputs) contenant les placeholders "
                    "DESCRIPTION_SCÈNE, LISTE_OBJETS et DESCRIPTION_ADVICE."
                )
            },
            {
                "role": "user",
                "content": (
                    "Exemples (JSON) :\n"
                    + json.dumps(examples, ensure_ascii=False, indent=2)
                    + "\n\nRépondez UNIQUEMENT par une liste JSON de chaînes, sans explications."
                )
            }
        ]
    }
    resp = requests.post(url, json=payload)
    resp.raise_for_status()
    body = resp.json()
    content = body["choices"][0]["message"]["content"]

    try:
        parsed = json.loads(content)
        if isinstance(parsed, list) and all(isinstance(x, str) for x in parsed):
            return parsed
    except json.JSONDecodeError:
        pass

    return [l.strip() for l in content.splitlines() if l.strip()]


def main():
    load_dotenv()

    # 1) charger les résultats
    results_path = Path("results/results.jsonl")
    if not results_path.exists():
        print("❌ Pas de results/results.jsonl trouvé.")
        return
    records = load_results(results_path)
    print(f"🔄 {len(records)} enregistrements chargés.")

    # 2) préparer les exemples
    pix_examples = [
        {"caption": r.get("caption", ""), "objects": r.get("objects", [])}
        for r in records
    ]
    llama_examples = [
        {
            "scene":  r.get("caption", ""),
            "objects": r.get("objects", []),
            "advice":  r.get("advice", "")
        }
        for r in records
    ]

    # 3) instancier LM Pixtrale
    pix_model = os.getenv("PIXTRALE_MODEL", "pixtral-large-latest")
    pix_api   = os.getenv("PIXTRALE_API_BASE", "https://api.mistral.ai/v1")
    pix_lm = LM(
        provider="openai",
        model=pix_model,
        api_base=pix_api,
        model_type="chat"
    )

    # 4) instancier paramètres Ollama
    ollama_api  = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
    llama_model = os.getenv("OLLAMA_MODEL", "llama3.2:latest")

    # 5) génération
    print("🚀 Génération Pixtrale…")
    pix_sugg = generate_with_pixtrale(pix_lm, pix_examples)
    print(f"   → {len(pix_sugg)} suggestions Pixtrale.")

    print("🚀 Génération Llama…")
    llama_sugg = generate_with_ollama(ollama_api, llama_model, llama_examples)
    print(f"   → {len(llama_sugg)} suggestions Llama.")

    # 6) sauvegarde
    out = Path("dspy_exports")
    out.mkdir(parents=True, exist_ok=True)
    out_file = out / "prompt_suggestions.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {"pixtrale_prompts": pix_sugg, "llama_prompts": llama_sugg},
            f,
            ensure_ascii=False,
            indent=2
        )
    print(f"✅ Suggestions sauvegardées dans {out_file}")


if __name__ == "__main__":
    main()
