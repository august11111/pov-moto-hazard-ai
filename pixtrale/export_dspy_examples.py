# export_dspy_examples.py

import json
import pandas as pd
from pathlib import Path

def load_results_jsonl(path: Path) -> list[dict]:
    """
    Lit chaque ligne de `results/results.jsonl` (JSONL) et renvoie une liste de dictionnaires.
    Chaque dictionnaire correspond à un record produit par MotoAdvisor.
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                records.append(rec)
            except json.JSONDecodeError:
                # Ignore les lignes mal formées
                continue
    return records

def export_all(records: list[dict], output_dir: Path):
    """
    À partir de la liste de records, extrait et écrit :
      - dspy_captions.csv / dspy_captions.json
      - dspy_advices.csv  / dspy_advices.json
    """
    captions = []
    advices = []

    for rec in records:
        # --- Captions (Pixtrale) ---
        captions.append({
            "frame_id":       rec.get("frame_id", ""),
            "timestamp":      rec.get("timestamp", ""),
            "prompt_variant": rec.get("caption_variant", ""),
            "caption_text":   rec.get("caption_text", "")
        })

        # --- Advices (LLaMA) ---
        for adv in rec.get("advice_variants", []):
            advices.append({
                "frame_id":         rec.get("frame_id", ""),
                "timestamp":        rec.get("timestamp", ""),
                "scene_description": rec.get("caption_text", ""),
                "objects_json":     json.dumps(rec.get("objects", []), ensure_ascii=False),
                "prompt_variant":   adv.get("variant", ""),
                "advice_text":      adv.get("advice", "")
            })

    # Créer le dossier d’export si nécessaire
    output_dir.mkdir(parents=True, exist_ok=True)

    # Exporter captions
    df_caps = pd.DataFrame(captions)
    csv_caps = output_dir / "dspy_captions.csv"
    json_caps = output_dir / "dspy_captions.json"
    df_caps.to_csv(csv_caps, index=False, encoding="utf-8")
    with open(json_caps, "w", encoding="utf-8") as f:
        json.dump(captions, f, indent=2, ensure_ascii=False)

    print(f"✅ Exported {len(captions)} caption rows to:\n  • {csv_caps}\n  • {json_caps}")

    # Exporter advices
    df_adv = pd.DataFrame(advices)
    csv_adv = output_dir / "dspy_advices.csv"
    json_adv = output_dir / "dspy_advices.json"
    df_adv.to_csv(csv_adv, index=False, encoding="utf-8")
    with open(json_adv, "w", encoding="utf-8") as f:
        json.dump(advices, f, indent=2, ensure_ascii=False)

    print(f"✅ Exported {len(advices)} advice rows to:\n  • {csv_adv}\n  • {json_adv}")

def main():
    results_path = Path("results") / "results.jsonl"
    if not results_path.exists():
        print("❌ Cannot find results/results.jsonl. Run moto_advice_pipeline.py first.")
        return

    print("🔄 Loading `results/results.jsonl` …")
    records = load_results_jsonl(results_path)
    print(f"   • Loaded {len(records)} records.")

    output_dir = Path("dspy_logs")
    export_all(records, output_dir)
    print("🏁 All data exported successfully into 'dspy_logs/'.")

if __name__ == "__main__":
    main()
