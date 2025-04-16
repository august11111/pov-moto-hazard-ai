import sys
import os
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from source.utils.llama_utils import ask_llama

input_csv = "test/data/descriptions/ytb_video_test.csv"
output_csv = "test/data/analyses/ytb_video_test_analyses.csv"


df = pd.read_csv(input_csv)

results = []

for idx, row in df.iterrows():
    description = row['description']
    
    prompt = f"""
Tu es un assistant expert en sécurité moto.
Analyse la situation suivante et donne des conseils pour éviter un accident.
il faut que tu prennes en comptes les voitures autour, les piétons si il y en a, les distances avec les véhicules.
Prends en compte la vitesse de la moto.
identifies des éventuels non respect du code de la route dans l'image.

Situation : {description}

Réponds de manière synthétique en 1 ou 2 phrases.
"""
    print(f"Frame {row['image']} → Analyse en cours...")
    analyse = ask_llama(prompt)
    print(f"Réponse : {analyse}\n")

    results.append({
        "image": row["image"],
        "description": description,
        "analyse_llama": analyse
    })

# Sauvegarde dans un CSV
output_df = pd.DataFrame(results)
os.makedirs(os.path.dirname(output_csv), exist_ok=True)
output_df.to_csv(output_csv, index=False)

print(f"\n✅ Analyse terminée et sauvegardée dans {output_csv}")
