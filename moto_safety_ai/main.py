from my_utils.blip_utils import generate_captions
from my_utils.yolo_utils import detect_objects_yolo
from my_utils.llama_utils import ask_llama
import os
import pandas as pd

input_dir = "data/frames_custom"
output_csv = "data/outputs/analysis.csv"

captions = generate_captions(input_dir)
results = []

for fname, caption in captions:
    img_path = os.path.join(input_dir, fname)
    objects = detect_objects_yolo(img_path)
    object_str = ", ".join(set(objects))

    prompt = f"""
Tu es un expert en sécurité routière spécialisé en conduite à moto.  
Voici une scène capturée depuis une caméra embarquée sur une moto.

Description visuelle : {caption}
Objets détectés automatiquement : {", ".join(objects)}

Ta mission :
- Analyse cette situation comme si tu étais un copilote de sécurité.
- Identifie uniquement les **dangers visibles ou fortement probables**, en te basant sur les éléments ci-dessus.
- Ne donne **aucun conseil générique** (ex : "porte un casque", "sois prudent").
- Si la situation ne présente **aucun danger clair**, réponds simplement : "Rien à signaler pour cette scène."

Ta réponse doit être brève (1 à 2 phrases), spécifique et exploitable par le pilote en temps réel.

"""

    analyse = ask_llama(prompt)

    results.append({
        "image": fname,
        "description": caption,
        "objects_detected": object_str,
        "analyse_llama": analyse
    })

os.makedirs(os.path.dirname(output_csv), exist_ok=True)
pd.DataFrame(results).to_csv(output_csv, index=False)
print(f"✅ Résultats sauvegardés dans {output_csv}")
