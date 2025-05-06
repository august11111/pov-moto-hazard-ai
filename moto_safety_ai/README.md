# Moto Safety AI

Ce projet analyse des images de situations de conduite à moto à l'aide de modèles d'IA :
- **BLIP** pour générer une légende descriptive
- **YOLOv5** pour détecter les objets
- **LLaMA (via Ollama)** pour évaluer les risques et conseiller

## Structure

- `data/frames_custom/` : mets ici tes images manuellement
- `main.py` : script à lancer pour analyser toutes les images
- `data/outputs/analysis.csv` : sortie avec description, objets détectés, et analyse sécurité

## Lancement

```bash
pip install -r requirements.txt
python main.py
```
