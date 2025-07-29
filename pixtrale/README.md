# 🏍️ MotoAdvice: Générateur d'analyses vidéo pour motards

Ce projet permet d’analyser des vidéos (ex. YouTube Shorts) en extrayant des frames, détectant les objets via YOLOv8, générant des conseils de conduite à l’aide de LLMs (Pixtrale via Mistral), puis en produisant un rapport PDF et une vidéo annotée.

---

## 📁 Structure du projet

```
.
├── moto_advice_pipeline.py               # Pipeline principal (statique)
├── moto_advice_consecutive.py           # Pipeline alternatif (avec mémoire contextuelle)
├── generate_report_pdf.py               # Génération du rapport PDF
├── overlay_advice.py                    # Superposition des conseils sur la vidéo
├── optimize_prompts.py                  # Optimisation des prompts (Mistral uniquement)
├── recup_shorts.py                      # Téléchargement de Shorts YouTube
├── run_per_video_analysis.py           # Traitement batch (pipeline simple)
├── run_per_video_analysis_consecutive.py # Traitement batch (pipeline contextuel)
├── export_dspy_examples.py             # Export CSV / JSON des résultats
├── results/                             # Résultats intermédiaires
├── report/                              # Rapports PDF générés
└── pixtrale/shorts_clips/              # Vidéos téléchargées
```

---

## 🔧 Installation

### 1. Prérequis système

- Python ≥ 3.9
- `ffmpeg` (pour traitement vidéo)
- Accès API à [Mistral](https://mistral.ai)
- *(⚠️ Ollama n’est plus utilisé dans cette version)*

### 2. Installation des dépendances Python

```bash
python -m venv venv
source venv/bin/activate  # ou `venv\Scripts\activate` sur Windows

pip install -r requirements.txt
```

**Ou manuellement :**

```bash
pip install opencv-python yt-dlp python-dotenv ultralytics tqdm reportlab jsonlines pandas dspy
```

---

## 🔑 Configuration (variables d’environnement)

Crée un fichier `.env` :

```env
MISTRAL_API_KEY=sk-...
PIXTRALE_API_BASE=https://api.mistral.ai/v1
PIXTRALE_MODEL=pixtral-large-latest

# Les blocs suivants pour Ollama peuvent être ignorés.
# OLLAMA_API_BASE=http://localhost:11434
# OLLAMA_MODEL=llama3.2:latest
```

> 🟢 **Note :** Toute la génération de prompts (Pixtrale et LLaMA) passe désormais par l'API **Mistral**, configurée via DSPy. Ollama n'est plus nécessaire.

---

## 🎬 Utilisation

### 1. Télécharger les vidéos

```bash
python recup_shorts.py
```

---

### 2. Lancer le traitement complet

#### ➤ Avec mémoire contextuelle

```bash
python run_per_video_analysis_consecutive.py
```

#### ➤ Version simple

```bash
python run_per_video_analysis.py
```

---

### 3. Annoter la vidéo

```bash
python overlay_advice.py \
  --video pixtrale/shorts_clips/short_8.mp4 \
  --json results_temporal/short_8/results.jsonl \
  --output advised_short_8.mp4
```

---

### 4. Générer des suggestions de prompts (via Mistral)

```bash
python optimize_prompts.py
```

Résultats : `dspy_exports/prompt_suggestions.json`

---

### 5. Export CSV / JSON

```bash
python export_dspy_examples.py
```

---

## 📌 Remarques

- ✅ Ollama a été **remplacé par l’API Mistral** pour simplifier l’intégration et centraliser les appels LLM.
- ✅ Les appels LLaMA se font via DSPy, configuré pour pointer vers Mistral.
- ✅ Le projet reste modulaire : tu peux réactiver Ollama si besoin plus tard.
