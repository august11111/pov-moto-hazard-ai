# 🏍️ MotoAdvice: Video Analysis Assistant for Motorcyclists

This project enables the analysis of POV videos (e.g., YouTube Shorts) by extracting frames, detecting objects via YOLOv8, generating riding advice using LLMs (Pixtrale via Mistral), and producing a PDF report and an annotated video.

---

## 📁 Project Structure

```
.
├── moto_advice_pipeline.py               # Main (static) pipeline
├── moto_advice_consecutive.py           # Alternative pipeline (with contextual memory)
├── generate_report_pdf.py               # PDF report generation
├── overlay_advice.py                    # Overlay advice on video
├── optimize_prompts.py                  # Prompt optimization (Mistral only)
├── recup_shorts.py                      # Download YouTube Shorts
├── run_per_video_analysis.py            # Batch processing (simple pipeline)
├── run_per_video_analysis_consecutive.py # Batch processing (contextual pipeline)
├── export_dspy_examples.py              # Export CSV / JSON results
├── results/                             # Intermediate results
├── report/                              # Generated PDF reports
└── pixtrale/shorts_clips/               # Downloaded videos
```

---

## 🔧 Installation

### 1. System Requirements

* Python ≥ 3.9
* `ffmpeg` (for video processing)
* API access to [Mistral](https://mistral.ai)
* *(⚠️ Ollama is no longer used in this version)*

### 2. Python Dependencies

```bash
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows

pip install -r requirements.txt
```

**Or manually:**

```bash
pip install opencv-python yt-dlp python-dotenv ultralytics tqdm reportlab jsonlines pandas dspy
```

---

## 🔑 Configuration (Environment Variables)

Create a `.env` file:

```env
MISTRAL_API_KEY=sk-...
PIXTRALE_API_BASE=https://api.mistral.ai/v1
PIXTRALE_MODEL=pixtral-large-latest

# The following Ollama variables can be ignored.
# OLLAMA_API_BASE=http://localhost:11434
# OLLAMA_MODEL=llama3.2:latest
```

> 🟢 **Note:** All prompt generation (Pixtrale and LLaMA) now goes through the **Mistral API**, configured via DSPy. Ollama is no longer required.

---

## 🎬 Usage

### 1. Download Shorts

```bash
python recup_shorts.py
```

---

### 2. Run the Full Pipeline

#### ➤ With Temporal Context

```bash
python run_per_video_analysis_consecutive.py
```

#### ➤ Simple Version

```bash
python run_per_video_analysis.py
```

---

### 3. Annotate the Video

```bash
python overlay_advice.py \
  --video pixtrale/shorts_clips/short_8.mp4 \
  --json results_temporal/short_8/results.jsonl \
  --output advised_short_8.mp4
```

---

### 4. Generate Prompt Suggestions (via Mistral)

```bash
python optimize_prompts.py
```

Results: `dspy_exports/prompt_suggestions.json`

---

### 5. Export to CSV / JSON

```bash
python export_dspy_examples.py
```

---

## 📌 Notes

* ✅ Ollama has been **replaced by Mistral API** for a simpler and unified LLM integration.
* ✅ LLaMA calls are handled via DSPy configured to use Mistral.
* ✅ The project remains modular — Ollama can be re-enabled later if needed.

---

Let me know if you'd like a version with Markdown formatting tailored for GitHub display (including badges, screenshots, etc.).

