Thymiko – Inkphora (PoC Touch Prototype)

Live Demo: https://huggingface.co/spaces/Miren-12/thymiko-inkphora-PoC-touch

---

A touch-based AI prototype that transforms freehand drawing gestures into:

Heuristic valence–arousal mapping

Short AI-generated poetic output

Demo mood visualization

Curated music suggestion layer

This PoC validates gesture-to-affect mapping and event-driven interaction logic within a lightweight deployed environment.

---

Architecture

User Touch Input (Canvas)
→ Feature Extraction (stroke metrics: direction bias, rhythm, speed variance)
→ Heuristic Valence–Arousal Mapping
→ LLM API (poetic generation conditioned on arousal)
→ Mood Visualization (demo graphic)
→ Music Suggestion Layer

---

Hosted via Hugging Face (Gradio-based interface).

Core Components
1. Gesture Feature Extraction

Left/Right bias

Stroke length

Speed variance

Rhythm / pause ratio

Heuristic mapping (non-clinical)

2. Heuristic Affect Mapping

Simplified Valence–Arousal model

Deterministic mapping (v1 PoC)

No training, no personalization

3. AI Text Generation

Prompt conditioning by arousal value

Controlled output length

Seed-based generation

4. Music Suggestion Layer

3 curated paths

Non-autoplay

UX-first exploration

---

Technical Stack:

Python (Gradio app)

LLM API integration

Heuristic affect mapping logic

Deployed on Hugging Face Spaces

Event-driven interaction model

---

Validation Scope:

UX testing of draw → reflect flow

Face-validity testing of heuristic mapping

Engagement observation (qualitative)

---

Privacy & Positioning

No raw images stored

No biometric data

Non-diagnostic

Experimental prototype

---

Status

Proof of Concept (Deployed)
Used for UX validation and interaction testing.
