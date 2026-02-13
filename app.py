# © 2025 Thymiko / Inkphora. All rights reserved.
# Thymiko — Heuristic Drawing PoC
# (VA -1..+1 + interactive SVG plot + music resonance + AI regulation + canvas lock)

import os
import io
import numpy as np
import gradio as gr
from PIL import Image
from transformers import pipeline

LOGO_FILENAME = "thymiko-logo.png"

# -------------------------
# Music links
# -------------------------
MUSIC_LINKS = {
    "Indian Classical Raga (grounding)": "https://ragya.com/",
    "Neural / Focus Music (regulation)": "https://www.brain.fm/",
    "AI Ambient / Generative (expansion)": "https://suno.ai/",
    "Demo Song (YouTube)": "https://www.youtube.com/watch?v=vFPajU-d-Ek",
}

# -------------------------
# DistilGPT-2 (regulative AI)
# -------------------------
generator = pipeline(
    "text-generation",
    model="distilgpt2",
    device=-1  # CPU safe
)

REGULATIVE_SEEDS = {
    ("neg", "pos"): "Everything feels fast.",
    ("neg", "neg"): "Nothing moves right now.",
    ("pos", "pos"): "Energy is high.",
    ("pos", "neg"): "Calm warmth settles.",
    ("mid", "pos"): "Energy is present.",
    ("mid", "neg"): "Stillness appears.",
    ("mid", "mid"): "Balance holds.",
}

# -------------------------
# Helpers
# -------------------------
def _bucket(v, t=0.2):
    if v <= -t:
        return "neg"
    if v >= t:
        return "pos"
    return "mid"

# -------------------------
# Prompt guardrails
# -------------------------
BLACKLIST = [
    "child", "children", "weak", "fragile",
    "disorder", "pathology", "patient",
    "diagnosis", "therapy", "treatment",
    "mental", "psychological"
]

def _sanitize(text):
    t = text.lower()
    if any(b in t for b in BLACKLIST):
        return "A steady presence remains."
    return text

def generate_regulative_text(v, a):
    seed = REGULATIVE_SEEDS.get((_bucket(v), _bucket(a)), "Stay present.")

    prompt = (
        "You are a neutral inner regulating voice.\n"
        "You describe internal experience without judgment.\n"
        "You speak briefly, calmly, and impersonally.\n"
        "You do not refer to people, groups, age, health, or psychology.\n"
        "You do not diagnose, explain causes, or give advice.\n"
        "You only reflect momentary sensation and regulation.\n\n"
        f"Internal state cue: {seed}\n"
        "Regulative response:"
    )

    out = generator(
        prompt,
        max_new_tokens=18,
        min_new_tokens=6,
        temperature=0.9,
        top_p=0.9,
        repetition_penalty=1.2,
        do_sample=True,
    )[0]["generated_text"]

    clean = out.replace(prompt, "").strip().split("\n")[0]
    return _sanitize(clean)

# -------------------------
# Canvas → PIL
# -------------------------
def _to_pil(x):
    if x is None:
        return None
    if isinstance(x, Image.Image):
        return x.convert("RGBA")
    if isinstance(x, (bytes, bytearray)):
        return Image.open(io.BytesIO(x)).convert("RGBA")
    if isinstance(x, np.ndarray):
        arr = x
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0, 1) * 255).astype("uint8")
        if arr.ndim == 2:
            return Image.fromarray(arr, mode="L").convert("RGBA")
        if arr.ndim == 3:
            return Image.fromarray(arr[:, :, :3], mode="RGB").convert("RGBA")
    if isinstance(x, dict):
        if x.get("composite") is not None:
            return _to_pil(x["composite"])
        if x.get("image") is not None:
            return _to_pil(x["image"])
    raise ValueError("Unsupported image format")

# -------------------------
# Heuristic VA
# -------------------------
def _compute_ink_map(img, size=256):
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    comp = Image.alpha_composite(bg, img)
    g = comp.convert("L").resize((size, size))
    a = np.asarray(g, dtype=np.float32)
    ink = (255 - a) / 255.0
    return np.clip(ink, 0, 1) ** 0.7

def _bias_lr_tb(ink):
    h, w = ink.shape
    mh, mw = h // 2, w // 2
    left = ink[:, :mw].sum()
    right = ink[:, mw:].sum()
    top = ink[:mh, :].sum()
    bottom = ink[mh:, :].sum()
    if left + right < 1e-6 or top + bottom < 1e-6:
        return 0.0, 0.0
    return (
        float((right - left) / (right + left)),
        float((top - bottom) / (top + bottom)),
    )

# -------------------------
# Music logic
# -------------------------
def music_resonance(v, a):
    if a < -0.2:
        return (
            "Indian Classical Raga (grounding)",
            "Neural / Focus Music (regulation)",
            "AI Ambient / Generative (expansion)",
        )
    elif a > 0.2:
        return (
            "Neural / Focus Music (regulation)",
            "AI Ambient / Generative (expansion)",
            "Indian Classical Raga (grounding)",
        )
    else:
        return (
            "AI Ambient / Generative (expansion)",
            "Neural / Focus Music (regulation)",
            "Indian Classical Raga (grounding)",
        )

def render_music_md(p, s, o):
    return f"""
### 🎵 Musical Resonance

**Primary resonance**  
→ [{p}]({MUSIC_LINKS[p]})

**Secondary support**  
→ [{s}]({MUSIC_LINKS[s]})

**Optional exploration**  
→ [{o}]({MUSIC_LINKS[o]})

_Demo only — symbolic resonance, not prescription._
"""

# -------------------------
# SVG plot
# -------------------------
def _svg_va_plot(v, a):
    W, H, pad = 320, 260, 30
    cx, cy = W / 2, H / 2
    x = cx + v * (W - 2 * pad) / 2
    y = cy - a * (H - 2 * pad) / 2
    return f"""
<svg width="{W}" height="{H}" style="border:1px solid rgba(255,255,255,.2);border-radius:12px">
  <line x1="{pad}" y1="{cy}" x2="{W-pad}" y2="{cy}" stroke="white" opacity=".4"/>
  <line x1="{cx}" y1="{pad}" x2="{cx}" y2="{H-pad}" stroke="white" opacity=".4"/>
  <circle cx="{x}" cy="{y}" r="7" fill="#ff4d4d"/>
  <text x="10" y="20" fill="white">V={v:+.2f}  A={a:+.2f}</text>
</svg>
"""

# -------------------------
# Callbacks
# -------------------------
def on_interpret(img):
    pil = _to_pil(img)
    ink = _compute_ink_map(pil)

    if ink.sum() < 10:
        return "", "", "", gr.update(visible=False), gr.update()

    v, a = _bias_lr_tb(ink)
    ai_text = generate_regulative_text(v, a)

    text = f"**Valence {v:+.2f} · Arousal {a:+.2f}**\n\n{ai_text}"
    svg = _svg_va_plot(v, a)

    p, s, o = music_resonance(v, a)
    music_md = render_music_md(p, s, o)

    return (
        text,
        svg,
        music_md,
        gr.update(visible=True),
        gr.update(value=pil, interactive=False),  # 🔒 freeze canvas
    )

def on_clear():
    return (
        "",
        "",
        "",
        gr.update(visible=False),
        gr.update(value=None, interactive=True),  # 🔓 new cycle
    )

# -------------------------
# UI
# -------------------------
def build_ui():
    with gr.Blocks() as demo:
        if os.path.exists(LOGO_FILENAME):
            gr.Image(LOGO_FILENAME, show_label=False, height=100)

        gr.Markdown("# Thymiko — Heuristic Drawing PoC")
        gr.Markdown("_Exploratory, non-diagnostic demo._")

        pad = gr.ImageEditor(label="Draw with your finger", type="pil", height=420)

        interpret = gr.Button("Interpret", variant="primary")
        clear = gr.Button("Clear")

        out_md = gr.Markdown()
        plot = gr.HTML()
        music_block = gr.Markdown(visible=False)

        interpret.click(
            on_interpret,
            inputs=[pad],
            outputs=[out_md, plot, music_block, music_block, pad],
        )

        clear.click(
            on_clear,
            outputs=[out_md, plot, music_block, music_block, pad],
        )

    return demo

if __name__ == "__main__":
    build_ui().launch(server_name="0.0.0.0", server_port=7860)