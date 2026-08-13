"""Mooré-Voice demo — translation + speech recognition for Mooré (mos_Latn).

Gradio app, deployable locally or as a Hugging Face Space:
    uv run --python 3.12 --with gradio --with 'transformers>=4.44' \
        --with 'peft>=0.11' --with torch --with sentencepiece --with protobuf \
        --with soundfile --with librosa python app.py

Env overrides:
    MOORE_MT_BASE      base NLLB model    (default facebook/nllb-200-distilled-600M)
    MOORE_MT_ADAPTER   LoRA adapter dir/repo (default Rekin226/nllb-600M-moore-lora)
    MOORE_ASR_MODEL    Whisper dir/repo   (default Rekin226/whisper-small-moore)
"""

from __future__ import annotations

import os

import gradio as gr
import torch

MT_BASE = os.environ.get("MOORE_MT_BASE", "facebook/nllb-200-distilled-600M")
MT_ADAPTER = os.environ.get("MOORE_MT_ADAPTER", "Rekin226/nllb-600M-moore-lora")
ASR_MODEL = os.environ.get("MOORE_ASR_MODEL", "Rekin226/whisper-small-moore")
ASR_LANG = os.environ.get("MOORE_ASR_LANG", "yo")  # anchor token used in training

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

LANGS = {"Mooré": "mos_Latn", "Français": "fra_Latn", "English": "eng_Latn"}

_mt = {}
_asr = {}


def get_mt():
    if not _mt:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(MT_BASE)
        model = AutoModelForSeq2SeqLM.from_pretrained(MT_BASE, torch_dtype=DTYPE)
        try:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, MT_ADAPTER).merge_and_unload()
            _mt["adapter"] = MT_ADAPTER
        except Exception as e:  # run zero-shot if the adapter is unavailable
            _mt["adapter"] = f"(zero-shot — adapter unavailable: {type(e).__name__})"
        _mt["tok"], _mt["model"] = tok, model.to(DEVICE).eval()
    return _mt


def get_asr():
    if not _asr:
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        _asr["proc"] = WhisperProcessor.from_pretrained(
            ASR_MODEL, language=ASR_LANG, task="transcribe")
        _asr["model"] = WhisperForConditionalGeneration.from_pretrained(
            ASR_MODEL, torch_dtype=DTYPE).to(DEVICE).eval()
    return _asr


def translate(text: str, src_name: str, tgt_name: str) -> str:
    if not text.strip():
        return ""
    mt = get_mt()
    tok, model = mt["tok"], mt["model"]
    tok.src_lang = LANGS[src_name]
    enc = tok(text, return_tensors="pt", truncation=True, max_length=256).to(DEVICE)
    with torch.inference_mode():
        out = model.generate(
            **enc,
            forced_bos_token_id=tok.convert_tokens_to_ids(LANGS[tgt_name]),
            num_beams=4,
            max_new_tokens=192,
        )
    return tok.batch_decode(out, skip_special_tokens=True)[0]


def transcribe(audio_path: str, translate_to: str) -> tuple[str, str]:
    if not audio_path:
        return "", ""
    import librosa
    asr = get_asr()
    audio, _ = librosa.load(audio_path, sr=16000, mono=True)
    inputs = asr["proc"](audio, sampling_rate=16000, return_tensors="pt")
    inputs = inputs.to(DEVICE, DTYPE)
    with torch.inference_mode():
        out = asr["model"].generate(
            inputs.input_features, language=ASR_LANG, task="transcribe",
            max_new_tokens=224)
    text = asr["proc"].batch_decode(out, skip_special_tokens=True)[0].strip()
    translation = ""
    if translate_to != "—" and text:
        translation = translate(text, "Mooré", translate_to)
    return text, translation


with gr.Blocks(title="Mooré-Voice") as demo:
    gr.Markdown(
        "# Mooré-Voice\n"
        "Open translation and speech recognition for **Mooré** (Mòoré / Mossi, "
        "`mos`) — the language of ~8M people in Burkina Faso.\n"
    )
    with gr.Tab("Translate"):
        with gr.Row():
            src = gr.Dropdown(list(LANGS), value="Mooré", label="From")
            tgt = gr.Dropdown(list(LANGS), value="Français", label="To")
        inp = gr.Textbox(lines=4, label="Text",
                         placeholder="Yãmb kibare?")
        out = gr.Textbox(lines=4, label="Translation")
        gr.Button("Translate", variant="primary").click(
            translate, [inp, src, tgt], out)
    with gr.Tab("Speech → Text"):
        audio = gr.Audio(sources=["microphone", "upload"], type="filepath",
                         label="Mooré speech")
        to = gr.Dropdown(["—", "Français", "English"], value="Français",
                         label="Also translate to")
        txt = gr.Textbox(lines=3, label="Mooré transcript")
        tr = gr.Textbox(lines=3, label="Translation")
        gr.Button("Transcribe", variant="primary").click(
            transcribe, [audio, to], [txt, tr])
    gr.Markdown(
        "Models: NLLB-200 + Mooré LoRA (translation), Whisper-small fine-tuned "
        "on Mooré speech (ASR). Corpus and recipes: "
        "[github.com/Rekin226/Moore-Voice](https://github.com/Rekin226/Moore-Voice)."
    )

if __name__ == "__main__":
    demo.launch()
