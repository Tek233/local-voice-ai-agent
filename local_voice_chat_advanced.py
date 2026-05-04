import argparse
import sys

import numpy as np
import scipy.signal
from faster_whisper import WhisperModel
from fastrtc import ReplyOnPause, Stream
from loguru import logger
from ollama import chat
from piper import PiperVoice

# -----------------------
# Config
# -----------------------
TARGET_SR = 16000
PIPER_SR = 22050


# -----------------------
# Models
# -----------------------
whisper_model = WhisperModel(
    "small",
    compute_type="int8",
)

voice = PiperVoice.load(
    model_path="fr_FR-upmc-medium.onnx",
    config_path="fr_FR-upmc-medium.onnx.json",
)


# -----------------------
# Logging
# -----------------------
logger.remove()
logger.add(sys.stderr, level="INFO")


# -----------------------
# Audio utils
# -----------------------
def to_float32(audio):
    if audio.dtype == np.int16:
        return audio.astype(np.float32) / 32768.0
    return audio.astype(np.float32)


def resample(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio
    return scipy.signal.resample_poly(audio, target_sr, orig_sr)


def preprocess_audio(audio):
    if isinstance(audio, tuple):
        sample_rate, audio = audio
    else:
        sample_rate = TARGET_SR

    if isinstance(audio, bytes):
        audio = np.frombuffer(audio, dtype=np.int16)

    if not isinstance(audio, np.ndarray):
        audio = np.array(audio)

    audio = to_float32(audio)
    audio = audio.ravel()

    if sample_rate != TARGET_SR:
        audio = resample(audio, sample_rate, TARGET_SR)

    return audio


# -----------------------
# Speech-to-Text
# -----------------------
def stt(audio):
    try:
        audio = preprocess_audio(audio)

        segments, _ = whisper_model.transcribe(
            audio,
            language="fr",
            vad_filter=True,
        )

        return " ".join(seg.text for seg in segments).strip()

    except Exception as e:
        logger.error(f"STT error: {e}")
        return ""


# -----------------------
# Piper TTS helpers
# -----------------------
def extract_audio_from_chunk(chunk):
    if hasattr(chunk, "audio_int16_array"):
        return chunk.audio_int16_array.astype(np.float32) / 32768.0

    if hasattr(chunk, "audio_float_array"):
        return chunk.audio_float_array.astype(np.float32)

    raise ValueError(f"Unsupported chunk format: {dir(chunk)}")


def float_to_int16(audio):
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767).astype(np.int16)


def piper_tts_stream(text):
    try:
        for chunk in voice.synthesize(text):
            audio = extract_audio_from_chunk(chunk)

            # fast resample
            audio = resample(audio, PIPER_SR, TARGET_SR)

            audio_int16 = float_to_int16(audio)

            yield (TARGET_SR, audio_int16.reshape(1, -1))

    except Exception as e:
        logger.error(f"TTS error: {e}")


# -----------------------
# LLM
# -----------------------
SYSTEM_PROMPT = (
    "Tu es un assistant vocal utile et naturel."
    "Réponds de manière concise, claire et adaptée à une conversation orale. "
    "Utilise un langage simple et fluide. "
    "Évite les émojis, les caractères spéciaux et les formulations trop formelles."
)


def generate_response(transcript):
    try:
        response = chat(
            model="mistral:7b",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": transcript},
            ],
            options={"num_predict": 150},
        )

        return response["message"]["content"].strip()

    except Exception as e:
        logger.error(f"LLM error: {e}")
        return ""


# -----------------------
# Main pipeline
# -----------------------
def echo(audio):
    transcript = stt(audio)

    if not transcript:
        return

    logger.info(f"User: {transcript}")

    response_text = generate_response(transcript)

    if not response_text:
        return

    logger.info(f"Assistant: {response_text}")

    yield from piper_tts_stream(response_text)


# -----------------------
# Stream setup
# -----------------------
def create_stream():
    return Stream(
        ReplyOnPause(
            echo,
            can_interrupt=False,
            output_sample_rate=TARGET_SR,
        ),
        modality="audio",
        mode="send-receive",
    )


# -----------------------
# Entry point
# -----------------------
def main():
    parser = argparse.ArgumentParser(description="Optimized Local Voice Chat")
    parser.add_argument(
        "--phone",
        action="store_true",
        help="Launch with FastRTC phone interface",
    )
    args = parser.parse_args()

    stream = create_stream()

    if args.phone:
        logger.info("Starting phone interface...")
        stream.fastphone()
    else:
        logger.info("Starting web UI...")
        stream.ui.launch(share=True)


if __name__ == "__main__":
    main()
