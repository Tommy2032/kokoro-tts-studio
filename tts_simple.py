import re
import os
import soundfile as sf
import numpy as np
from kokoro_onnx import Kokoro, MAX_PHONEME_LENGTH, SAMPLE_RATE

# Защита от бага типа аргумента speed в ONNX
_original_create_audio = Kokoro._create_audio
def _patched_create_audio(self, phonemes, voice, speed):
    if len(phonemes) > MAX_PHONEME_LENGTH:
        phonemes = phonemes[:MAX_PHONEME_LENGTH]
    tokens = np.array(self.tokenizer.tokenize(phonemes), dtype=np.int64)
    if len(tokens) > MAX_PHONEME_LENGTH:
        tokens = tokens[:MAX_PHONEME_LENGTH]
    voice_part = voice[len(tokens)]
    tokens_batch = [[0, *tokens, 0]]
    if "input_ids" in [i.name for i in self.sess.get_inputs()]:
        inputs = {
            "input_ids": tokens_batch,
            "style": np.array(voice_part, dtype=np.float32),
            "speed": np.array([speed], dtype=np.float32),
        }
    else:
        inputs = {
            "tokens": tokens_batch,
            "style": voice_part,
            "speed": np.ones(1, dtype=np.float32) * speed,
        }
    audio = self.sess.run(None, inputs)[0]
    return audio, SAMPLE_RATE
Kokoro._create_audio = _patched_create_audio

# Загрузка модели и голосов
print("Загрузка модели Kokoro...")
kokoro = Kokoro("kokoro-v1.0.onnx", "voices-v1.0.bin")

# Текст для озвучки: из файла input.txt или значение по умолчанию
input_file = "input.txt"
if os.path.exists(input_file):
    with open(input_file, encoding="utf-8") as f:
        text = f.read().strip()
else:
    text = "Привет! Это проверка синтеза речи с помощью Kokoro ONNX."

# Автоопределение языка (русский или английский)
if re.search(r"[\u0400-\u04FF]", text):
    detected_lang = "ru"
else:
    detected_lang = "en-us"

print(f"Синтез текста ({len(text)} симв., язык: {detected_lang})...")
samples, sr = kokoro.create(
    text,
    voice="af_heart",   # голос (af_heart, am_adam, bf_alice и др.)
    speed=1.0,          # 0.8 медленнее, 1.2 быстрее
    lang=detected_lang, # "ru", "en-us", "en-gb", "es", "ja" и др.
)

output_file = "output.wav"
sf.write(output_file, samples, sr)
duration = len(samples) / sr
print(f"Готово: {output_file} (Длительность: {duration:.2f} сек., {sr} Гц)")
