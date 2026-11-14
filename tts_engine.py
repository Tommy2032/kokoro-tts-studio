"""
Модуль синтеза речи Kokoro TTS с поддержкой ONNX, смешивания голосов и умной разбивки текста.
"""

import os
import re
import time
import numpy as np
import soundfile as sf
import kokoro_onnx
from kokoro_onnx import Kokoro, MAX_PHONEME_LENGTH, SAMPLE_RATE, trim_audio
from kokoro_onnx.tokenizer import Tokenizer

# Гарантия исправления бага типа speed (int32 -> float32) в ONNX
_original_create_audio = Kokoro._create_audio

def _patched_create_audio(self, phonemes: str, voice: np.ndarray, speed: float):
    if len(phonemes) > MAX_PHONEME_LENGTH:
        phonemes = phonemes[:MAX_PHONEME_LENGTH]
    
    tokens = np.array(self.tokenizer.tokenize(phonemes), dtype=np.int64)
    if len(tokens) > MAX_PHONEME_LENGTH:
        tokens = tokens[:MAX_PHONEME_LENGTH]
        
    voice_part = voice[len(tokens)]
    tokens_batch = [[0, *tokens, 0]]
    
    input_names = [i.name for i in self.sess.get_inputs()]
    if "input_ids" in input_names:
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


# Каталог голосов Kokoro с русским описанием, категориями и полом
VOICES_CATALOG = {
    # Американские женские
    "af_heart": {
        "name": "Heart (Самый популярный)",
        "category": "us_female",
        "category_title": "🇺🇸 США — Женские",
        "gender": "Женский",
        "lang": "en-us",
        "desc": "Мягкий, тёплый, выразительный и естественный тембр. Идеален для аудиокниг и ассистентов.",
    },
    "af_bella": {
        "name": "Bella (Мелодичный)",
        "category": "us_female",
        "category_title": "🇺🇸 США — Женские",
        "gender": "Женский",
        "lang": "en-us",
        "desc": "Яркий, звонкий, чёткая дикция, отлично подходит для презентаций.",
    },
    "af_sarah": {
        "name": "Sarah (Профессиональный)",
        "category": "us_female",
        "category_title": "🇺🇸 США — Женские",
        "gender": "Женский",
        "lang": "en-us",
        "desc": "Деловой, нейтральный, дикторский голос для новостей и обучающих видео.",
    },
    "af_sky": {
        "name": "Sky (Молодёжный)",
        "category": "us_female",
        "category_title": "🇺🇸 США — Женские",
        "gender": "Женский",
        "lang": "en-us",
        "desc": "Лёгкий, непринуждённый, дружелюбный и современный.",
    },
    "af_nicole": {
        "name": "Nicole (Спокойный)",
        "category": "us_female",
        "category_title": "🇺🇸 США — Женские",
        "gender": "Женский",
        "lang": "en-us",
        "desc": "Размеренный, вкрадчивый, умиротворяющий тон.",
    },
    "af_nova": {
        "name": "Nova (Энергичный)",
        "category": "us_female",
        "category_title": "🇺🇸 США — Женские",
        "gender": "Женский",
        "lang": "en-us",
        "desc": "Динамичный, бодрый тембр для рекламы и активных диалогов.",
    },
    "af_jessica": {
        "name": "Jessica (Уверенный)",
        "category": "us_female",
        "category_title": "🇺🇸 США — Женские",
        "gender": "Женский",
        "lang": "en-us",
        "desc": "Открытый, уверенный голос с отличной интонационной подачей.",
    },
    "af_river": {
        "name": "River (Бархатный)",
        "category": "us_female",
        "category_title": "🇺🇸 США — Женские",
        "gender": "Женский",
        "lang": "en-us",
        "desc": "Глубокий, низкий женский тембр, богатый обертонами.",
    },
    "af_alloy": {
        "name": "Alloy (Нейтральный)",
        "category": "us_female",
        "category_title": "🇺🇸 США — Женские",
        "gender": "Женский",
        "lang": "en-us",
        "desc": "Сбалансированный голос в стиле ChatGPT Alloy.",
    },
    "af_aoede": {
        "name": "Aoede (Артистичный)",
        "category": "us_female",
        "category_title": "🇺🇸 США — Женские",
        "gender": "Женский",
        "lang": "en-us",
        "desc": "Поэтичный, эмоциональный, выразительный тон для рассказов.",
    },
    "af_kore": {
        "name": "Kore (Камерный)",
        "category": "us_female",
        "category_title": "🇺🇸 США — Женские",
        "gender": "Женский",
        "lang": "en-us",
        "desc": "Интимный, тихий, доверительный голос.",
    },
    # Американские мужские
    "am_adam": {
        "name": "Adam (Глубокий диктор)",
        "category": "us_male",
        "category_title": "🇺🇸 США — Мужские",
        "gender": "Мужской",
        "lang": "en-us",
        "desc": "Авторитетный, низкий, харизматичный голос ведущего.",
    },
    "am_michael": {
        "name": "Michael (Тёплый нарратор)",
        "category": "us_male",
        "category_title": "🇺🇸 США — Мужские",
        "gender": "Мужской",
        "lang": "en-us",
        "desc": "Приятный рассказчик, отличная разборчивость для подкастов.",
    },
    "am_liam": {
        "name": "Liam (Естественный)",
        "category": "us_male",
        "category_title": "🇺🇸 США — Мужские",
        "gender": "Мужской",
        "lang": "en-us",
        "desc": "Молодой, живой мужской голос для непринуждённой беседы.",
    },
    "am_echo": {
        "name": "Echo (Кинематографичный)",
        "category": "us_male",
        "category_title": "🇺🇸 США — Мужские",
        "gender": "Мужской",
        "lang": "en-us",
        "desc": "Звучный, мощный голос для трейлеров и драматических текстов.",
    },
    "am_eric": {
        "name": "Eric (Деловой)",
        "category": "us_male",
        "category_title": "🇺🇸 США — Мужские",
        "gender": "Мужской",
        "lang": "en-us",
        "desc": "Чёткий, конструктивный мужской голос для инструкций и курсов.",
    },
    "am_fenrir": {
        "name": "Fenrir (Басовитый)",
        "category": "us_male",
        "category_title": "🇺🇸 США — Мужские",
        "gender": "Мужской",
        "lang": "en-us",
        "desc": "Очень глубокий, суровый, мужественный тембр.",
    },
    "am_onyx": {
        "name": "Onyx (Харизматичный)",
        "category": "us_male",
        "category_title": "🇺🇸 США — Мужские",
        "gender": "Мужской",
        "lang": "en-us",
        "desc": "Тёмный, бархатный, уверенный мужской голос.",
    },
    "am_puck": {
        "name": "Puck (Игривый)",
        "category": "us_male",
        "category_title": "🇺🇸 США — Мужские",
        "gender": "Мужской",
        "lang": "en-us",
        "desc": "Быстрый, задорный, эмоциональный персонажный голос.",
    },
    "am_santa": {
        "name": "Santa (Сказочный баритон)",
        "category": "us_male",
        "category_title": "🇺🇸 США — Мужские",
        "gender": "Мужской",
        "lang": "en-us",
        "desc": "Добродушный, басовитый, праздничный тембр Деда Мороза.",
    },
    # Британские голоса
    "bf_alice": {
        "name": "Alice (Британская леди)",
        "category": "uk",
        "category_title": "🇬🇧 Великобритания",
        "gender": "Женский",
        "lang": "en-gb",
        "desc": "Изысканный британский акцент Received Pronunciation.",
    },
    "bf_emma": {
        "name": "Emma (Классический английский)",
        "category": "uk",
        "category_title": "🇬🇧 Великобритания",
        "gender": "Женский",
        "lang": "en-gb",
        "desc": "Мягкая интеллигентная британская речь для литературы.",
    },
    "bf_isabella": {
        "name": "Isabella (Благородный)",
        "category": "uk",
        "category_title": "🇬🇧 Великобритания",
        "gender": "Женский",
        "lang": "en-gb",
        "desc": "Аристократичный, выразительный британский тембр.",
    },
    "bf_lily": {
        "name": "Lily (Молодой британский)",
        "category": "uk",
        "category_title": "🇬🇧 Великобритания",
        "gender": "Женский",
        "lang": "en-gb",
        "desc": "Свежий, звонкий британский женский голос.",
    },
    "bm_george": {
        "name": "George (Британский джентльмен)",
        "category": "uk",
        "category_title": "🇬🇧 Великобритания",
        "gender": "Мужской",
        "lang": "en-gb",
        "desc": "Классический, солидный британский мужской голос BBC-стиля.",
    },
    "bm_daniel": {
        "name": "Daniel (Академический)",
        "category": "uk",
        "category_title": "🇬🇧 Великобритания",
        "gender": "Мужской",
        "lang": "en-gb",
        "desc": "Интеллектуальный британский тембр для лекций и науки.",
    },
    "bm_lewis": {
        "name": "Lewis (Повествователь)",
        "category": "uk",
        "category_title": "🇬🇧 Великобритания",
        "gender": "Мужской",
        "lang": "en-gb",
        "desc": "Тёплый, спокойный английский рассказчик.",
    },
    "bm_fable": {
        "name": "Fable (Атмосферный)",
        "category": "uk",
        "category_title": "🇬🇧 Великобритания",
        "gender": "Мужской",
        "lang": "en-gb",
        "desc": "Загадочный, сказочный тембр для фэнтези и историй.",
    },
    # Испанские
    "ef_dora": {
        "name": "Dora (Испанский женский)",
        "category": "es",
        "category_title": "🇪🇸 Испания",
        "gender": "Женский",
        "lang": "es",
        "desc": "Натуральный испанский женский голос.",
    },
    "em_alex": {
        "name": "Alex (Испанский мужской)",
        "category": "es",
        "category_title": "🇪🇸 Испания",
        "gender": "Мужской",
        "lang": "es",
        "desc": "Натуральный испанский мужской голос.",
    },
    "em_santa": {
        "name": "Papá Noel (Испанский Санта)",
        "category": "es",
        "category_title": "🇪🇸 Испания",
        "gender": "Мужской",
        "lang": "es",
        "desc": "Праздничный басовитый голос на испанском.",
    },
    # Французский
    "ff_siwis": {
        "name": "Siwis (Французский женский)",
        "category": "fr",
        "category_title": "🇫🇷 Франция",
        "gender": "Женский",
        "lang": "fr-fr",
        "desc": "Элегантный французский женский голос.",
    },
    # Итальянские
    "if_sara": {
        "name": "Sara (Итальянский женский)",
        "category": "it",
        "category_title": "🇮🇹 Италия",
        "gender": "Женский",
        "lang": "it",
        "desc": "Мелодичный итальянский женский голос.",
    },
    "im_nicola": {
        "name": "Nicola (Итальянский мужской)",
        "category": "it",
        "category_title": "🇮🇹 Италия",
        "gender": "Мужской",
        "lang": "it",
        "desc": "Энергичный итальянский мужской тембр.",
    },
    # Японские
    "jf_alpha": {
        "name": "Alpha (Японский женский)",
        "category": "ja",
        "category_title": "🇯🇵 Япония",
        "gender": "Женский",
        "lang": "ja",
        "desc": "Чистый японский женский голос в стиле аниме/ассистентов.",
    },
    "jf_gongitsune": {
        "name": "Gongitsune (Японская лисица)",
        "category": "ja",
        "category_title": "🇯🇵 Япония",
        "gender": "Женский",
        "lang": "ja",
        "desc": "Традиционный японский повествовательный тембр.",
    },
    "jf_nezumi": {
        "name": "Nezumi (Японский сказочный)",
        "category": "ja",
        "category_title": "🇯🇵 Япония",
        "gender": "Женский",
        "lang": "ja",
        "desc": "Милый сказочный японский голос.",
    },
    "jf_tebukuro": {
        "name": "Tebukuro (Японский нежный)",
        "category": "ja",
        "category_title": "🇯🇵 Япония",
        "gender": "Женский",
        "lang": "ja",
        "desc": "Нежный и мягкий японский женский голос.",
    },
    "jm_kumo": {
        "name": "Kumo (Японский мужской)",
        "category": "ja",
        "category_title": "🇯🇵 Япония",
        "gender": "Мужской",
        "lang": "ja",
        "desc": "Спокойный японский мужской тембр.",
    },
    # Китайские
    "zf_xiaoxiao": {
        "name": "Xiaoxiao (Китайский женский)",
        "category": "zh",
        "category_title": "🇨🇳 Китай (Мандарин)",
        "gender": "Женский",
        "lang": "cmn",
        "desc": "Популярный ясный китайский женский голос.",
    },
    "zf_xiaobei": {
        "name": "Xiaobei (Китайский женский)",
        "category": "zh",
        "category_title": "🇨🇳 Китай (Мандарин)",
        "gender": "Женский",
        "lang": "cmn",
        "desc": "Энергичный современный голос.",
    },
    "zf_xiaoni": {
        "name": "Xiaoni (Китайский женский)",
        "category": "zh",
        "category_title": "🇨🇳 Китай (Мандарин)",
        "gender": "Женский",
        "lang": "cmn",
        "desc": "Мягкий китайский женский голос.",
    },
    "zf_xiaoyi": {
        "name": "Xiaoyi (Китайский женский)",
        "category": "zh",
        "category_title": "🇨🇳 Китай (Мандарин)",
        "gender": "Женский",
        "lang": "cmn",
        "desc": "Классический китайский женский диктор.",
    },
    "zm_yunjian": {
        "name": "Yunjian (Китайский мужской)",
        "category": "zh",
        "category_title": "🇨🇳 Китай (Мандарин)",
        "gender": "Мужской",
        "lang": "cmn",
        "desc": "Глубокий мужской китайский голос.",
    },
    "zm_yunxi": {
        "name": "Yunxi (Китайский мужской)",
        "category": "zh",
        "category_title": "🇨🇳 Китай (Мандарин)",
        "gender": "Мужской",
        "lang": "cmn",
        "desc": "Молодой естественный мужской голос.",
    },
    "zm_yunxia": {
        "name": "Yunxia (Китайский мужской)",
        "category": "zh",
        "category_title": "🇨🇳 Китай (Мандарин)",
        "gender": "Мужской",
        "lang": "cmn",
        "desc": "Разговорный китайский тембр.",
    },
    "zm_yunyang": {
        "name": "Yunyang (Китайский ведущий)",
        "category": "zh",
        "category_title": "🇨🇳 Китай (Мандарин)",
        "gender": "Мужской",
        "lang": "cmn",
        "desc": "Телевизионный дикторский китайский голос.",
    },
    # Португальские
    "pf_dora": {
        "name": "Dora (Бразильский женский)",
        "category": "pt",
        "category_title": "🇧🇷 Португальский",
        "gender": "Женский",
        "lang": "pt-br",
        "desc": "Бразильский португальский женский голос.",
    },
    "pm_alex": {
        "name": "Alex (Бразильский мужской)",
        "category": "pt",
        "category_title": "🇧🇷 Португальский",
        "gender": "Мужской",
        "lang": "pt-br",
        "desc": "Бразильский португальский мужской голос.",
    },
    "pm_santa": {
        "name": "Papai Noel (Бразильский Санта)",
        "category": "pt",
        "category_title": "🇧🇷 Португальский",
        "gender": "Мужской",
        "lang": "pt-br",
        "desc": "Бразильский праздничный мужской голос.",
    },
    # Хинди
    "hf_alpha": {
        "name": "Alpha (Хинди женский)",
        "category": "hi",
        "category_title": "🇮🇳 Хинди",
        "gender": "Женский",
        "lang": "hi",
        "desc": "Женский голос на хинди.",
    },
    "hf_beta": {
        "name": "Beta (Хинди женский)",
        "category": "hi",
        "category_title": "🇮🇳 Хинди",
        "gender": "Женский",
        "lang": "hi",
        "desc": "Мягкий женский голос на хинди.",
    },
    "hm_omega": {
        "name": "Omega (Хинди мужской)",
        "category": "hi",
        "category_title": "🇮🇳 Хинди",
        "gender": "Мужской",
        "lang": "hi",
        "desc": "Мужской голос на хинди.",
    },
    "hm_psi": {
        "name": "Psi (Хинди мужской)",
        "category": "hi",
        "category_title": "🇮🇳 Хинди",
        "gender": "Мужской",
        "lang": "hi",
        "desc": "Глубокий мужской тембр на хинди.",
    },
}

SUPPORTED_LANGUAGES = [
    ("auto", "✨ Автоопределение (по тексту)"),
    ("ru", "🇷🇺 Русский (IPA фонетизация espeak)"),
    ("en-us", "🇺🇸 Английский (США)"),
    ("en-gb", "🇬🇧 Английский (Великобритания)"),
    ("es", "🇪🇸 Испанский"),
    ("fr-fr", "🇫🇷 Французский"),
    ("it", "🇮🇹 Итальянский"),
    ("ja", "🇯🇵 Японский"),
    ("cmn", "🇨🇳 Китайский (Мандарин)"),
    ("pt-br", "🇧🇷 Португальский (Бразилия)"),
    ("hi", "🇮🇳 Хинди"),
]


class TTSEngine:
    def __init__(self, model_path: str = "kokoro-v1.0.onnx", voices_path: str = "voices-v1.0.bin"):
        self.model_path = model_path
        self.voices_path = voices_path
        self.kokoro: Kokoro | None = None
        self._init_kokoro()

    def _init_kokoro(self):
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Файл модели не найден: {self.model_path}")
        if not os.path.exists(self.voices_path):
            raise FileNotFoundError(f"Файл голосов не найден: {self.voices_path}")
        
        self.kokoro = Kokoro(self.model_path, self.voices_path)

    @property
    def sample_rate(self) -> int:
        return SAMPLE_RATE

    def get_available_voices(self) -> list[str]:
        if self.kokoro:
            return self.kokoro.get_voices()
        return list(VOICES_CATALOG.keys())

    def get_voice_info(self, voice_id: str) -> dict:
        if voice_id in VOICES_CATALOG:
            return VOICES_CATALOG[voice_id]
        return {
            "name": voice_id,
            "category": "other",
            "category_title": "Прочие",
            "gender": "Не указан",
            "lang": "en-us",
            "desc": "Голос из модели Kokoro",
        }

    def detect_language(self, text: str, voice_id: str) -> str:
        # Проверка на кириллицу
        if re.search(r"[\u0400-\u04FF]", text):
            return "ru"
        # Проверка на японские иероглифы (хирагана, катакана, кандзи)
        if re.search(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff]", text):
            if voice_id.startswith("zf_") or voice_id.startswith("zm_"):
                return "cmn"
            return "ja"
        # Проверка префикса голоса для языка
        if voice_id.startswith("bf_") or voice_id.startswith("bm_"):
            return "en-gb"
        if voice_id.startswith("ef_") or voice_id.startswith("em_"):
            return "es"
        if voice_id.startswith("ff_"):
            return "fr-fr"
        if voice_id.startswith("if_") or voice_id.startswith("im_"):
            return "it"
        if voice_id.startswith("pf_") or voice_id.startswith("pm_"):
            return "pt-br"
        if voice_id.startswith("hf_") or voice_id.startswith("hm_"):
            return "hi"
        
        return "en-us"

    def blend_voices(self, voice_id_a: str, voice_id_b: str, ratio_a: float = 0.7) -> np.ndarray:
        """Смешивание двух стилей голосов в заданной пропорции."""
        ratio_a = max(0.0, min(1.0, float(ratio_a)))
        ratio_b = 1.0 - ratio_a
        style_a = self.kokoro.get_voice_style(voice_id_a)
        style_b = self.kokoro.get_voice_style(voice_id_b)
        blended = (style_a * ratio_a + style_b * ratio_b).astype(np.float32)
        return blended

    def split_text_into_chunks(self, text: str, max_chunk_chars: int = 350) -> list[str]:
        """Умное разбиение длинного текста на смысловые фрагменты по абзацам и предложениям."""
        paragraphs = text.splitlines()
        chunks = []
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if len(p) <= max_chunk_chars:
                chunks.append(p)
            else:
                # Разбиение по предложениям
                sentences = re.split(r"(?<=[.!?…])\s+", p)
                current_chunk = ""
                for s in sentences:
                    s = s.strip()
                    if not s:
                        continue
                    if len(current_chunk) + len(s) + 1 <= max_chunk_chars:
                        current_chunk = (current_chunk + " " + s).strip()
                    else:
                        if current_chunk:
                            chunks.append(current_chunk)
                        current_chunk = s
                if current_chunk:
                    chunks.append(current_chunk)
        return chunks if chunks else [text.strip()]

    def synthesize(
        self,
        text: str,
        voice: str = "af_heart",
        blend_voice: str | None = None,
        blend_ratio: float = 0.5,
        speed: float = 1.0,
        lang: str = "auto",
        trim: bool = True,
        progress_callback=None,
        is_cancelled=None,
    ) -> tuple[np.ndarray, int]:
        """
        Синтез аудио из текста с поддержкой смешивания голосов и прогресса.
        """
        if not text or not text.strip():
            raise ValueError("Текст для озвучки не может быть пустым.")

        speed = max(0.5, min(2.0, float(speed)))

        if lang == "auto":
            resolved_lang = self.detect_language(text, voice)
        else:
            resolved_lang = lang

        # Вычисление вектора голоса (одиночный или смешанный)
        if blend_voice and blend_voice in self.kokoro.voices and blend_voice != voice:
            voice_style = self.blend_voices(voice, blend_voice, ratio_a=(1.0 - blend_ratio))
        else:
            voice_style = self.kokoro.get_voice_style(voice)

        chunks = self.split_text_into_chunks(text)
        total_chunks = len(chunks)
        audio_parts = []

        for idx, chunk in enumerate(chunks):
            if is_cancelled and is_cancelled():
                raise InterruptedError("Синтез прерван пользователем.")

            if progress_callback:
                progress_callback(idx, total_chunks, f"Озвучивание фрагмента {idx + 1}/{total_chunks}...")

            part, sr = self.kokoro.create(
                chunk,
                voice=voice_style,
                speed=speed,
                lang=resolved_lang,
                trim=trim,
            )
            audio_parts.append(part)

        if not audio_parts:
            return np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE

        full_audio = np.concatenate(audio_parts)
        if progress_callback:
            progress_callback(total_chunks, total_chunks, "Синтез успешно завершён!")

        return full_audio, SAMPLE_RATE

    def preview_voice(self, voice_id: str, lang: str = "auto", speed: float = 1.0) -> tuple[np.ndarray, int]:
        """Быстрое превью для знакомства с голосом."""
        if lang == "ru":
            sample_text = "Привет! Я готов озвучить ваш текст."
        elif lang in ("en-us", "en-gb") or voice_id.startswith(("af_", "am_", "bf_", "bm_")):
            sample_text = "Hello! I am ready to read your text with high quality."
        elif lang == "es" or voice_id.startswith(("ef_", "em_")):
            sample_text = "¡Hola! Estoy listo para leer tu texto."
        elif lang == "fr-fr" or voice_id.startswith("ff_"):
            sample_text = "Bonjour! Je suis prêt à lire votre texte."
        elif lang == "it" or voice_id.startswith(("if_", "im_")):
            sample_text = "Ciao! Sono pronto a leggere il tuo testo."
        elif lang == "ja" or voice_id.startswith(("jf_", "jm_")):
            sample_text = "こんにちは！テキストを読み上げる準備ができました。"
        elif lang == "cmn" or voice_id.startswith(("zf_", "zm_")):
            sample_text = "你好！我已经准备好为您朗读文本。"
        else:
            sample_text = "Hello! This voice is ready."

        return self.synthesize(
            text=sample_text,
            voice=voice_id,
            speed=speed,
            lang=lang,
        )
