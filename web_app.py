"""
Kokoro TTS Web Studio — Современный веб-интерфейс для синтеза речи.
Работает локально через FastAPI и доступен в любом браузере (Chrome, Edge, Firefox).
"""

import os
import io
import time
import soundfile as sf
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from tts_engine import TTSEngine, VOICES_CATALOG, SUPPORTED_LANGUAGES

app = FastAPI(title="Kokoro TTS Web Studio")

# Глобальный экземпляр движка TTS
engine = None

def get_engine():
    global engine
    if engine is None:
        engine = TTSEngine()
    return engine

class SynthRequest(BaseModel):
    text: str
    voice: str = "af_heart"
    blend_voice: str | None = None
    blend_ratio: float = 0.5
    speed: float = 1.0
    lang: str = "auto"
    trim: bool = True

HTML_UI = """<!DOCTYPE html>
<html lang="ru">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Kokoro TTS Web Studio — Синтез речи</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg: #090d16;
            --card-bg: rgba(21, 29, 47, 0.75);
            --border: #222f47;
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --accent: #6366f1;
            --accent-hover: #4f46e5;
            --cyan: #06b6d4;
            --green: #10b981;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }

        body {
            background-color: var(--bg);
            background-image: 
                radial-gradient(at 0% 0%, rgba(99, 102, 241, 0.12) 0px, transparent 50%),
                radial-gradient(at 100% 100%, rgba(6, 182, 212, 0.1) 0px, transparent 50%);
            color: var(--text-main);
            min-height: 100vh;
            padding: 20px;
            display: flex;
            flex-direction: column;
            align-items: center;
        }

        .container {
            width: 100%;
            max-width: 1200px;
            display: flex;
            flex-direction: column;
            gap: 16px;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border);
            padding: 16px 24px;
            border-radius: 16px;
        }

        .logo-area {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .logo-icon {
            font-size: 28px;
        }

        h1 {
            font-size: 20px;
            font-weight: 700;
            background: linear-gradient(135deg, #fff 0%, #cbd5e1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .subtitle {
            font-size: 12px;
            color: var(--text-muted);
        }

        .badge {
            background: rgba(16, 185, 129, 0.15);
            color: #34d399;
            border: 1px solid rgba(16, 185, 129, 0.3);
            padding: 6px 12px;
            border-radius: 20px;
            font-size: 12px;
            font-weight: 600;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .badge::before {
            content: '';
            width: 8px;
            height: 8px;
            background: #10b981;
            border-radius: 50%;
            box-shadow: 0 0 8px #10b981;
        }

        .main-grid {
            display: grid;
            grid-template-columns: 1fr 400px;
            gap: 16px;
        }

        @media (max-width: 900px) {
            .main-grid {
                grid-template-columns: 1fr;
            }
        }

        .card {
            background: var(--card-bg);
            backdrop-filter: blur(12px);
            border: 1px solid var(--border);
            border-radius: 16px;
            padding: 20px;
            display: flex;
            flex-direction: column;
            gap: 14px;
        }

        .card-title {
            font-size: 14px;
            font-weight: 600;
            color: #38bdf8;
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .toolbar {
            display: flex;
            gap: 8px;
            align-items: center;
            flex-wrap: wrap;
        }

        button {
            cursor: pointer;
            border: none;
            outline: none;
            padding: 8px 14px;
            border-radius: 8px;
            font-weight: 500;
            font-size: 13px;
            transition: all 0.2s ease;
            display: inline-flex;
            align-items: center;
            gap: 6px;
        }

        .btn-primary {
            background: linear-gradient(135deg, #4f46e5 0%, #6366f1 100%);
            color: #fff;
            padding: 12px 20px;
            font-size: 14px;
            font-weight: 600;
            border-radius: 10px;
            box-shadow: 0 4px 14px rgba(99, 102, 241, 0.35);
        }

        .btn-primary:hover {
            transform: translateY(-1px);
            box-shadow: 0 6px 20px rgba(99, 102, 241, 0.45);
        }

        .btn-secondary {
            background: #1e293b;
            color: #cbd5e1;
            border: 1px solid #334155;
        }

        .btn-secondary:hover {
            background: #27354a;
            color: #fff;
        }

        .btn-success {
            background: #059669;
            color: #fff;
        }

        .btn-success:hover {
            background: #047857;
        }

        textarea {
            width: 100%;
            height: 180px;
            background: #0d1322;
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 12px;
            color: #fff;
            font-size: 14px;
            line-height: 1.6;
            resize: vertical;
            outline: none;
        }

        textarea:focus {
            border-color: var(--accent);
            box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.2);
        }

        .stats-bar {
            display: flex;
            justify-content: space-between;
            font-size: 12px;
            color: var(--text-muted);
        }

        select, input[type="text"] {
            background: #0d1322;
            border: 1px solid var(--border);
            color: #fff;
            padding: 8px 12px;
            border-radius: 8px;
            font-size: 13px;
            outline: none;
            width: 100%;
        }

        select:focus, input[type="text"]:focus {
            border-color: var(--accent);
        }

        .form-group {
            display: flex;
            flex-direction: column;
            gap: 6px;
        }

        label {
            font-size: 12px;
            color: var(--text-muted);
            font-weight: 500;
        }

        .slider-row {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        input[type="range"] {
            flex: 1;
            accent-color: var(--cyan);
            cursor: pointer;
        }

        .slider-val {
            font-size: 12px;
            font-weight: 600;
            color: #38bdf8;
            min-width: 45px;
            text-align: right;
        }

        .voice-list {
            height: 220px;
            overflow-y: auto;
            border: 1px solid var(--border);
            border-radius: 10px;
            background: #0d1322;
            padding: 4px;
            display: flex;
            flex-direction: column;
            gap: 4px;
        }

        .voice-item {
            padding: 8px 12px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 13px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            transition: all 0.15s ease;
            border: 1px solid transparent;
        }

        .voice-item:hover {
            background: #1a253d;
        }

        .voice-item.selected {
            background: #2e2a72;
            border-color: var(--accent);
            color: #fff;
            font-weight: 600;
        }

        .voice-badge {
            font-size: 11px;
            padding: 2px 6px;
            border-radius: 4px;
            background: #1e293b;
            color: #94a3b8;
        }

        .player-box {
            background: #0d1322;
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        canvas {
            width: 100%;
            height: 50px;
            border-radius: 8px;
            background: #070a12;
        }

        audio {
            width: 100%;
            height: 40px;
            outline: none;
            border-radius: 8px;
        }

        .spinner {
            display: none;
            width: 18px;
            height: 18px;
            border: 2px solid rgba(255,255,255,0.3);
            border-top: 2px solid #fff;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
        }

        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }

        .loading .spinner {
            display: inline-block;
        }

        .loading .btn-text {
            opacity: 0.8;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- Шапка -->
        <header>
            <div class="logo-area">
                <span class="logo-icon">🎙️</span>
                <div>
                    <h1>Kokoro TTS Web Studio</h1>
                    <div class="subtitle">Нейросетевой синтез речи высокого качества • 54 голоса • 24 кГц</div>
                </div>
            </div>
            <div class="badge">Модель Kokoro ONNX активна</div>
        </header>

        <!-- Основная сетка -->
        <div class="main-grid">
            <!-- Левая колонка: Текст и Плеер -->
            <div style="display: flex; flex-direction: column; gap: 16px;">
                <div class="card">
                    <div class="card-title">
                        <span>📝 Текст для озвучки</span>
                    </div>

                    <div class="toolbar">
                        <select id="sampleSelect" style="max-width: 220px;" onchange="loadSample(this.value)">
                            <option value="">— Выбрать готовый текст —</option>
                            <option value="ru_dialog">Русский диалог</option>
                            <option value="audiobook">Аудиокнига / Природа</option>
                            <option value="tech">Технический обзор</option>
                            <option value="en_quote">Английская цитата</option>
                            <option value="tongue">Скороговорка</option>
                        </select>
                        <button class="btn-secondary" onclick="document.getElementById('textInput').value = ''; updateStats();">🧹 Очистить</button>
                    </div>

                    <textarea id="textInput" placeholder="Введите текст для озвучивания..." oninput="updateStats()">Здравствуйте! Это персональная студия озвучки речи Kokoro TTS. Теперь нейросеть умеет говорить на русском и других языках прямо на вашем компьютере без интернета, обеспечивая естественное звучание и выразительную интонацию!</textarea>

                    <div class="stats-bar">
                        <span id="charCount">Символов: 0 | Слов: 0</span>
                        <span id="detectedLang" style="color: #38bdf8; font-weight: 600;">Язык: Русский (ru)</span>
                    </div>
                </div>

                <!-- Карточка Плеера -->
                <div class="card">
                    <div class="card-title">
                        <span>🎵 Аудиоплеер и управление</span>
                    </div>

                    <div class="player-box">
                        <canvas id="visualizerCanvas"></canvas>
                        <audio id="audioPlayer" controls style="display: none;"></audio>

                        <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
                            <button id="btnSynthesize" class="btn-primary" onclick="synthesizeSpeech()">
                                <div class="spinner"></div>
                                <span class="btn-text">⚡ Озвучить и воспроизвести</span>
                            </button>
                            <a id="downloadLink" style="display: none;" download="kokoro_speech.wav">
                                <button class="btn-success">💾 Скачать WAV</button>
                            </a>
                            <span id="statusMsg" style="font-size: 13px; color: #94a3b8;">Готов к синтезу</span>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Правая колонка: Настройки голоса -->
            <div class="card">
                <div class="card-title">
                    <span>🗣️ Выбор голоса и параметров</span>
                </div>

                <div class="form-group">
                    <label>Категория голосов:</label>
                    <select id="categoryFilter" onchange="filterVoices()">
                        <option value="all">🌟 Все категории (54)</option>
                        <option value="us_female">🇺🇸 США — Женские (11)</option>
                        <option value="us_male">🇺🇸 США — Мужские (9)</option>
                        <option value="uk">🇬🇧 Великобритания (8)</option>
                        <option value="ja">🇯🇵 Япония (5)</option>
                        <option value="zh">🇨🇳 Китай (8)</option>
                        <option value="other">🌍 Другие (Испания, Франция...)</option>
                    </select>
                </div>

                <div class="form-group">
                    <label>Поиск по голосам:</label>
                    <input type="text" id="voiceSearch" placeholder="Поиск (напр. heart, bella, диктор)..." oninput="filterVoices()">
                </div>

                <div class="voice-list" id="voiceListContainer">
                    <!-- Заполняется динамически -->
                </div>

                <div id="voiceDescription" style="font-size: 12px; color: #cbd5e1; background: #0d1322; padding: 10px; border-radius: 8px; border: 1px solid var(--border);">
                    Загрузка голосов...
                </div>

                <!-- Параметры -->
                <div class="form-group" style="margin-top: 6px;">
                    <label>Язык / Режим произношения:</label>
                    <select id="langSelect">
                        <option value="auto">✨ Автоопределение (по тексту)</option>
                        <option value="ru">🇷🇺 Русский (IPA фонетизация)</option>
                        <option value="en-us">🇺🇸 Английский (США)</option>
                        <option value="en-gb">🇬🇧 Английский (Великобритания)</option>
                        <option value="es">🇪🇸 Испанский</option>
                        <option value="fr-fr">🇫🇷 Французский</option>
                        <option value="it">🇮🇹 Итальянский</option>
                        <option value="ja">🇯🇵 Японский</option>
                        <option value="cmn">🇨🇳 Китайский</option>
                    </select>
                </div>

                <div class="form-group">
                    <div style="display: flex; justify-content: space-between;">
                        <label>Скорость речи:</label>
                        <span id="speedVal" class="slider-val">1.00x</span>
                    </div>
                    <div class="slider-row">
                        <input type="range" id="speedSlider" min="50" max="200" value="100" oninput="document.getElementById('speedVal').innerText = (this.value/100).toFixed(2) + 'x'">
                        <button class="btn-secondary" style="padding: 4px 8px; font-size: 11px;" onclick="document.getElementById('speedSlider').value = 100; document.getElementById('speedVal').innerText = '1.00x';">1.0x</button>
                    </div>
                </div>

                <!-- Смешивание -->
                <div class="form-group" style="margin-top: 4px;">
                    <label style="display: flex; align-items: center; gap: 6px; cursor: pointer;">
                        <input type="checkbox" id="chkBlend" onchange="document.getElementById('blendBox').style.display = this.checked ? 'flex' : 'none';">
                        <span>✨ Смешивание двух голосов</span>
                    </label>
                    <div id="blendBox" style="display: none; flex-direction: column; gap: 8px; background: #0d1322; padding: 10px; border-radius: 8px; border: 1px solid var(--border); margin-top: 6px;">
                        <label>Второй голос для подмешивания:</label>
                        <select id="blendVoiceSelect"></select>
                        <div style="display: flex; justify-content: space-between; font-size: 11px; color: #a78bfa;">
                            <span>Баланс:</span>
                            <span id="blendRatioVal">50% / 50%</span>
                        </div>
                        <input type="range" id="blendSlider" min="0" max="100" value="50" oninput="document.getElementById('blendRatioVal').innerText = (100 - this.value) + '% / ' + this.value + '%'">
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let allVoices = [];
        let selectedVoice = "af_heart";

        const samples = {
            ru_dialog: "Здравствуйте! Это персональная студия озвучки речи Kokoro TTS. Теперь нейросеть умеет говорить на русском и других языках прямо на вашем компьютере без интернета, обеспечивая естественное звучание и выразительную интонацию!",
            audiobook: "Вечернее солнце медленно опускалось за горизонт, окрашивая верхушки вековых сосен в багряные и золотые тона. В тишине леса слышался лишь лёгкий шелест ветра и далёкое журчание лесного ручья.",
            tech: "Модель Kokoro ONNX представляет собой компактную и высокопроизводительную нейросеть для синтеза речи. Она работает с частотой дискретизации 24 килогерца и поддерживает более пятидесяти уникальных тембров.",
            en_quote: "Kokoro is a state-of-the-art open weight text-to-speech model. It delivers remarkably natural prosody, warmth, and fast generation speed.",
            tongue: "Карл у Клары украл кораллы, а Клара у Карла украла кларнет. Шла Саша по шоссе и сосала сушку."
        };

        function loadSample(key) {
            if (samples[key]) {
                document.getElementById('textInput').value = samples[key];
                updateStats();
            }
        }

        function updateStats() {
            const text = document.getElementById('textInput').value;
            const chars = text.length;
            const words = text.trim() ? text.trim().split(/\\s+/).length : 0;
            document.getElementById('charCount').innerText = `Символов: ${chars} | Слов: ${words}`;

            const isRu = /[\\u0400-\\u04FF]/.test(text);
            document.getElementById('detectedLang').innerText = isRu ? "Язык: Русский (ru)" : "Язык: Английский / Латиница (en-us)";
        }

        async function loadVoices() {
            try {
                const res = await fetch('/api/voices');
                allVoices = await res.json();
                populateVoiceList();
                populateBlendCombo();
            } catch (e) {
                console.error("Ошибка загрузки голосов", e);
            }
        }

        function populateBlendCombo() {
            const combo = document.getElementById('blendVoiceSelect');
            combo.innerHTML = '';
            allVoices.forEach(v => {
                const opt = document.createElement('option');
                opt.value = v.id;
                opt.textContent = `${v.flag} ${v.name}`;
                if (v.id === "am_adam") opt.selected = true;
                combo.appendChild(opt);
            });
        }

        function populateVoiceList() {
            const container = document.getElementById('voiceListContainer');
            container.innerHTML = '';

            const cat = document.getElementById('categoryFilter').value;
            const query = document.getElementById('voiceSearch').value.toLowerCase().trim();

            allVoices.forEach(v => {
                const matchCat = (cat === 'all') || (cat === 'other' ? !['us_female','us_male','uk','ja','zh'].includes(v.category) : v.category === cat);
                const matchQuery = !query || v.id.toLowerCase().includes(query) || v.name.toLowerCase().includes(query) || v.desc.toLowerCase().includes(query);

                if (matchCat && matchQuery) {
                    const item = document.createElement('div');
                    item.className = 'voice-item' + (v.id === selectedVoice ? ' selected' : '');
                    item.onclick = () => selectVoice(v);

                    const genderIcon = v.gender === 'Женский' ? '🌸' : '🔷';
                    item.innerHTML = `
                        <span>${v.flag} ${v.name}</span>
                        <span class="voice-badge">${genderIcon} ${v.gender}</span>
                    `;
                    container.appendChild(item);
                }
            });

            const current = allVoices.find(v => v.id === selectedVoice) || allVoices[0];
            if (current) selectVoice(current);
        }

        function selectVoice(voice) {
            selectedVoice = voice.id;
            document.querySelectorAll('.voice-item').forEach(el => {
                el.classList.toggle('selected', el.textContent.includes(voice.name));
            });
            document.getElementById('voiceDescription').innerHTML = `
                <b>${voice.flag} ${voice.name}</b> (${voice.gender})<br>
                <span style="color: #94a3b8;">${voice.desc}</span>
            `;
        }

        function filterVoices() {
            populateVoiceList();
        }

        // Синтез речи
        async function synthesizeSpeech() {
            const text = document.getElementById('textInput').value.trim();
            if (!text) {
                alert("Пожалуйста, введите текст для озвучивания!");
                return;
            }

            const btn = document.getElementById('btnSynthesize');
            const statusMsg = document.getElementById('statusMsg');
            btn.classList.add('loading');
            btn.disabled = true;
            statusMsg.innerText = "Генерация нейросетью Kokoro...";

            const payload = {
                text: text,
                voice: selectedVoice,
                blend_voice: document.getElementById('chkBlend').checked ? document.getElementById('blendVoiceSelect').value : null,
                blend_ratio: parseFloat(document.getElementById('blendSlider').value) / 100.0,
                speed: parseFloat(document.getElementById('speedSlider').value) / 100.0,
                lang: document.getElementById('langSelect').value,
                trim: true
            };

            try {
                const res = await fetch('/api/synthesize', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || "Ошибка генерации");
                }

                const blob = await res.blob();
                const audioUrl = URL.createObjectURL(blob);
                const audioPlayer = document.getElementById('audioPlayer');
                audioPlayer.src = audioUrl;
                audioPlayer.style.display = 'block';
                audioPlayer.play();

                const downloadLink = document.getElementById('downloadLink');
                downloadLink.href = audioUrl;
                downloadLink.download = `kokoro_${selectedVoice}_${Date.now()}.wav`;
                downloadLink.style.display = 'inline-block';

                statusMsg.innerText = "Готово! Воспроизведение...";
                initVisualizer(audioPlayer);
            } catch (e) {
                alert("Ошибка: " + e.message);
                statusMsg.innerText = "Ошибка синтеза";
            } finally {
                btn.classList.remove('loading');
                btn.disabled = false;
            }
        }

        // Web Audio API Визуализатор
        let audioCtx, analyser, visualizerInited = false;
        function initVisualizer(audioEl) {
            if (visualizerInited) return;
            try {
                audioCtx = new (window.AudioContext || window.webkitAudioContext)();
                analyser = audioCtx.createAnalyser();
                analyser.fftSize = 64;
                const source = audioCtx.createMediaElementSource(audioEl);
                source.connect(analyser);
                analyser.connect(audioCtx.destination);
                visualizerInited = true;
                drawVisualizer();
            } catch (e) {
                console.log("Visualizer note:", e);
            }
        }

        function drawVisualizer() {
            requestAnimationFrame(drawVisualizer);
            const canvas = document.getElementById('visualizerCanvas');
            const ctx = canvas.getContext('2d');
            if (!analyser) return;

            const bufferLength = analyser.frequencyBinCount;
            const dataArray = new Uint8Array(bufferLength);
            analyser.getByteFrequencyData(dataArray);

            ctx.clearRect(0, 0, canvas.width, canvas.height);
            const barWidth = (canvas.width / bufferLength) * 2;
            let x = 0;

            for (let i = 0; i < bufferLength; i++) {
                const barHeight = (dataArray[i] / 255) * canvas.height;
                const grad = ctx.createLinearGradient(0, canvas.height, 0, 0);
                grad.addColorStop(0, '#4f46e5');
                grad.addColorStop(0.5, '#06b6d4');
                grad.addColorStop(1, '#10b981');

                ctx.fillStyle = grad;
                ctx.fillRect(x, canvas.height - barHeight, barWidth - 2, barHeight);
                x += barWidth;
            }
        }

        // Инициализация при старте
        window.addEventListener('DOMContentLoaded', () => {
            updateStats();
            loadVoices();
            const canvas = document.getElementById('visualizerCanvas');
            canvas.width = canvas.clientWidth;
            canvas.height = canvas.clientHeight;
        });
    </script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def index():
    return HTMLResponse(content=HTML_UI)

@app.get("/api/voices")
def get_voices():
    eng = get_engine()
    available = eng.get_available_voices()
    result = []
    for vid in available:
        info = eng.get_voice_info(vid)
        flag = "🇺🇸"
        if vid.startswith(("bf_", "bm_")): flag = "🇬🇧"
        elif vid.startswith(("ef_", "em_")): flag = "🇪🇸"
        elif vid.startswith("ff_"): flag = "🇫🇷"
        elif vid.startswith(("if_", "im_")): flag = "🇮🇹"
        elif vid.startswith(("jf_", "jm_")): flag = "🇯🇵"
        elif vid.startswith(("zf_", "zm_")): flag = "🇨🇳"
        elif vid.startswith(("pf_", "pm_")): flag = "🇧🇷"
        elif vid.startswith(("hf_", "hm_")): flag = "🇮🇳"

        result.append({
            "id": vid,
            "name": info.get("name", vid),
            "category": info.get("category", "other"),
            "category_title": info.get("category_title", ""),
            "gender": info.get("gender", "Не указан"),
            "desc": info.get("desc", ""),
            "flag": flag,
        })
    return result

@app.post("/api/synthesize")
def synthesize(req: SynthRequest):
    if not req.text.strip():
        raise HTTPException(status_code=400, detail="Текст не может быть пустым")

    try:
        eng = get_engine()
        audio, sr = eng.synthesize(
            text=req.text,
            voice=req.voice,
            blend_voice=req.blend_voice,
            blend_ratio=req.blend_ratio,
            speed=req.speed,
            lang=req.lang,
            trim=req.trim,
        )

        buffer = io.BytesIO()
        sf.write(buffer, audio, sr, format="WAV")
        buffer.seek(0)
        return Response(content=buffer.read(), media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def main():
    import uvicorn
    import webbrowser
    port = 7860
    url = f"http://127.0.0.1:{port}"
    print(f"\n=======================================================")
    print(f"  Kokoro TTS Web Studio запущена!")
    print(f"  Откройте браузер по адресу: {url}")
    print(f"=======================================================\n")
    try:
        webbrowser.open(url)
    except Exception:
        pass
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="info")

if __name__ == "__main__":
    main()
