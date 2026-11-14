"""
Kokoro TTS Studio — Полнофункциональное графическое приложение для синтеза речи на русском языке.
"""

import sys
import os
import time
import subprocess
from datetime import datetime
import numpy as np
import soundfile as sf

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QTextEdit, QSlider, QComboBox, QListWidget,
    QListWidgetItem, QFileDialog, QMessageBox, QProgressBar, QSplitter,
    QFrame, QGroupBox, QCheckBox, QLineEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QAbstractItemView
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, QTimer, QSize
from PyQt6.QtGui import QFont, QColor, QPainter, QBrush, QPen, QLinearGradient, QIcon

from tts_engine import TTSEngine, VOICES_CATALOG, SUPPORTED_LANGUAGES
from audio_player import AudioPlayer

# Предустановленные примеры текстов для быстрой проверки
SAMPLE_TEXTS = {
    "Русский диалог": (
        "Здравствуйте! Это персональная студия озвучки речи Kokoro TTS. "
        "Теперь нейросеть умеет говорить на русском и других языках прямо на вашем компьютере без интернета, "
        "обеспечивая естественное звучание и выразительную интонацию!"
    ),
    "Аудиокнига": (
        "Вечернее солнце медленно опускалось за горизонт, окрашивая верхушки вековых сосен в багряные и золотые тона. "
        "В тишине леса слышался лишь лёгкий шелест ветра и далёкое журчание лесного ручья."
    ),
    "Технический обзор": (
        "Модель Kokoro ONNX представляет собой компактную и высокопроизводительную нейросеть для синтеза речи. "
        "Она работает с частотой дискретизации 24 килогерца и поддерживает более пятидесяти уникальных тембров."
    ),
    "Английская цитата": (
        "Kokoro is a state-of-the-art open weight text-to-speech model. "
        "It delivers remarkably natural prosody, warmth, and fast generation speed."
    ),
    "Скороговорка": (
        "Карл у Клары украл кораллы, а Клара у Карла украла кларнет. "
        "Шла Саша по шоссе и сосала сушку."
    ),
}

DARK_STYLESHEET = """
QMainWindow, QWidget {
    background-color: #0b0f19;
    color: #e2e8f0;
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    font-size: 13px;
}

QFrame.card {
    background-color: #151d2f;
    border: 1px solid #222f47;
    border-radius: 12px;
    padding: 12px;
}

QFrame.header-card {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #1a2238, stop:1 #111827);
    border: 1px solid #2d3b59;
    border-radius: 12px;
    padding: 14px 18px;
}

QLabel {
    color: #cbd5e1;
}

QLabel.title {
    font-size: 20px;
    font-weight: 700;
    color: #ffffff;
}

QLabel.subtitle {
    font-size: 12px;
    color: #94a3b8;
}

QLabel.section-title {
    font-size: 14px;
    font-weight: 600;
    color: #38bdf8;
}

QTextEdit {
    background-color: #0d1322;
    border: 1px solid #222f47;
    border-radius: 8px;
    padding: 10px;
    color: #f1f5f9;
    font-size: 14px;
    line-height: 1.5;
    selection-background-color: #4f46e5;
}

QTextEdit:focus {
    border: 1px solid #6366f1;
}

QLineEdit {
    background-color: #0d1322;
    border: 1px solid #222f47;
    border-radius: 6px;
    padding: 7px 10px;
    color: #f1f5f9;
}

QLineEdit:focus {
    border: 1px solid #6366f1;
}

QComboBox {
    background-color: #17223b;
    border: 1px solid #2a3b5c;
    border-radius: 6px;
    padding: 6px 10px;
    color: #f1f5f9;
}

QComboBox:hover {
    border-color: #4f46e5;
}

QComboBox::drop-down {
    border: none;
    width: 20px;
}

QComboBox QAbstractItemView {
    background-color: #151d2f;
    border: 1px solid #2d3b59;
    color: #f1f5f9;
    selection-background-color: #4f46e5;
    selection-color: #ffffff;
    padding: 4px;
}

QListWidget {
    background-color: #0d1322;
    border: 1px solid #222f47;
    border-radius: 8px;
    padding: 4px;
}

QListWidget::item {
    border-radius: 6px;
    padding: 8px;
    margin-bottom: 3px;
    border: 1px solid transparent;
}

QListWidget::item:hover {
    background-color: #1a253d;
    border-color: #2e3f66;
}

QListWidget::item:selected {
    background-color: #2e2a72;
    border-color: #6366f1;
    color: #ffffff;
}

QPushButton {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 6px;
    color: #f8fafc;
    padding: 7px 14px;
    font-weight: 500;
}

QPushButton:hover {
    background-color: #293548;
    border-color: #475569;
}

QPushButton:pressed {
    background-color: #1e293b;
}

QPushButton.primary-btn {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f46e5, stop:1 #6366f1);
    border: none;
    color: #ffffff;
    font-weight: 600;
    font-size: 14px;
    padding: 10px 18px;
    border-radius: 8px;
}

QPushButton.primary-btn:hover {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4338ca, stop:1 #4f46e5);
}

QPushButton.success-btn {
    background-color: #059669;
    border: none;
    color: #ffffff;
    font-weight: 600;
}

QPushButton.success-btn:hover {
    background-color: #047857;
}

QPushButton.secondary-btn {
    background-color: #1e293b;
    border: 1px solid #334155;
    color: #cbd5e1;
}

QPushButton.secondary-btn:hover {
    background-color: #334155;
    color: #ffffff;
}

QSlider::groove:horizontal {
    height: 6px;
    background: #1e293b;
    border-radius: 3px;
}

QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f46e5, stop:1 #06b6d4);
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #38bdf8;
    border: 2px solid #0f172a;
    width: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #ffffff;
}

QProgressBar {
    background-color: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 6px;
    text-align: center;
    color: #cbd5e1;
    font-size: 11px;
    font-weight: 600;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 #4f46e5, stop:0.5 #06b6d4, stop:1 #10b981);
    border-radius: 5px;
}

QCheckBox {
    spacing: 8px;
    color: #cbd5e1;
}

QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border-radius: 4px;
    border: 1px solid #334155;
    background-color: #0d1322;
}

QCheckBox::indicator:checked {
    background-color: #4f46e5;
    border-color: #6366f1;
}

QTableWidget {
    background-color: #0d1322;
    border: 1px solid #222f47;
    border-radius: 8px;
    gridline-color: #17223b;
}

QTableWidget::item {
    padding: 6px;
    border-bottom: 1px solid #17223b;
}

QTableWidget::item:selected {
    background-color: #2e2a72;
    color: #ffffff;
}

QHeaderView::section {
    background-color: #151d2f;
    color: #94a3b8;
    padding: 6px;
    border: none;
    border-bottom: 1px solid #222f47;
    font-weight: 600;
}

QScrollBar:vertical {
    border: none;
    background: #0b0f19;
    width: 8px;
    margin: 0px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background: #2a374f;
    min-height: 20px;
    border-radius: 4px;
}

QScrollBar::handle:vertical:hover {
    background: #475569;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}
"""


class WaveformVisualizer(QWidget):
    """Визуализатор звуковых волн с плавной анимацией."""
    def __init__(self, bars_count: int = 32, parent=None):
        super().__init__(parent)
        self.bars_count = bars_count
        self.levels = [0.05] * bars_count
        self.target_levels = [0.05] * bars_count
        self.setFixedHeight(50)

        # Таймер плавной анимации спада столбиков
        self._anim_timer = QTimer(self)
        self._anim_timer.setInterval(25)
        self._anim_timer.timeout.connect(self._step_animation)
        self._anim_timer.start()

    def set_levels(self, new_levels: list[float]):
        if not new_levels:
            self.target_levels = [0.05] * self.bars_count
            return
        
        # Интерполяция до нужного количества полос
        if len(new_levels) != self.bars_count:
            x_old = np.linspace(0, 1, len(new_levels))
            x_new = np.linspace(0, 1, self.bars_count)
            res = np.interp(x_new, x_old, new_levels)
            self.target_levels = [float(val) for val in res]
        else:
            self.target_levels = [float(val) for val in new_levels]

    def _step_animation(self):
        changed = False
        for i in range(self.bars_count):
            tgt = self.target_levels[i]
            cur = self.levels[i]
            if cur < tgt:
                self.levels[i] = min(tgt, cur + 0.15)
                changed = True
            elif cur > tgt:
                self.levels[i] = max(tgt, cur - 0.05)
                changed = True
        if changed:
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()
        bar_w = (w - (self.bars_count - 1) * 3) / self.bars_count
        bar_w = max(2, bar_w)

        gradient = QLinearGradient(0, h, 0, 0)
        gradient.setColorAt(0.0, QColor("#4f46e5"))
        gradient.setColorAt(0.5, QColor("#06b6d4"))
        gradient.setColorAt(1.0, QColor("#10b981"))

        brush = QBrush(gradient)
        painter.setBrush(brush)
        painter.setPen(Qt.PenStyle.NoPen)

        for i in range(self.bars_count):
            level = max(0.06, min(1.0, self.levels[i]))
            bar_h = level * (h - 8)
            x = i * (bar_w + 3)
            y = (h - bar_h) / 2
            painter.drawRoundedRect(int(x), int(y), int(bar_w), int(bar_h), 2, 2)


class SynthesisWorker(QThread):
    """Фоновый поток синтеза речи, предотвращающий зависание интерфейса."""
    progress = pyqtSignal(int, int, str)
    finished = pyqtSignal(object, int, float)  # audio_data, sample_rate, duration_sec
    error = pyqtSignal(str)

    def __init__(
        self,
        engine: TTSEngine,
        text: str,
        voice: str,
        blend_voice: str | None,
        blend_ratio: float,
        speed: float,
        lang: str,
        trim: bool,
        parent=None,
    ):
        super().__init__(parent)
        self.engine = engine
        self.text = text
        self.voice = voice
        self.blend_voice = blend_voice
        self.blend_ratio = blend_ratio
        self.speed = speed
        self.lang = lang
        self.trim = trim
        self._is_cancelled = False

    def cancel(self):
        self._is_cancelled = True

    def run(self):
        start_time = time.time()
        try:
            def on_progress(idx, total, msg):
                if not self._is_cancelled:
                    self.progress.emit(idx, total, msg)

            audio, sr = self.engine.synthesize(
                text=self.text,
                voice=self.voice,
                blend_voice=self.blend_voice,
                blend_ratio=self.blend_ratio,
                speed=self.speed,
                lang=self.lang,
                trim=self.trim,
                progress_callback=on_progress,
                is_cancelled=lambda: self._is_cancelled,
            )

            if self._is_cancelled:
                return

            duration = len(audio) / sr
            self.finished.emit(audio, sr, duration)

        except Exception as e:
            if not self._is_cancelled:
                self.error.emit(str(e))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Kokoro TTS Studio — Синтез речи (Русский интерфейс)")
        self.resize(1180, 800)
        self.setMinimumSize(980, 680)

        # Инициализация движка и плеера
        try:
            self.engine = TTSEngine()
        except Exception as e:
            QMessageBox.critical(
                self,
                "Ошибка загрузки Kokoro",
                f"Не удалось загрузить файлы модели:\n{e}\n\nУбедитесь, что 'kokoro-v1.0.onnx' и 'voices-v1.0.bin' находятся в рабочей папке."
            )
            sys.exit(1)

        self.player = AudioPlayer(self)
        self.worker: SynthesisWorker | None = None
        self.last_synthesized_audio = None
        self.last_synthesized_sr = 24000
        self.history_items = []

        self._init_ui()
        self._connect_signals()
        self._populate_voices()

        # Загрузка примера по умолчанию
        self.text_edit.setPlainText(SAMPLE_TEXTS["Русский диалог"])
        self._update_text_stats()

    def _init_ui(self):
        self.setStyleSheet(DARK_STYLESHEET)

        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # 1. Шапка (Header)
        header_card = QFrame()
        header_card.setProperty("class", "header-card")
        header_layout = QHBoxLayout(header_card)
        header_layout.setContentsMargins(14, 10, 14, 10)

        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        title_lbl = QLabel("🎙️ Kokoro TTS Studio")
        title_lbl.setProperty("class", "title")
        subtitle_lbl = QLabel("Нейросетевой синтез речи высокой чёткости • 24 кГц • 54 голоса • Озвучивание текстов и книг")
        subtitle_lbl.setProperty("class", "subtitle")
        title_box.addWidget(title_lbl)
        title_box.addWidget(subtitle_lbl)
        header_layout.addLayout(title_box)

        header_layout.addStretch()

        # Меню быстрых примеров в шапке
        sample_lbl = QLabel("Быстрый пример:")
        header_layout.addWidget(sample_lbl)
        self.sample_combo = QComboBox()
        self.sample_combo.addItem("— Выбрать готовый текст —")
        for sample_name in SAMPLE_TEXTS.keys():
            self.sample_combo.addItem(sample_name)
        self.sample_combo.currentTextChanged.connect(self._on_sample_selected)
        header_layout.addWidget(self.sample_combo)

        badge_status = QLabel("🟢 Модель готова")
        badge_status.setStyleSheet("background-color: #064e3b; color: #34d399; padding: 4px 10px; border-radius: 6px; font-weight: 600;")
        header_layout.addWidget(badge_status)

        main_layout.addWidget(header_card)

        # 2. Центральный сплиттер (Левая панель: Текст и Плеер, Правая панель: Голоса и Настройки)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(8)

        # --- ЛЕВАЯ ПАНЕЛЬ ---
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        # Карточка текста
        text_card = QFrame()
        text_card.setProperty("class", "card")
        text_layout = QVBoxLayout(text_card)
        text_layout.setContentsMargins(12, 12, 12, 12)
        text_layout.setSpacing(8)

        # Панель инструментов текста
        tb_layout = QHBoxLayout()
        txt_title = QLabel("📝 Текст для озвучки")
        txt_title.setProperty("class", "section-title")
        tb_layout.addWidget(txt_title)
        tb_layout.addStretch()

        self.btn_load_file = QPushButton("📂 Загрузить файл...")
        self.btn_load_file.setProperty("class", "secondary-btn")
        self.btn_paste = QPushButton("📋 Вставить")
        self.btn_paste.setProperty("class", "secondary-btn")
        self.btn_clear = QPushButton("🧹 Очистить")
        self.btn_clear.setProperty("class", "secondary-btn")

        tb_layout.addWidget(self.btn_load_file)
        tb_layout.addWidget(self.btn_paste)
        tb_layout.addWidget(self.btn_clear)
        text_layout.addLayout(tb_layout)

        # Поле ввода текста
        self.text_edit = QTextEdit()
        self.text_edit.setPlaceholderText("Введите здесь текст для озвучивания (на русском, английском или других языках)...")
        text_layout.addWidget(self.text_edit)

        # Статистика текста
        stats_layout = QHBoxLayout()
        self.lbl_stats = QLabel("Символов: 0 | Слов: 0 | Примерно: 0 сек")
        self.lbl_stats.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.lbl_detected_lang = QLabel("Определённый язык: Русский (ru)")
        self.lbl_detected_lang.setStyleSheet("color: #38bdf8; font-size: 11px; font-weight: 600;")
        stats_layout.addWidget(self.lbl_stats)
        stats_layout.addStretch()
        stats_layout.addWidget(self.lbl_detected_lang)
        text_layout.addLayout(stats_layout)

        left_layout.addWidget(text_card, stretch=2)

        # Карточка плеера и синтеза
        player_card = QFrame()
        player_card.setProperty("class", "card")
        player_layout = QVBoxLayout(player_card)
        player_layout.setContentsMargins(14, 14, 14, 14)
        player_layout.setSpacing(10)

        # Верх плеера: Название трека и время
        player_header = QHBoxLayout()
        self.lbl_player_status = QLabel("Готов к озвучке")
        self.lbl_player_status.setStyleSheet("color: #38bdf8; font-weight: 600;")
        self.lbl_time = QLabel("00:00 / 00:00")
        self.lbl_time.setStyleSheet("font-family: monospace; font-size: 13px; color: #f8fafc;")
        player_header.addWidget(self.lbl_player_status)
        player_header.addStretch()
        player_header.addWidget(self.lbl_time)
        player_layout.addLayout(player_header)

        # Визуализатор звука
        self.visualizer = WaveformVisualizer(bars_count=36)
        player_layout.addWidget(self.visualizer)

        # Слайдер перемотки
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setRange(0, 1000)
        self.timeline_slider.setValue(0)
        player_layout.addWidget(self.timeline_slider)

        # Кнопки управления плеером
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(8)

        self.btn_synthesize = QPushButton("⚡ Озвучить и воспроизвести")
        self.btn_synthesize.setProperty("class", "primary-btn")
        self.btn_synthesize.setCursor(Qt.CursorShape.PointingHandCursor)

        self.btn_pause = QPushButton("⏸ Пауза")
        self.btn_pause.setEnabled(False)

        self.btn_stop = QPushButton("⏹ Стоп")
        self.btn_stop.setEnabled(False)

        controls_layout.addWidget(self.btn_synthesize, stretch=2)
        controls_layout.addWidget(self.btn_pause, stretch=1)
        controls_layout.addWidget(self.btn_stop, stretch=1)

        # Регулятор громкости
        vol_box = QHBoxLayout()
        vol_box.setSpacing(4)
        vol_ico = QLabel("🔊")
        self.vol_slider = QSlider(Qt.Orientation.Horizontal)
        self.vol_slider.setRange(0, 200)
        self.vol_slider.setValue(100)
        self.vol_slider.setFixedWidth(80)
        self.lbl_vol_val = QLabel("100%")
        self.lbl_vol_val.setFixedWidth(36)
        vol_box.addWidget(vol_ico)
        vol_box.addWidget(self.vol_slider)
        vol_box.addWidget(self.lbl_vol_val)
        controls_layout.addLayout(vol_box)

        # Кнопка сохранения в WAV
        self.btn_save_wav = QPushButton("💾 Сохранить WAV")
        self.btn_save_wav.setProperty("class", "success-btn")
        self.btn_save_wav.setEnabled(False)
        controls_layout.addWidget(self.btn_save_wav)

        player_layout.addLayout(controls_layout)

        # Прогресс-бар синтеза
        progress_layout = QHBoxLayout()
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)
        self.btn_cancel_synth = QPushButton("✕ Отмена")
        self.btn_cancel_synth.setVisible(False)
        self.btn_cancel_synth.setProperty("class", "secondary-btn")
        progress_layout.addWidget(self.progress_bar, stretch=1)
        progress_layout.addWidget(self.btn_cancel_synth)
        player_layout.addLayout(progress_layout)

        left_layout.addWidget(player_card)

        # Таблица истории генераций
        history_card = QFrame()
        history_card.setProperty("class", "card")
        history_layout = QVBoxLayout(history_card)
        history_layout.setContentsMargins(10, 8, 10, 8)
        history_layout.setSpacing(6)

        hist_header = QHBoxLayout()
        hist_title = QLabel("🕒 История недавних озвучек")
        hist_title.setStyleSheet("font-size: 12px; font-weight: 600; color: #94a3b8;")
        self.btn_open_folder = QPushButton("📁 Открыть папку программы")
        self.btn_open_folder.setStyleSheet("font-size: 11px; padding: 4px 8px;")
        hist_header.addWidget(hist_title)
        hist_header.addStretch()
        hist_header.addWidget(self.btn_open_folder)
        history_layout.addLayout(hist_header)

        self.table_history = QTableWidget(0, 4)
        self.table_history.setHorizontalHeaderLabels(["Время", "Голос", "Длительность", "Текст"])
        self.table_history.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table_history.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table_history.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table_history.horizontalHeader().setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self.table_history.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table_history.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table_history.setFixedHeight(95)
        history_layout.addWidget(self.table_history)

        left_layout.addWidget(history_card)

        splitter.addWidget(left_widget)

        # --- ПРАВАЯ ПАНЕЛЬ (НАСТРОЙКИ ГОЛОСА) ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)

        # Карточка каталога голосов
        voices_card = QFrame()
        voices_card.setProperty("class", "card")
        voices_layout = QVBoxLayout(voices_card)
        voices_layout.setContentsMargins(12, 12, 12, 12)
        voices_layout.setSpacing(8)

        v_head = QHBoxLayout()
        v_title = QLabel("🗣️ Выбор голоса")
        v_title.setProperty("class", "section-title")
        v_head.addWidget(v_title)
        v_head.addStretch()
        self.lbl_voices_count = QLabel("54 голоса")
        self.lbl_voices_count.setStyleSheet("color: #94a3b8; font-size: 11px;")
        v_head.addWidget(self.lbl_voices_count)
        voices_layout.addLayout(v_head)

        # Поиск по голосам
        self.voice_search = QLineEdit()
        self.voice_search.setPlaceholderText("🔍 Поиск голоса (напр. heart, диктор, женский)...")
        voices_layout.addWidget(self.voice_search)

        # Фильтр категорий
        self.category_combo = QComboBox()
        self.category_combo.addItem("🌟 Все категории (54)", "all")
        self.category_combo.addItem("🇺🇸 США — Женские (11)", "us_female")
        self.category_combo.addItem("🇺🇸 США — Мужские (9)", "us_male")
        self.category_combo.addItem("🇬🇧 Великобритания (8)", "uk")
        self.category_combo.addItem("🇯🇵 Япония (5)", "ja")
        self.category_combo.addItem("🇨🇳 Китай (8)", "zh")
        self.category_combo.addItem("🌍 Другие языки (Испания, Франция, Италия...)", "other")
        voices_layout.addWidget(self.category_combo)

        # Список голосов
        self.voice_list = QListWidget()
        voices_layout.addWidget(self.voice_list)

        # Описание выбранного голоса
        self.lbl_voice_desc = QLabel("Выберите голос из списка выше.")
        self.lbl_voice_desc.setWordWrap(True)
        self.lbl_voice_desc.setStyleSheet("background-color: #0d1322; border: 1px solid #222f47; border-radius: 6px; padding: 8px; color: #cbd5e1; font-size: 12px;")
        voices_layout.addWidget(self.lbl_voice_desc)

        # Кнопка прослушивания образца
        self.btn_preview_voice = QPushButton("🎧 Прослушать образец этого голоса")
        self.btn_preview_voice.setProperty("class", "secondary-btn")
        voices_layout.addWidget(self.btn_preview_voice)

        right_layout.addWidget(voices_card, stretch=2)

        # Карточка тонкой настройки параметров речи
        params_card = QFrame()
        params_card.setProperty("class", "card")
        params_layout = QVBoxLayout(params_card)
        params_layout.setContentsMargins(12, 12, 12, 12)
        params_layout.setSpacing(10)

        p_title = QLabel("⚙️ Параметры речи и тембра")
        p_title.setProperty("class", "section-title")
        params_layout.addWidget(p_title)

        # Язык синтеза
        lang_box = QVBoxLayout()
        lang_box.setSpacing(4)
        lang_box.addWidget(QLabel("Язык / Режим произношения:"))
        self.lang_combo = QComboBox()
        for code, label in SUPPORTED_LANGUAGES:
            self.lang_combo.addItem(label, code)
        lang_box.addWidget(self.lang_combo)
        params_layout.addLayout(lang_box)

        # Скорость речи (Speed)
        speed_box = QVBoxLayout()
        speed_box.setSpacing(4)
        speed_header = QHBoxLayout()
        speed_header.addWidget(QLabel("Скорость речи:"))
        self.lbl_speed_val = QLabel("1.00x")
        self.lbl_speed_val.setStyleSheet("color: #38bdf8; font-weight: 600;")
        speed_header.addStretch()
        speed_header.addWidget(self.lbl_speed_val)
        self.btn_reset_speed = QPushButton("Сброс (1.0x)")
        self.btn_reset_speed.setStyleSheet("font-size: 10px; padding: 2px 6px;")
        speed_header.addWidget(self.btn_reset_speed)
        speed_box.addLayout(speed_header)

        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setRange(50, 200)  # 0.5x - 2.0x
        self.speed_slider.setValue(100)
        speed_box.addWidget(self.speed_slider)
        params_layout.addLayout(speed_box)

        # Смешивание голосов (Voice Blending)
        blend_box = QVBoxLayout()
        blend_box.setSpacing(6)
        self.chk_blend = QCheckBox("✨ Смешивание двух голосов (Создать свой тембр)")
        blend_box.addWidget(self.chk_blend)

        self.blend_widget = QWidget()
        bw_layout = QVBoxLayout(self.blend_widget)
        bw_layout.setContentsMargins(0, 0, 0, 0)
        bw_layout.setSpacing(6)

        blend_sel_layout = QHBoxLayout()
        blend_sel_layout.addWidget(QLabel("Второй голос:"))
        self.blend_voice_combo = QComboBox()
        blend_sel_layout.addWidget(self.blend_voice_combo, stretch=1)
        bw_layout.addLayout(blend_sel_layout)

        blend_ratio_header = QHBoxLayout()
        self.lbl_blend_ratio = QLabel("Пропорция: 50% / 50%")
        self.lbl_blend_ratio.setStyleSheet("color: #a78bfa; font-size: 11px;")
        blend_ratio_header.addWidget(QLabel("Баланс тембров:"))
        blend_ratio_header.addStretch()
        blend_ratio_header.addWidget(self.lbl_blend_ratio)
        bw_layout.addLayout(blend_ratio_header)

        self.blend_slider = QSlider(Qt.Orientation.Horizontal)
        self.blend_slider.setRange(0, 100)
        self.blend_slider.setValue(50)
        bw_layout.addWidget(self.blend_slider)

        self.blend_widget.setVisible(False)
        blend_box.addWidget(self.blend_widget)
        params_layout.addLayout(blend_box)

        # Опция удаления тишины
        self.chk_trim = QCheckBox("Удалять краевую тишину (Trim silence)")
        self.chk_trim.setChecked(True)
        params_layout.addWidget(self.chk_trim)

        # Автовоспроизведение
        self.chk_autoplay = QCheckBox("Автоматически воспроизводить после генерации")
        self.chk_autoplay.setChecked(True)
        params_layout.addWidget(self.chk_autoplay)

        right_layout.addWidget(params_card)

        splitter.addWidget(right_widget)

        # Пропорция ширины панелей (60% левая, 40% правая)
        splitter.setSizes([660, 440])
        main_layout.addWidget(splitter, stretch=1)

    def _connect_signals(self):
        # Текст
        self.text_edit.textChanged.connect(self._update_text_stats)
        self.btn_load_file.clicked.connect(self._on_load_file)
        self.btn_paste.clicked.connect(self.text_edit.paste)
        self.btn_clear.clicked.connect(self.text_edit.clear)

        # Поиск и фильтрация голосов
        self.voice_search.textChanged.connect(self._filter_voices)
        self.category_combo.currentIndexChanged.connect(self._filter_voices)
        self.voice_list.currentItemChanged.connect(self._on_voice_selected)
        self.btn_preview_voice.clicked.connect(self._on_preview_voice)

        # Смешивание
        self.chk_blend.toggled.connect(self._on_blend_toggled)
        self.blend_slider.valueChanged.connect(self._on_blend_slider_changed)

        # Скорость
        self.speed_slider.valueChanged.connect(self._on_speed_changed)
        self.btn_reset_speed.clicked.connect(lambda: self.speed_slider.setValue(100))

        # Плеер
        self.btn_synthesize.clicked.connect(self._on_start_synthesis)
        self.btn_pause.clicked.connect(self._on_pause_toggle)
        self.btn_stop.clicked.connect(self.player.stop)
        self.btn_cancel_synth.clicked.connect(self._on_cancel_synthesis)
        self.btn_save_wav.clicked.connect(self._on_save_wav)
        self.btn_open_folder.clicked.connect(self._on_open_folder)

        self.vol_slider.valueChanged.connect(self._on_volume_changed)
        self.timeline_slider.sliderMoved.connect(self._on_seek_moved)

        # Сигналы AudioPlayer
        self.player.state_changed.connect(self._on_player_state_changed)
        self.player.position_changed.connect(self._on_player_position_changed)
        self.player.waveform_levels.connect(self.visualizer.set_levels)
        self.player.playback_finished.connect(self._on_playback_finished)
        self.player.error_occurred.connect(self._on_player_error)

        # Таблица истории
        self.table_history.cellDoubleClicked.connect(self._on_history_item_clicked)

    def _populate_voices(self):
        self.voice_list.clear()
        self.blend_voice_combo.clear()

        all_voices = self.engine.get_available_voices()

        for voice_id in all_voices:
            info = self.engine.get_voice_info(voice_id)
            flag = "🇺🇸"
            if voice_id.startswith(("bf_", "bm_")):
                flag = "🇬🇧"
            elif voice_id.startswith(("ef_", "em_")):
                flag = "🇪🇸"
            elif voice_id.startswith("ff_"):
                flag = "🇫🇷"
            elif voice_id.startswith(("if_", "im_")):
                flag = "🇮🇹"
            elif voice_id.startswith(("jf_", "jm_")):
                flag = "🇯🇵"
            elif voice_id.startswith(("zf_", "zm_")):
                flag = "🇨🇳"
            elif voice_id.startswith(("pf_", "pm_")):
                flag = "🇧🇷"
            elif voice_id.startswith(("hf_", "hm_")):
                flag = "🇮🇳"

            gender_icon = "🌸" if info.get("gender") == "Женский" else "🔷"
            display_text = f"{flag} {info.get('name', voice_id)}  ({gender_icon} {info.get('gender', '')})"

            item = QListWidgetItem(display_text)
            item.setData(Qt.ItemDataRole.UserRole, voice_id)
            item.setToolTip(f"{info.get('desc', '')}\nЯзык: {info.get('lang', '')}")
            self.voice_list.addItem(item)

            self.blend_voice_combo.addItem(f"{flag} {voice_id}", voice_id)

        # По умолчанию выбираем af_heart
        for i in range(self.voice_list.count()):
            item = self.voice_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == "af_heart":
                self.voice_list.setCurrentItem(item)
                break
        if self.voice_list.currentItem() is None and self.voice_list.count() > 0:
            self.voice_list.setCurrentRow(0)

        # Второй голос для смешивания по умолчанию am_adam
        idx = self.blend_voice_combo.findData("am_adam")
        if idx >= 0:
            self.blend_voice_combo.setCurrentIndex(idx)

    def _filter_voices(self):
        query = self.voice_search.text().lower().strip()
        cat = self.category_combo.currentData()

        visible_count = 0
        for i in range(self.voice_list.count()):
            item = self.voice_list.item(i)
            voice_id = item.data(Qt.ItemDataRole.UserRole)
            info = self.engine.get_voice_info(voice_id)

            match_query = (
                not query
                or query in voice_id.lower()
                or query in info.get("name", "").lower()
                or query in info.get("desc", "").lower()
                or query in info.get("gender", "").lower()
            )

            match_cat = True
            if cat != "all":
                if cat == "other":
                    match_cat = info.get("category") not in ("us_female", "us_male", "uk", "ja", "zh")
                else:
                    match_cat = (info.get("category") == cat)

            visible = match_query and match_cat
            item.setHidden(not visible)
            if visible:
                visible_count += 1

        self.lbl_voices_count.setText(f"{visible_count} из {self.voice_list.count()} голосов")

    def _on_voice_selected(self, current, previous):
        if not current:
            return
        voice_id = current.data(Qt.ItemDataRole.UserRole)
        info = self.engine.get_voice_info(voice_id)
        desc_text = f"<b>{info.get('name', voice_id)}</b> — {info.get('gender', '')} голос<br>"
        desc_text += f"<span style='color: #94a3b8;'>{info.get('desc', '')}</span><br>"
        desc_text += f"<span style='color: #38bdf8;'>Категория:</span> {info.get('category_title', '')}"
        self.lbl_voice_desc.setText(desc_text)

    def _on_blend_toggled(self, checked: bool):
        self.blend_widget.setVisible(checked)

    def _on_blend_slider_changed(self, value: int):
        val_b = value
        val_a = 100 - val_b
        self.lbl_blend_ratio.setText(f"Основной: {val_a}% / Второй: {val_b}%")

    def _on_speed_changed(self, value: int):
        speed = value / 100.0
        self.lbl_speed_val.setText(f"{speed:.2f}x")

    def _on_volume_changed(self, value: int):
        self.player.set_volume(value)
        self.lbl_vol_val.setText(f"{value}%")

    def _update_text_stats(self):
        text = self.text_edit.toPlainText()
        char_count = len(text)
        word_count = len(text.split())
        # Примерный расчёт длительности: ~120 слов в минуту = 2 слова в сек
        speed = self.speed_slider.value() / 100.0
        est_sec = int((word_count / 2.2) / max(0.5, speed)) if word_count > 0 else 0

        self.lbl_stats.setText(f"Символов: {char_count:,} | Слов: {word_count:,} | Примерно: ~{est_sec} сек")

        voice_id = "af_heart"
        if self.voice_list.currentItem():
            voice_id = self.voice_list.currentItem().data(Qt.ItemDataRole.UserRole)
        detected = self.engine.detect_language(text, voice_id)
        lang_names = {
            "ru": "Русский (ru)",
            "en-us": "Английский США (en-us)",
            "en-gb": "Английский Великобритания (en-gb)",
            "es": "Испанский (es)",
            "fr-fr": "Французский (fr-fr)",
            "it": "Итальянский (it)",
            "ja": "Японский (ja)",
            "cmn": "Китайский (cmn)",
            "pt-br": "Португальский (pt-br)",
            "hi": "Хинди (hi)",
        }
        self.lbl_detected_lang.setText(f"Определённый язык: {lang_names.get(detected, detected)}")

    def _on_sample_selected(self, sample_name: str):
        if sample_name in SAMPLE_TEXTS:
            self.text_edit.setPlainText(SAMPLE_TEXTS[sample_name])

    def _on_load_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите файл с текстом",
            os.getcwd(),
            "Текстовые файлы (*.txt *.md *.json *.py *.csv);;Все файлы (*.*)"
        )
        if file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()
                self.text_edit.setPlainText(content)
            except UnicodeDecodeError:
                try:
                    with open(file_path, "r", encoding="cp1251") as f:
                        content = f.read()
                    self.text_edit.setPlainText(content)
                except Exception as e:
                    QMessageBox.warning(self, "Ошибка чтения", f"Не удалось прочитать файл: {e}")
            except Exception as e:
                QMessageBox.warning(self, "Ошибка", f"Не удалось открыть файл: {e}")

    def _on_start_synthesis(self):
        text = self.text_edit.toPlainText().strip()
        if not text:
            QMessageBox.information(self, "Пустой текст", "Пожалуйста, введите или загрузите текст для озвучивания.")
            return

        current_item = self.voice_list.currentItem()
        if not current_item:
            QMessageBox.information(self, "Выбор голоса", "Пожалуйста, выберите голос из списка.")
            return

        voice_id = current_item.data(Qt.ItemDataRole.UserRole)
        blend_voice = None
        blend_ratio = 0.5
        if self.chk_blend.isChecked():
            blend_voice = self.blend_voice_combo.currentData()
            blend_ratio = self.blend_slider.value() / 100.0

        speed = self.speed_slider.value() / 100.0
        lang = self.lang_combo.currentData()
        trim = self.chk_trim.isChecked()

        # Блокировка UI на время синтеза
        self.btn_synthesize.setEnabled(False)
        self.btn_preview_voice.setEnabled(False)
        self.btn_cancel_synth.setVisible(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)
        self.lbl_player_status.setText("🧠 Синтез речи нейросетью Kokoro...")

        self.worker = SynthesisWorker(
            engine=self.engine,
            text=text,
            voice=voice_id,
            blend_voice=blend_voice,
            blend_ratio=blend_ratio,
            speed=speed,
            lang=lang,
            trim=trim,
            parent=self,
        )
        self.worker.progress.connect(self._on_worker_progress)
        self.worker.finished.connect(self._on_worker_finished)
        self.worker.error.connect(self._on_worker_error)
        self.worker.start()

    def _on_worker_progress(self, idx: int, total: int, msg: str):
        if total > 0:
            pct = int((idx / total) * 100)
            self.progress_bar.setValue(pct)
        self.lbl_player_status.setText(msg)

    def _on_worker_finished(self, audio: np.ndarray, sr: int, duration: float):
        self.btn_synthesize.setEnabled(True)
        self.btn_preview_voice.setEnabled(True)
        self.btn_cancel_synth.setVisible(False)
        self.progress_bar.setVisible(False)

        self.last_synthesized_audio = audio
        self.last_synthesized_sr = sr

        self.player.load(audio, sr)
        self.btn_pause.setEnabled(True)
        self.btn_stop.setEnabled(True)
        self.btn_save_wav.setEnabled(True)

        voice_name = self.voice_list.currentItem().text() if self.voice_list.currentItem() else "Kokoro"
        self.lbl_player_status.setText(f"Готово: {duration:.2f} сек ({sr // 1000} кГц)")

        # Добавление в историю
        self._add_to_history(voice_name, duration, self.text_edit.toPlainText().strip()[:80])

        # Сохранение в auto output.wav для совместимости
        try:
            sf.write("output.wav", audio, sr)
        except Exception:
            pass

        if self.chk_autoplay.isChecked():
            self.player.play()

    def _on_worker_error(self, err_msg: str):
        self.btn_synthesize.setEnabled(True)
        self.btn_preview_voice.setEnabled(True)
        self.btn_cancel_synth.setVisible(False)
        self.progress_bar.setVisible(False)
        self.lbl_player_status.setText("Ошибка при синтезе")
        QMessageBox.critical(self, "Ошибка синтеза речи", f"Произошла ошибка при генерации аудио:\n{err_msg}")

    def _on_cancel_synthesis(self):
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(1000)
            self.btn_synthesize.setEnabled(True)
            self.btn_preview_voice.setEnabled(True)
            self.btn_cancel_synth.setVisible(False)
            self.progress_bar.setVisible(False)
            self.lbl_player_status.setText("Синтез отменён")

    def _on_preview_voice(self):
        item = self.voice_list.currentItem()
        if not item:
            return
        voice_id = item.data(Qt.ItemDataRole.UserRole)
        lang = self.lang_combo.currentData()
        speed = self.speed_slider.value() / 100.0

        self.lbl_player_status.setText(f"Генерация образца для {voice_id}...")
        self.btn_preview_voice.setEnabled(False)
        QApplication.processEvents()

        try:
            audio, sr = self.engine.preview_voice(voice_id, lang=lang, speed=speed)
            self.last_synthesized_audio = audio
            self.last_synthesized_sr = sr
            self.player.load(audio, sr)
            self.btn_pause.setEnabled(True)
            self.btn_stop.setEnabled(True)
            self.btn_save_wav.setEnabled(True)
            self.lbl_player_status.setText(f"Образец: {voice_id}")
            self.player.play()
        except Exception as e:
            QMessageBox.warning(self, "Ошибка образца", f"Не удалось прослушать голос: {e}")
        finally:
            self.btn_preview_voice.setEnabled(True)

    def _on_pause_toggle(self):
        if self.player.state == "playing":
            self.player.pause()
        elif self.player.state == "paused":
            self.player.play()
        elif self.player.state == "stopped" and self.last_synthesized_audio is not None:
            self.player.play()

    def _on_seek_moved(self, value: int):
        total = self.player.total_duration
        if total > 0:
            target_sec = (value / 1000.0) * total
            self.player.seek(target_sec)

    def _on_player_state_changed(self, state: str):
        if state == "playing":
            self.btn_pause.setText("⏸ Пауза")
            self.btn_pause.setEnabled(True)
            self.btn_stop.setEnabled(True)
        elif state == "paused":
            self.btn_pause.setText("▶ Продолжить")
            self.btn_pause.setEnabled(True)
            self.btn_stop.setEnabled(True)
        elif state == "stopped":
            self.btn_pause.setText("▶ Воспроизвести")
            self.timeline_slider.setValue(0)

    def _on_player_position_changed(self, current_sec: float, total_sec: float):
        cur_m, cur_s = divmod(int(current_sec), 60)
        tot_m, tot_s = divmod(int(total_sec), 60)
        self.lbl_time.setText(f"{cur_m:02d}:{cur_s:02d} / {tot_m:02d}:{tot_s:02d}")

        if total_sec > 0:
            slider_pos = int((current_sec / total_sec) * 1000)
            self.timeline_slider.blockSignals(True)
            self.timeline_slider.setValue(slider_pos)
            self.timeline_slider.blockSignals(False)

    def _on_playback_finished(self):
        self.lbl_player_status.setText("Воспроизведение завершено")
        self.btn_pause.setText("▶ Воспроизвести")

    def _on_player_error(self, err: str):
        QMessageBox.warning(self, "Аудиоустройство", err)

    def _on_save_wav(self):
        if self.last_synthesized_audio is None:
            QMessageBox.information(self, "Нет аудио", "Сначала озвучьте текст.")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        voice_id = self.voice_list.currentItem().data(Qt.ItemDataRole.UserRole) if self.voice_list.currentItem() else "voice"
        default_name = f"kokoro_{voice_id}_{timestamp}.wav"

        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Сохранить аудиозапись в WAV",
            os.path.join(os.getcwd(), default_name),
            "Аудиофайлы WAV (*.wav)"
        )
        if file_path:
            try:
                self.player.save_wav(file_path)
                QMessageBox.information(self, "Сохранено", f"Файл успешно сохранён:\n{file_path}")
            except Exception as e:
                QMessageBox.critical(self, "Ошибка сохранения", f"Не удалось сохранить файл:\n{e}")

    def _on_open_folder(self):
        folder = os.getcwd()
        try:
            if sys.platform == "win32":
                os.startfile(folder)
            else:
                subprocess.Popen(["xdg-open", folder])
        except Exception as e:
            QMessageBox.warning(self, "Ошибка", f"Не удалось открыть папку: {e}")

    def _add_to_history(self, voice_name: str, duration: float, snippet: str):
        now_str = datetime.now().strftime("%H:%M:%S")
        dur_str = f"{duration:.1f} с"

        row = self.table_history.rowCount()
        self.table_history.insertRow(row)
        self.table_history.setItem(row, 0, QTableWidgetItem(now_str))
        self.table_history.setItem(row, 1, QTableWidgetItem(voice_name))
        self.table_history.setItem(row, 2, QTableWidgetItem(dur_str))
        self.table_history.setItem(row, 3, QTableWidgetItem(snippet + "..."))

        # Сохраняем ссылку на аудиокопию
        self.history_items.append((self.last_synthesized_audio.copy(), self.last_synthesized_sr))
        self.table_history.scrollToBottom()

    def _on_history_item_clicked(self, row: int, col: int):
        if 0 <= row < len(self.history_items):
            audio, sr = self.history_items[row]
            self.last_synthesized_audio = audio
            self.last_synthesized_sr = sr
            self.player.load(audio, sr)
            self.btn_pause.setEnabled(True)
            self.btn_stop.setEnabled(True)
            self.btn_save_wav.setEnabled(True)
            self.player.play()


def main():
    try:
        os.environ["QT_AUTO_SCREEN_SCALE_FACTOR"] = "1"
        app = QApplication(sys.argv)
        app.setApplicationName("Kokoro TTS Studio")

        window = MainWindow()
        window.show()
        sys.exit(app.exec())
    except Exception as e:
        import traceback
        err_str = traceback.format_exc()
        try:
            with open("error.log", "w", encoding="utf-8") as f:
                f.write(err_str)
        except Exception:
            pass
        print("ОШИБКА ЗАПУСКА KOKORO TTS STUDIO:")
        print(err_str)
        try:
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(None, "Ошибка запуска Kokoro TTS", f"Произошла ошибка при запуске программы:\n\n{e}\n\nПолный лог сохранен в файл error.log")
        except Exception:
            pass


if __name__ == "__main__":
    main()

