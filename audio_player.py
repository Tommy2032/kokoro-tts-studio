"""
Модуль воспроизведения звука на базе sounddevice с поддержкой паузы, перемотки, регулировки громкости и визуализации спектра.
"""

import time
import numpy as np
import soundfile as sf
import sounddevice as sd
from PyQt6.QtCore import QObject, pyqtSignal, QTimer


class AudioPlayer(QObject):
    # Сигналы Qt для обновления интерфейса
    state_changed = pyqtSignal(str)              # "playing", "paused", "stopped"
    position_changed = pyqtSignal(float, float)  # current_sec, total_sec
    waveform_levels = pyqtSignal(list)           # список уровней для визуализатора
    playback_finished = pyqtSignal()             # завершение воспроизведения
    error_occurred = pyqtSignal(str)             # ошибка аудиоустройства

    def __init__(self, parent=None):
        super().__init__(parent)
        self.audio_data: np.ndarray | None = None
        self.sample_rate: int = 24000
        self.current_frame: int = 0
        self.volume: float = 1.0  # 1.0 = 100%
        self.state: str = "stopped"  # "playing", "paused", "stopped"
        self.stream: sd.OutputStream | None = None

        # Таймер для периодического обновления позиции в UI (каждые 40мс = ~25 fps)
        self._update_timer = QTimer(self)
        self._update_timer.setInterval(40)
        self._update_timer.timeout.connect(self._on_timer_tick)

    def load(self, audio_data: np.ndarray, sample_rate: int = 24000):
        """Загрузка новых аудиоданных."""
        self.stop()
        if audio_data is None or len(audio_data) == 0:
            self.audio_data = None
            return

        # Приводим к 1D float32
        data = np.asarray(audio_data, dtype=np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)
        self.audio_data = data
        self.sample_rate = sample_rate
        self.current_frame = 0
        total_sec = len(self.audio_data) / self.sample_rate
        self.position_changed.emit(0.0, total_sec)

    @property
    def total_duration(self) -> float:
        if self.audio_data is None or len(self.audio_data) == 0:
            return 0.0
        return len(self.audio_data) / self.sample_rate

    @property
    def current_time(self) -> float:
        if self.audio_data is None or len(self.audio_data) == 0:
            return 0.0
        return self.current_frame / self.sample_rate

    def set_volume(self, volume_percent: int):
        """Громкость в процентах (0 - 200)."""
        self.volume = max(0.0, min(2.0, volume_percent / 100.0))

    def play(self):
        """Начать или продолжить воспроизведение."""
        if self.audio_data is None or len(self.audio_data) == 0:
            return

        if self.state == "playing":
            return

        if self.current_frame >= len(self.audio_data):
            self.current_frame = 0

        try:
            self._start_stream()
            self.state = "playing"
            self.state_changed.emit(self.state)
            self._update_timer.start()
        except Exception as e:
            self.error_occurred.emit(f"Не удалось запустить воспроизведение: {e}")
            self.stop()

    def pause(self):
        """Поставить на паузу."""
        if self.state != "playing":
            return

        self._stop_stream()
        self.state = "paused"
        self.state_changed.emit(self.state)
        self._update_timer.stop()

    def stop(self):
        """Полная остановка и возврат в начало."""
        self._stop_stream()
        self.current_frame = 0
        self.state = "stopped"
        self.state_changed.emit(self.state)
        self._update_timer.stop()
        self.position_changed.emit(0.0, self.total_duration)
        self.waveform_levels.emit([0.0] * 16)

    def seek(self, seconds: float):
        """Перемотка на указанную секунду."""
        if self.audio_data is None or len(self.audio_data) == 0:
            return

        target_frame = int(seconds * self.sample_rate)
        target_frame = max(0, min(len(self.audio_data), target_frame))
        self.current_frame = target_frame
        self.position_changed.emit(self.current_time, self.total_duration)

    def _audio_callback(self, outdata, frames, time_info, status):
        """Callback потока sounddevice."""
        if self.audio_data is None:
            outdata.fill(0)
            raise sd.CallbackStop

        start = self.current_frame
        end = start + frames
        total_len = len(self.audio_data)

        if start >= total_len:
            outdata.fill(0)
            raise sd.CallbackStop

        chunk = self.audio_data[start:end]
        chunk_len = len(chunk)

        # Применяем громкость
        scaled_chunk = chunk * self.volume
        # Защита от перегрузки / клиппинга
        scaled_chunk = np.clip(scaled_chunk, -1.0, 1.0)

        if chunk_len < frames:
            outdata[:chunk_len, 0] = scaled_chunk
            outdata[chunk_len:, 0] = 0
            self.current_frame = total_len
            raise sd.CallbackStop
        else:
            outdata[:, 0] = scaled_chunk
            self.current_frame = end

    def _start_stream(self):
        self._stop_stream()
        self.stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            callback=self._audio_callback,
            finished_callback=self._on_stream_finished,
        )
        self.stream.start()

    def _stop_stream(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def _on_stream_finished(self):
        """Вызывается sounddevice при завершении буфера."""
        if self.state == "playing" and self.current_frame >= (len(self.audio_data) if self.audio_data is not None else 0):
            self.state = "stopped"
            self.current_frame = 0
            # Вызываем через таймер или отложенный вызов в GUI поток
            QTimer.singleShot(0, self._emit_finished)

    def _emit_finished(self):
        self._update_timer.stop()
        self.state_changed.emit("stopped")
        self.position_changed.emit(0.0, self.total_duration)
        self.waveform_levels.emit([0.0] * 16)
        self.playback_finished.emit()

    def _on_timer_tick(self):
        """Периодическое обновление шкалы времени и уровней для анимации визуализатора."""
        total = self.total_duration
        curr = self.current_time
        self.position_changed.emit(curr, total)

        # Вычисляем мини-спектр для анимации 16 столбиков визуализатора
        if self.state == "playing" and self.audio_data is not None:
            frame = self.current_frame
            window_size = 1024
            start = max(0, frame - window_size)
            slice_data = self.audio_data[start:frame]
            if len(slice_data) > 0:
                # Разбиваем на 16 полос и считаем RMS
                sub_chunks = np.array_split(np.abs(slice_data), 16)
                levels = [float(np.mean(c) * 4.0 * self.volume) for c in sub_chunks]
                levels = [min(1.0, max(0.05, lvl)) for lvl in levels]
                self.waveform_levels.emit(levels)
            else:
                self.waveform_levels.emit([0.05] * 16)
        else:
            self.waveform_levels.emit([0.0] * 16)

    def save_wav(self, file_path: str):
        """Сохранение текущего аудио в WAV-файл."""
        if self.audio_data is None or len(self.audio_data) == 0:
            raise ValueError("Нет аудиоданных для сохранения.")
        sf.write(file_path, self.audio_data, self.sample_rate)
