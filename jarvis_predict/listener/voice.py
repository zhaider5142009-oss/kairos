"""
Voice listener — Whisper STT, intent parser, dispatcher bridge, push-to-talk.

Pipeline:
    1. Record audio from the default input device.
    2. Transcribe with faster-whisper (offline).
    3. Lowercase + simple regex intent parser.
    4. Dispatch intent (confirm/cancel/status/quit).
    5. Sleep poll_seconds, repeat.

Two capture modes:
    - periodic loop       — the original fixed-cadence "always listening" mode
    - capture_once()      — one-shot push-to-talk (UI voice button / hotkey):
                            records until end-of-speech, transcribes, returns
                            the text. All non-HW work the caller decides.

v1.2 hardening:
    - ``capture_lock`` serialises captures so periodic + push-to-talk can't
      record simultaneously (we never run two Whisper sessions at once).
    - adaptive end-of-speech detection for one-shot capture (InputStream with
      an RMS gate + silence hold), falling back to fixed-length capture.
    - aggressive try/except everywhere — a failed mic / missing model can
      never kill the thread.
    - ``last_text`` exposes the most recent transcription for the UI.
    - ``on_transcript`` hook is fired after every successful transcription.

Intents recognised (periodic loop only — one-shot returns raw text):
    "jarvis confirm" / "jarvis yes"      → confirm latest pending
    "jarvis cancel" / "jarvis no"        → cancel latest pending
    "jarvis cancel all" / "jarvis stop"  → cancel all pending
    "jarvis status"                      → emit a STATUS event
    "jarvis quit" / "jarvis shutdown"    → emit SHUTDOWN event
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Callable

import numpy as np

from .dispatcher import Dispatcher, EventType

LOGGER = logging.getLogger("jarvis.voice")


_CONFIRM_PATTERNS = [
    r"\b(confirm|yes|approve|go|execute|send it|do it|take it)\b",
]
_CANCEL_PATTERNS = [
    r"\b(cancel( all)?|no|stop|abort|reject|skip|kill)\b",
]
_STATUS_PATTERNS = [r"\b(status|state|how are we doing|what'?s up)\b"]
_QUIT_PATTERNS = [r"\b(quit|shutdown|exit|goodbye|stop listening)\b"]


class VoiceListener:
    """Periodic-listen + one-shot capture. Whisper loaded lazily, thread-safe."""

    def __init__(
        self,
        *,
        dispatcher: Dispatcher,
        enabled: bool = True,
        wake_word: str = "jarvis",
        model_size: str = "base",
        language: str = "en",
        listen_seconds: float = 4.0,
        sample_rate: int = 16000,
        poll_seconds: float = 6.0,
        silence_hold: float = 1.2,
        on_transcript: Callable[[str], None] | None = None,
    ) -> None:
        self.dispatcher = dispatcher
        self.enabled = enabled
        self.wake_word = wake_word.strip().lower()
        self.model_size = model_size
        self.language = language
        self.listen_seconds = max(1.0, float(listen_seconds))
        self.sample_rate = int(sample_rate)
        self.poll_seconds = max(2.0, float(poll_seconds))
        self.silence_hold = max(0.3, float(silence_hold))
        self.on_transcript = on_transcript or (lambda t: None)

        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._model = None          # loaded lazily
        self._model_error: str | None = None
        self._capture_lock = threading.Lock()
        self._in_dev: int | None = None
        self._in_rate: int = 0
        self._in_failed_at: float = 0.0
        self.last_text = ""

    # --- public ---------------------------------------------------------

    def start(self) -> None:
        if not self.enabled:
            LOGGER.info("Voice listener disabled in config")
            return
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="jarvis-voice", daemon=True)
        self._thread.start()
        LOGGER.info("VoiceListener started (model=%s, listen=%.1fs, wake=%r)",
                    self.model_size, self.listen_seconds, self.wake_word)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)

    def capture_once(self, *, timeout: float | None = None,
                     wake_word_gate: bool = False) -> str:
        """One-shot push-to-talk.

        Records until end-of-speech (silence hold) or `timeout`, transcribes,
        fires ``on_transcript`` and returns the lowercased text ("" on silence
        or error). Never raises — callers get "" and a logged reason.
        """
        if not self.enabled:
            return ""
        self._load_model()
        if self._model is None:
            return ""
        with self._capture_lock:
            audio = self._record_adaptive(timeout or max(6.0, self.listen_seconds + 2.0))
            if audio is None or audio.size < self.sample_rate * 0.3:
                return ""
            rms = float((audio ** 2).mean() ** 0.5)
            if rms < 0.005:
                return ""
            text = self._transcribe(audio)
        if not text:
            return ""
        self.last_text = text
        try:
            self.on_transcript(text)
        except Exception:
            LOGGER.debug("on_transcript hook failed", exc_info=True)
        return text

    # --- thread loop ----------------------------------------------------

    def _run(self) -> None:
        try:
            self._load_model()
        except Exception:
            LOGGER.exception("VoiceListener: failed to load Whisper; thread exiting")
            return
        while not self._stop.is_set():
            try:
                self._listen_once()
            except Exception:
                LOGGER.exception("VoiceListener loop error")
            # Throttle the periodic loop. Without this, a missing input device
            # makes _listen_once return immediately and the loop hammers the
            # audio stack + logger at 100% CPU.
            self._stop.wait(timeout=float(self.poll_seconds))

    # --- model loading --------------------------------------------------

    def _load_model(self) -> None:
        if self._model is not None or self._model_error:
            return
        try:
            from faster_whisper import WhisperModel  # type: ignore
        except ImportError as e:
            self._model_error = "faster-whisper not installed (pip install faster-whisper)"
            LOGGER.warning("Voice: %s", self._model_error)
            return
        LOGGER.info("Loading Whisper model %s (this may take a minute)…", self.model_size)
        try:
            self._model = WhisperModel(self.model_size, device="cpu", compute_type="int8")
        except Exception as e:
            self._model = None
            self._model_error = str(e)
            LOGGER.warning("Voice: Whisper failed to load: %s", e)
            return
        LOGGER.info("Whisper ready")

    # --- periodic listen -------------------------------------------------

    def _listen_once(self) -> None:
        if self._model is None:
            self._stop.wait(timeout=10.0)
            return
        with self._capture_lock:
            audio = self._record_fixed(self.listen_seconds)
            if audio is None or audio.size < self.sample_rate * 0.3:
                return
            rms = float((audio ** 2).mean() ** 0.5)
            if rms < 0.005:
                return  # silent → no inference
            text = self._transcribe(audio)
        if not text:
            return
        self.last_text = text
        LOGGER.info("Voice: %r", text)
        self._handle_utterance(text)

    def _transcribe(self, audio: np.ndarray) -> str:
        try:
            segments, _info = self._model.transcribe(
                audio, language=self.language, vad_filter=True, beam_size=1,
            )
            text = " ".join(seg.text.strip() for seg in segments).strip().lower()
        except Exception as e:
            LOGGER.debug("transcribe failed: %s", e)
            return ""
        try:
            self.on_transcript(text)
        except Exception:
            pass
        return text

    # --- recording ------------------------------------------------------

    def _resolve_input(self) -> tuple[int, int]:
        """Pick a working input device + native rate.

        Windows quirk: some backends (WDM-KS) only open at the device's native
        sample rate and reject the callback API entirely. So we:
          1. prefer the configured default input, else any input device,
          2. first try to open it at ``self.sample_rate`` (16 kHz),
          3. otherwise fall back to the device's native rate and resample.
        The result is cached per instance.
        """
        import sounddevice as sd
        if self._in_dev is not None and self._in_rate:
            return self._in_dev, self._in_rate
        # Blackout after a failed probe so we don't hammer sd.rec every cycle.
        if self._in_dev == -1 and (time.time() - self._in_failed_at) < 60.0:
            return -1, 0

        default_in = -1
        try:
            d = sd.default.device
            default_in = int(d[1]) if isinstance(d, (list, tuple)) and len(d) > 1 else int(d)
        except Exception:
            default_in = -1
        input_devs: list[int] = []
        for i, dev in enumerate(sd.query_devices()):
            if dev["max_input_channels"] > 0:
                input_devs.append(i)
        if default_in in input_devs:
            input_devs.remove(default_in)
            input_devs.insert(0, default_in)

        def opens(idx: int, sr: int) -> bool:
            try:
                a = sd.rec(int(sr * 0.4), samplerate=sr, channels=1,
                           device=idx, dtype="float32", blocking=True)
                return a is not None and a.size and bool(np.isfinite(a).all())
            except Exception:
                return False

        tried: set[int] = set()
        for i in input_devs:
            if i in tried:
                continue
            tried.add(i)
            if opens(i, self.sample_rate):
                self._in_dev, self._in_rate = i, self.sample_rate
                LOGGER.info("Voice: input device %d @ %d Hz", i, self._in_rate)
                return i, self._in_rate
        for i in input_devs:
            if i in tried:
                continue
            tried.add(i)
            try:
                native = int(sd.query_devices(i, "input")["default_samplerate"])
            except Exception:
                continue
            if native > 0 and opens(i, native):
                self._in_dev, self._in_rate = i, native
                LOGGER.info("Voice: input device %d @ native %d Hz (resampled to %d)",
                            i, native, self.sample_rate)
                return i, native
        self._in_dev = -1
        self._in_failed_at = time.time()
        LOGGER.warning("Voice: no working input device found")
        return -1, 0

    @staticmethod
    def _resample(audio: np.ndarray, from_sr: int, to_sr: int) -> np.ndarray:
        """Linear-ish resample to 16 kHz for Whisper (clamped)."""
        from_sr = int(from_sr)
        to_sr = int(to_sr)
        if from_sr == to_sr or audio.size == 0:
            return audio
        try:
            import math
            from scipy.signal import resample_poly
            g = math.gcd(from_sr, to_sr)
            return resample_poly(audio, to_sr // g, from_sr // g).astype(np.float32)
        except Exception:
            n_out = max(1, int(round(audio.size * to_sr / from_sr)))
            x = np.linspace(0.0, max(audio.size - 1, 1), n_out)
            return np.interp(x, np.arange(audio.size), audio).astype(np.float32)

    def _record_adaptive(self, timeout: float) -> np.ndarray | None:
        """InputStream record with RMS end-of-speech gate. Falls back to fixed."""
        import sounddevice as sd
        dev, sr = self._resolve_input()
        if dev < 0 or sr <= 0:
            return None
        silence_frames = int(self.silence_hold * sr)
        done = threading.Event()
        buf_lock = threading.Lock()
        buffer: list[np.ndarray] = []
        silence = 0

        def _callback(indata, frames, time_info, status) -> None:
            nonlocal silence
            chunk = indata[:, 0].copy()
            with buf_lock:
                buffer.append(chunk)
            rms = float((chunk ** 2).mean() ** 0.5)
            if rms < 0.005:
                silence += frames
            else:
                silence = 0
            if silence >= silence_frames:
                done.set()

        try:
            with sd.InputStream(
                samplerate=sr, channels=1, dtype="float32",
                device=dev, callback=_callback, blocksize=int(sr * 0.2),
            ):
                done.wait(timeout=timeout)
        except Exception as e:
            LOGGER.debug("adaptive record unsupported (%s); using fixed capture", e)
            return self._record_fixed(timeout)
        with buf_lock:
            if not buffer:
                return None
            audio = np.concatenate(buffer)
        return self._resample(audio, sr, self.sample_rate)

    def _record_fixed(self, seconds: float) -> np.ndarray | None:
        import sounddevice as sd
        dev, sr = self._resolve_input()
        if dev < 0 or sr <= 0:
            return None
        frames = int(seconds * sr)
        try:
            audio = sd.rec(frames, samplerate=sr, channels=1, device=dev,
                           dtype="float32", blocking=True)
            audio = np.asarray(audio).flatten().copy()
        except Exception as e:
            LOGGER.warning("Microphone capture failed: %s", e)
            return None
        return self._resample(audio, sr, self.sample_rate)

    # --- intent handling -------------------------------------------------

    def _handle_utterance(self, text: str) -> None:
        if self.wake_word and self.wake_word not in text:
            return
        if self.wake_word:
            text = text.replace(self.wake_word, "").strip()

        intent = _classify(text)
        if intent is None:
            return
        kind, _payload = intent
        self.dispatcher.publish(EventType.VOICE_COMMAND, {"text": text, "intent": kind}, source="voice")
        if kind == "confirm":
            self._act_on_latest("confirm", "voice")
        elif kind == "cancel":
            self._act_on_latest("cancel", "voice")
        elif kind == "cancel_all":
            self.dispatcher.cancel_all(source="voice", reason="voice:cancel all")
        elif kind == "status":
            self.dispatcher.publish(EventType.STATUS, self.dispatcher.status(), source="voice")
        elif kind == "quit":
            self.dispatcher.publish(EventType.SHUTDOWN, {"source": "voice"}, source="voice")

    def _act_on_latest(self, action: str, source: str) -> None:
        pending = self.dispatcher.pending_snapshot()
        if not pending:
            LOGGER.info("Voice %s — no pending signals", action)
            return
        latest = max(pending, key=lambda p: p["staged_at"])
        if action == "confirm":
            self.dispatcher.confirm(latest["signal_id"], source=source)
        elif action == "cancel":
            self.dispatcher.cancel(latest["signal_id"], source=source, reason="voice:cancel")


def _classify(text: str) -> tuple[str, dict] | None:
    for pat in _QUIT_PATTERNS:
        if re.search(pat, text):
            return ("quit", {})
    for pat in _CANCEL_PATTERNS:
        if re.search(pat, text):
            kind = "cancel_all" if "all" in text else "cancel"
            return (kind, {})
    for pat in _STATUS_PATTERNS:
        if re.search(pat, text):
            return ("status", {})
    for pat in _CONFIRM_PATTERNS:
        if re.search(pat, text):
            return ("confirm", {})
    return None


__all__ = ["VoiceListener"]