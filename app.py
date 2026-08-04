from __future__ import annotations

import gc
import io
import os
import re
import sys
import tempfile
import wave
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.sax.saxutils import escape

# Все модели загружаются только с локального диска. Сетевые обращения и
# телеметрия используемых библиотек отключены до их импорта.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_DISABLE_UPDATE_CHECK"] = "1"
os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["PYANNOTE_METRICS_ENABLED"] = "0"

import sounddevice as sd
from faster_whisper import WhisperModel
from PyQt6.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt6.QtGui import QFont, QIcon, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


if getattr(sys, "frozen", False):
    PROJECT_DIR = Path(sys.executable).resolve().parent
else:
    PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
OUTPUT_DIR = PROJECT_DIR / "transcripts"
DEFAULT_MODEL_DIR = MODELS_DIR / "faster-whisper-small"
DEFAULT_DIARIZATION_MODEL_DIR = MODELS_DIR / "speaker-diarization-community-1"
WHISPER_REQUIRED_FILES = (
    "model.bin",
)
DIARIZATION_REQUIRED_FILES = (
    "config.yaml",
    "embedding/pytorch_model.bin",
    "segmentation/pytorch_model.bin",
    "plda/plda.npz",
    "plda/xvec_transform.npz",
)
RECORDING_SAMPLE_RATE = 16000
LOGO_RELATIVE_PATH = Path("assets") / "academy_logo.png"
ICON_RELATIVE_PATH = Path("assets") / "app_icon.ico"

SPEAKER_NAMES = {
    "SPEAKER_00": "Говорящий №1",
    "SPEAKER_01": "Говорящий №2",
    "SPEAKER_02": "Говорящий №3",
    "SPEAKER_03": "Говорящий №4",
    "SPEAKER_04": "Говорящий №5",
    "SPEAKER_05": "Говорящий №6",
    "SPEAKER_06": "Говорящий №7",
    "SPEAKER_07": "Говорящий №8",
    "SPEAKER_08": "Говорящий №9",
    "SPEAKER_09": "Говорящий №10",
}

ACADEMY_NAME = "Военная академия защиты информации\n14 кафедра (основ информационной безопасности и моделирования угроз\nв информационной сфере) 1 факультета (организации защиты государственной тайны)"
APP_TITLE = "Система распознавания речи"
APP_SUBTITLE = (
    "Локальная транскрибация и диаризация: загрузите аудиофайл или "
    "запишите голос, выберите модели и формат результата."
)
AUTHORS_TEXT = "Авторы:     Краснов В.А.    Шабля В.О.    Брянцев А.Ю."

AUDIO_EXTENSIONS = {
    ".aac",
    ".aiff",
    ".amr",
    ".flac",
    ".m4a",
    ".mp3",
    ".mp4",
    ".ogg",
    ".opus",
    ".wav",
    ".webm",
    ".wma",
}


@dataclass(frozen=True)
class TimedWord:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptSegment:
    start: float
    end: float
    text: str
    words: tuple[TimedWord, ...]


@dataclass(frozen=True)
class SpeakerTurn:
    start: float
    end: float
    speaker: str


@dataclass
class AttributedChunk:
    start: float
    end: float
    speaker: str
    text: str


def bundled_path(relative_path: Path) -> Path:
    base_dir = Path(getattr(sys, "_MEIPASS", PROJECT_DIR))
    bundled = base_dir / relative_path
    if bundled.exists():
        return bundled
    return PROJECT_DIR / relative_path


def discover_models(models_dir: Path = MODELS_DIR) -> dict[str, Path]:
    if not models_dir.exists():
        return {}

    models: dict[str, Path] = {}
    for path in sorted(models_dir.iterdir()):
        if path.is_dir() and model_files_are_complete(
            path, WHISPER_REQUIRED_FILES
        ):
            models[path.name] = path
    return models


def discover_diarization_models(
    models_dir: Path = MODELS_DIR,
) -> dict[str, Path]:
    if not models_dir.exists():
        return {}

    models: dict[str, Path] = {}
    for path in sorted(models_dir.iterdir()):
        if path.is_dir() and model_files_are_complete(
            path, DIARIZATION_REQUIRED_FILES
        ):
            models[path.name] = path
    return models


def missing_model_files(
    model_path: Path,
    required_files: tuple[str, ...],
) -> list[str]:
    return [
        relative_path
        for relative_path in required_files
        if not (model_path / relative_path).is_file()
        or (model_path / relative_path).stat().st_size == 0
    ]


def model_files_are_complete(
    model_path: Path,
    required_files: tuple[str, ...],
) -> bool:
    return not missing_model_files(model_path, required_files)


def ensure_local_model(
    model_path: Path,
    required_files: tuple[str, ...],
    model_name: str,
) -> Path:
    model_path = model_path.expanduser().resolve()
    missing = missing_model_files(model_path, required_files)
    if missing:
        raise RuntimeError(
            f"Локальная модель {model_name} отсутствует или скопирована "
            "не полностью.\n\n"
            f"Папка: {model_path}\n\n"
            "Отсутствуют файлы:\n" + "\n".join(missing)
        )
    return model_path


def format_timestamp(seconds: float) -> str:
    total_ms = int(round(seconds * 1000))
    ms = total_ms % 1000
    total_seconds = total_ms // 1000
    sec = total_seconds % 60
    minutes = (total_seconds // 60) % 60
    hours = total_seconds // 3600
    return f"{hours:02d}:{minutes:02d}:{sec:02d}.{ms:03d}"


def extension_for_format(output_format: str) -> str:
    return ".docx" if output_format == "docx" else ".txt"


def normalize_output_path(path: Path, output_format: str) -> Path:
    expected = extension_for_format(output_format)
    if not path.suffix:
        return path.with_suffix(expected)
    if output_format == "docx" and path.suffix.lower() != ".docx":
        return path.with_suffix(expected)
    if output_format == "txt" and path.suffix.lower() != ".txt":
        return path.with_suffix(expected)
    return path


def write_docx(path: Path, text: str) -> None:
    paragraphs = text.splitlines() or [""]
    body = "".join(
        f'<w:p><w:r><w:t xml:space="preserve">{escape(line)}</w:t></w:r></w:p>'
        for line in paragraphs
    )
    document_xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    {body}
    <w:sectPr>
      <w:pgSz w:w="11906" w:h="16838"/>
      <w:pgMar w:top="1440" w:right="1440" w:bottom="1440" w:left="1440"/>
    </w:sectPr>
  </w:body>
</w:document>
"""

    path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr(
            "[Content_Types].xml",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/word/document.xml" ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>
</Types>
""",
        )
        archive.writestr(
            "_rels/.rels",
            """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="word/document.xml"/>
</Relationships>
""",
        )
        archive.writestr("word/document.xml", document_xml)


def write_transcript(path: Path, text: str, output_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if output_format == "docx":
        write_docx(path, text)
    else:
        path.write_text(text + ("\n" if text else ""), encoding="utf-8-sig")


def transcribe_audio(
    model: WhisperModel,
    audio_path: Path,
    language: str | None,
    beam_size: int,
    vad_filter: bool,
    word_timestamps: bool,
) -> list[TranscriptSegment]:
    segments, _info = model.transcribe(
        str(audio_path),
        language=language,
        task="transcribe",
        beam_size=beam_size,
        vad_filter=vad_filter,
        condition_on_previous_text=True,
        word_timestamps=word_timestamps,
    )

    result: list[TranscriptSegment] = []
    for segment in segments:
        text = segment.text.strip()
        if not text:
            continue

        words: list[TimedWord] = []
        for word in getattr(segment, "words", None) or ():
            if word.start is None or word.end is None or not word.word.strip():
                continue
            words.append(
                TimedWord(
                    start=float(word.start),
                    end=float(word.end),
                    text=str(word.word),
                )
            )

        result.append(
            TranscriptSegment(
                start=float(segment.start),
                end=float(segment.end),
                text=text,
                words=tuple(words),
            )
        )

    return result


def make_rttm_uri(file_stem: str) -> str:
    uri = re.sub(r"\s+", "_", file_stem.strip())
    return uri or "audio"


def prepare_torchaudio_compatibility() -> None:
    try:
        import torchaudio
    except ImportError:
        return

    if not hasattr(torchaudio, "set_audio_backend"):
        setattr(torchaudio, "set_audio_backend", lambda _backend: None)


def create_diarization_pipeline(model_path: Path):
    try:
        import torch

        prepare_torchaudio_compatibility()
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"torchcodec is not installed correctly",
                category=UserWarning,
            )
            from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Для локальной диаризации требуется установленный pyannote.audio."
        ) from exc

    pipeline = Pipeline.from_pretrained(str(model_path))
    if pipeline is None:
        raise RuntimeError("Не удалось открыть локальную модель диаризации.")

    if torch.cuda.is_available():
        try:
            pipeline.to(torch.device("cuda"))
            return pipeline, "GPU CUDA"
        except Exception:
            pass
    pipeline.to(torch.device("cpu"))
    return pipeline, "CPU"


def decode_audio_for_diarization(audio_path: Path) -> dict[str, object]:
    try:
        import torch
        from faster_whisper.audio import decode_audio
    except ImportError as exc:
        raise RuntimeError("Не удалось загрузить локальный аудиодекодер.") from exc

    samples = decode_audio(str(audio_path), sampling_rate=RECORDING_SAMPLE_RATE)
    if samples.size == 0:
        raise RuntimeError("В аудиофайле не обнаружен звук.")

    waveform = torch.from_numpy(samples.copy()).unsqueeze(0)
    return {
        "waveform": waveform,
        "sample_rate": RECORDING_SAMPLE_RATE,
        "uri": make_rttm_uri(audio_path.stem),
    }


def diarize_audio(
    pipeline,
    audio_path: Path,
    num_speakers: int | None,
) -> tuple[list[SpeakerTurn], str]:
    options: dict[str, int] = {}
    if num_speakers is not None:
        options["num_speakers"] = num_speakers

    output = pipeline(decode_audio_for_diarization(audio_path), **options)
    annotation = getattr(output, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(output, "speaker_diarization", output)

    turns = [
        SpeakerTurn(float(turn.start), float(turn.end), str(speaker))
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda item: (item.start, item.end, item.speaker))

    rttm_buffer = io.StringIO()
    annotation.write_rttm(rttm_buffer)
    return turns, rttm_buffer.getvalue()


def interval_overlap(
    first_start: float,
    first_end: float,
    second_start: float,
    second_end: float,
) -> float:
    return max(0.0, min(first_end, second_end) - max(first_start, second_start))


def speaker_for_interval(
    start: float,
    end: float,
    turns: list[SpeakerTurn],
) -> str:
    scores: dict[str, float] = {}
    for turn in turns:
        overlap = interval_overlap(start, end, turn.start, turn.end)
        if overlap > 0:
            scores[turn.speaker] = scores.get(turn.speaker, 0.0) + overlap

    if scores:
        return max(scores, key=scores.get)
    if not turns:
        return "SPEAKER_UNKNOWN"

    midpoint = (start + end) / 2
    nearest = min(
        turns,
        key=lambda turn: (
            0.0
            if turn.start <= midpoint <= turn.end
            else min(abs(midpoint - turn.start), abs(midpoint - turn.end))
        ),
    )
    return nearest.speaker


def append_text(current: str, addition: str) -> str:
    addition = addition.strip()
    if not addition:
        return current
    if not current:
        return addition
    if addition[0] in ",.!?:;…)]}»%":
        return current + addition
    return current + " " + addition


def combine_transcript_with_speakers(
    segments: list[TranscriptSegment],
    turns: list[SpeakerTurn],
    timestamps: bool,
) -> str:
    units: list[TimedWord] = []
    for segment in segments:
        if segment.words:
            units.extend(segment.words)
        else:
            units.append(TimedWord(segment.start, segment.end, segment.text))

    chunks: list[AttributedChunk] = []
    for unit in units:
        speaker = speaker_for_interval(unit.start, unit.end, turns)
        if (
            chunks
            and chunks[-1].speaker == speaker
            and unit.start - chunks[-1].end <= 1.5
        ):
            chunks[-1].end = max(chunks[-1].end, unit.end)
            chunks[-1].text = append_text(chunks[-1].text, unit.text)
        else:
            chunks.append(
                AttributedChunk(
                    start=unit.start,
                    end=unit.end,
                    speaker=speaker,
                    text=unit.text.strip(),
                )
            )

    lines: list[str] = []
    for chunk in chunks:
        prefix = ""
        if timestamps:
            prefix = (
                f"[{format_timestamp(chunk.start)} - "
                f"{format_timestamp(chunk.end)}] "
            )
        speaker_name = SPEAKER_NAMES.get(chunk.speaker, chunk.speaker)
        lines.append(f"{prefix}{speaker_name}: {chunk.text}")
    return "\n".join(lines).strip()


def format_transcript_without_speakers(
    segments: list[TranscriptSegment],
    timestamps: bool,
) -> str:
    lines: list[str] = []
    for segment in segments:
        if timestamps:
            lines.append(
                f"[{format_timestamp(segment.start)} - "
                f"{format_timestamp(segment.end)}] {segment.text}"
            )
        else:
            lines.append(segment.text)
    return "\n".join(lines).strip()


def detect_cuda_device_count() -> int:
    try:
        import ctranslate2

        return int(ctranslate2.get_cuda_device_count())
    except Exception:
        return 0


def preferred_cuda_compute_types() -> list[str]:
    preferred = ["float16", "int8_float16", "int8", "float32"]
    try:
        import ctranslate2

        supported = ctranslate2.get_supported_compute_types("cuda")
    except Exception:
        return preferred

    return [compute_type for compute_type in preferred if compute_type in supported]


def create_whisper_model(model_path: Path) -> tuple[WhisperModel, str]:
    if detect_cuda_device_count() > 0:
        for compute_type in preferred_cuda_compute_types():
            try:
                model = WhisperModel(
                    str(model_path),
                    device="cuda",
                    compute_type=compute_type,
                )
                return model, f"GPU CUDA ({compute_type})"
            except Exception:
                continue

    model = WhisperModel(str(model_path), device="cpu", compute_type="int8")
    return model, "CPU (int8)"


class AudioRecorder:
    def __init__(self, sample_rate: int = RECORDING_SAMPLE_RATE) -> None:
        self.sample_rate = sample_rate
        self.frames: list[bytes] = []
        self.stream: sd.InputStream | None = None

    def start(self) -> None:
        self.frames = []
        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="int16",
            callback=self._callback,
        )
        self.stream.start()

    def _callback(self, indata, frames, time, status) -> None:
        del frames, time
        if status:
            print(status, file=sys.stderr)
        self.frames.append(bytes(indata))

    def stop_to_wav(self) -> Path:
        if self.stream is None:
            raise RuntimeError("Запись не была запущена.")

        self.stream.stop()
        self.stream.close()
        self.stream = None

        if not self.frames:
            raise RuntimeError("Не удалось записать звук с микрофона.")

        handle = tempfile.NamedTemporaryFile(
            prefix="whisper_recording_",
            suffix=".wav",
            delete=False,
        )
        audio_path = Path(handle.name)
        handle.close()

        with wave.open(str(audio_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(self.sample_rate)
            wav_file.writeframes(b"".join(self.frames))

        return audio_path


class TranscriptionWorker(QObject):
    finished = pyqtSignal(str, str, str, str)
    failed = pyqtSignal(str)
    progress = pyqtSignal(str)

    def __init__(
        self,
        audio_path: Path,
        model_path: Path,
        diarization_model_path: Path | None,
        use_diarization: bool,
        num_speakers: int | None,
        language: str | None,
        timestamps: bool,
        suggested_name: str,
        cleanup_audio: bool,
    ) -> None:
        super().__init__()
        self.audio_path = audio_path
        self.model_path = model_path
        self.diarization_model_path = diarization_model_path
        self.use_diarization = use_diarization
        self.num_speakers = num_speakers
        self.language = language
        self.timestamps = timestamps
        self.suggested_name = suggested_name
        self.cleanup_audio = cleanup_audio

    def run(self) -> None:
        try:
            whisper_path = ensure_local_model(
                self.model_path, WHISPER_REQUIRED_FILES, "Whisper"
            )
            self.progress.emit("Распознаю речь локальной моделью Whisper...")
            model, whisper_device = create_whisper_model(whisper_path)
            segments = transcribe_audio(
                model=model,
                audio_path=self.audio_path,
                language=self.language,
                beam_size=5,
                vad_filter=True,
                word_timestamps=self.use_diarization,
            )
            del model
            gc.collect()

            if self.use_diarization:
                if self.diarization_model_path is None:
                    raise RuntimeError("Не выбрана локальная модель диаризации.")
                diarization_path = ensure_local_model(
                    self.diarization_model_path,
                    DIARIZATION_REQUIRED_FILES,
                    "диаризации",
                )
                self.progress.emit("Определяю спикеров локальной моделью...")
                pipeline, diarization_device = create_diarization_pipeline(
                    diarization_path
                )
                turns, rttm_text = diarize_audio(
                    pipeline=pipeline,
                    audio_path=self.audio_path,
                    num_speakers=self.num_speakers,
                )
                del pipeline
                gc.collect()

                self.progress.emit("Совмещаю текст с интервалами спикеров...")
                text = combine_transcript_with_speakers(
                    segments=segments,
                    turns=turns,
                    timestamps=self.timestamps,
                )
                device_label = (
                    f"Whisper: {whisper_device}; "
                    f"диаризация: {diarization_device}"
                )
            else:
                text = format_transcript_without_speakers(
                    segments,
                    timestamps=self.timestamps,
                )
                rttm_text = ""
                device_label = f"Whisper: {whisper_device}"
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        finally:
            if self.cleanup_audio:
                try:
                    self.audio_path.unlink(missing_ok=True)
                except OSError:
                    pass

        self.finished.emit(text, rttm_text, self.suggested_name, device_label)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Система распознавания речи и спикеров v.1.2 Offline")
        icon_path = bundled_path(ICON_RELATIVE_PATH)
        if icon_path.exists():
            self.setWindowIcon(QIcon(str(icon_path)))
        self.setMinimumSize(720, 700)
        self.resize(960, 800)

        self.models = discover_models()
        self.diarization_models = discover_diarization_models()
        self.result_path: Path | None = None
        self.pending_text: str | None = None
        self.pending_rttm: str | None = None
        self.pending_name = "result"
        self.thread: QThread | None = None
        self.worker: TranscriptionWorker | None = None
        self.recorder: AudioRecorder | None = None
        self.is_recording = False
        self.ui_scale = 1.0

        self._build_ui()
        self._update_scale()
        self._load_models()
        self._load_diarization_models()
        self._update_diarization_controls()

    def _build_ui(self) -> None:
        root = QWidget()
        self.setCentralWidget(root)

        self.page_layout = QVBoxLayout(root)
        self.page_layout.setContentsMargins(28, 24, 28, 24)
        self.page_layout.setSpacing(18)

        self.header = QFrame()
        self.header.setObjectName("Header")
        self.header.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.header_layout = QHBoxLayout(self.header)
        self.header_layout.setContentsMargins(14, 10, 14, 10)
        self.header_layout.setSpacing(14)

        self.logo_pixmap = QPixmap(str(bundled_path(LOGO_RELATIVE_PATH)))
        self.logo_label = QLabel()
        self.logo_label.setObjectName("Logo")
        self.logo_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.logo_label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

        header_text_layout = QVBoxLayout()
        header_text_layout.setContentsMargins(0, 0, 0, 0)
        header_text_layout.setSpacing(4)

        self.academy_label = QLabel(ACADEMY_NAME)
        self.academy_label.setObjectName("Academy")
        self.academy_label.setWordWrap(True)
        self.title_label = QLabel(APP_TITLE)
        self.title_label.setObjectName("Title")
        self.title_label.setWordWrap(True)
        self.subtitle_label = QLabel(APP_SUBTITLE)
        self.subtitle_label.setObjectName("Subtitle")
        self.subtitle_label.setWordWrap(True)

        header_text_layout.addWidget(self.academy_label)
        header_text_layout.addWidget(self.title_label)
        header_text_layout.addWidget(self.subtitle_label)

        self.header_layout.addWidget(self.logo_label)
        self.header_layout.addLayout(header_text_layout, 1)
        self.page_layout.addWidget(self.header)

        self.card = QFrame()
        self.card.setObjectName("Card")
        self.card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.card_layout = QGridLayout(self.card)
        self.card_layout.setContentsMargins(22, 22, 22, 22)
        self.card_layout.setHorizontalSpacing(12)
        self.card_layout.setVerticalSpacing(14)
        self.card_layout.setColumnStretch(1, 1)

        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("Например: C:\\audio\\meeting.mp3")
        self.file_edit.setReadOnly(True)
        self.file_edit.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.file_button = QPushButton("Выбрать файл")
        self.file_button.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.file_button.clicked.connect(self.select_file)

        self.model_combo = QComboBox()
        self.model_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.model_button = QPushButton("Другая папка")
        self.model_button.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.model_button.clicked.connect(self.select_model_dir)

        self.diarization_model_combo = QComboBox()
        self.diarization_model_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.diarization_model_button = QPushButton("Другая папка")
        self.diarization_model_button.setSizePolicy(
            QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed
        )
        self.diarization_model_button.clicked.connect(
            self.select_diarization_model_dir
        )

        self.use_diarization_check = QCheckBox("Использовать диаризацию")
        self.use_diarization_check.setChecked(True)
        self.use_diarization_check.toggled.connect(
            self._update_diarization_controls
        )

        self.speaker_count_combo = QComboBox()
        self.speaker_count_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.speaker_count_combo.addItem("Определить автоматически", None)
        for speaker_count in range(1, 11):
            self.speaker_count_combo.addItem(str(speaker_count), speaker_count)

        self.language_combo = QComboBox()
        self.language_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.language_combo.addItems(["ru", "auto", "en"])
        self.language_combo.setCurrentText("ru")

        self.format_combo = QComboBox()
        self.format_combo.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.format_combo.addItem("TXT (.txt)", "txt")
        self.format_combo.addItem("Word (.docx)", "docx")

        self.timestamps_check = QCheckBox("Добавить таймкоды")
        self.timestamps_check.setChecked(True)
        self.timestamps_check.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.card_layout.addWidget(QLabel("Аудиофайл"), 0, 0)
        self.card_layout.addWidget(self.file_edit, 0, 1)
        self.card_layout.addWidget(self.file_button, 0, 2)
        self.card_layout.addWidget(QLabel("Модель"), 1, 0)
        self.card_layout.addWidget(self.model_combo, 1, 1)
        self.card_layout.addWidget(self.model_button, 1, 2)
        self.card_layout.addWidget(self.use_diarization_check, 2, 0)
        self.card_layout.addWidget(self.diarization_model_combo, 2, 1)
        self.card_layout.addWidget(self.diarization_model_button, 2, 2)
        self.card_layout.addWidget(QLabel("Спикеров"), 3, 0)
        self.card_layout.addWidget(self.speaker_count_combo, 3, 1)
        self.card_layout.addWidget(QLabel("Язык"), 4, 0)
        self.card_layout.addWidget(self.language_combo, 4, 1)
        self.card_layout.addWidget(QLabel("Формат"), 5, 0)
        self.card_layout.addWidget(self.format_combo, 5, 1)
        self.card_layout.addWidget(self.timestamps_check, 6, 1, 1, 2)

        self.page_layout.addWidget(self.card)

        self.action_container = QWidget()
        self.action_container.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.action_grid = QGridLayout(self.action_container)
        self.action_grid.setContentsMargins(0, 0, 0, 0)
        self.action_grid.setSpacing(10)

        self.start_button = QPushButton("Распознать")
        self.start_button.setObjectName("PrimaryButton")
        self.start_button.clicked.connect(self.start_file_transcription)

        self.record_button = QPushButton("Начать запись")
        self.record_button.setObjectName("RecordButton")
        self.record_button.clicked.connect(self.toggle_recording)

        self.save_button = QPushButton("Сохранить")
        self.save_button.clicked.connect(self.save_result)
        self.save_button.setEnabled(False)

        self.open_button = QPushButton("Открыть")
        self.open_button.clicked.connect(self.open_result)
        self.open_button.setEnabled(False)

        for button in (
            self.start_button,
            self.record_button,
            self.save_button,
            self.open_button,
        ):
            button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.action_buttons = (
            self.start_button,
            self.record_button,
            self.save_button,
            self.open_button,
        )
        self._relayout_action_buttons()
        self.page_layout.addWidget(self.action_container)

        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        self.page_layout.addWidget(self.progress)

        self.status_label = QLabel("Готово к работе.")
        self.status_label.setObjectName("Status")
        self.status_label.setWordWrap(True)
        self.page_layout.addWidget(self.status_label)
        self.page_layout.addStretch(1)

        self.footer = QFrame()
        self.footer.setObjectName("Footer")
        self.footer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.footer_layout = QHBoxLayout(self.footer)
        self.footer_layout.setContentsMargins(14, 10, 14, 10)

        self.authors_label = QLabel(AUTHORS_TEXT)
        self.authors_label.setObjectName("Authors")
        self.authors_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.authors_label.setWordWrap(True)

        self.footer_layout.addWidget(self.authors_label)
        self.page_layout.addWidget(self.footer)

    def resizeEvent(self, event) -> None:
        if hasattr(self, "page_layout"):
            self._update_scale()
        super().resizeEvent(event)

    def _update_scale(self) -> None:
        width_scale = self.width() / 960
        height_scale = self.height() / 800
        self.ui_scale = max(0.78, min(1.0, min(width_scale, height_scale)))

        self.page_layout.setContentsMargins(
            self._scaled(24),
            self._scaled(20),
            self._scaled(24),
            self._scaled(20),
        )
        self.page_layout.setSpacing(self._scaled(14))
        self.header_layout.setContentsMargins(
            self._scaled(14),
            self._scaled(10),
            self._scaled(14),
            self._scaled(10),
        )
        self.header_layout.setSpacing(self._scaled(14))
        self.card_layout.setContentsMargins(
            self._scaled(18),
            self._scaled(18),
            self._scaled(18),
            self._scaled(18),
        )
        self.card_layout.setHorizontalSpacing(self._scaled(10))
        self.card_layout.setVerticalSpacing(self._scaled(10))
        self.card_layout.setColumnMinimumWidth(0, self._scaled(88))
        self.card_layout.setColumnMinimumWidth(2, self._scaled(132))
        self._apply_style()
        self._update_widget_sizes()
        self.action_grid.setSpacing(self._scaled(8))
        self._relayout_action_buttons()
        self.footer_layout.setContentsMargins(
            self._scaled(14),
            self._scaled(10),
            self._scaled(14),
            self._scaled(10),
        )

    def _scaled(self, value: int) -> int:
        return max(1, round(value * self.ui_scale))

    def _relayout_action_buttons(self) -> None:
        buttons = getattr(self, "action_buttons", ())
        for button in buttons:
            self.action_grid.removeWidget(button)

        available_width = getattr(self, "action_container", self).width()
        if available_width <= 1:
            margins = self.page_layout.contentsMargins()
            available_width = self.width() - margins.left() - margins.right()
        required_width = sum(button.minimumWidth() for button in buttons)
        required_width += self.action_grid.spacing() * max(0, len(buttons) - 1)
        columns = 4 if available_width >= required_width else 2

        for index, button in enumerate(buttons):
            row = index // columns
            column = index % columns
            self.action_grid.addWidget(button, row, column)

        for column in range(4):
            self.action_grid.setColumnMinimumWidth(column, 0)
            self.action_grid.setColumnStretch(column, 1 if column < columns else 0)

    def _apply_style(self) -> None:
        scale = self.ui_scale
        QApplication.instance().setFont(QFont("Segoe UI", max(9, round(10 * scale))))
        academy_size = self._scaled(16)
        title_size = self._scaled(24)
        subtitle_size = self._scaled(13)
        authors_size = self._scaled(12)
        input_padding_v = self._scaled(6)
        input_padding_h = self._scaled(9)
        button_padding_v = self._scaled(7)
        button_padding_h = self._scaled(11)
        combo_drop_width = self._scaled(28)
        combo_item_height = self._scaled(28)
        progress_height = self._scaled(10)
        message_button_width = self._scaled(64)
        message_padding = self._scaled(3)
        self.setStyleSheet(
            f"""
            QMainWindow {{
                background: #f4f7fb;
            }}
            QLabel {{
                color: #1f2937;
            }}
            QLabel#Academy {{
                color: #0f2f57;
                font-size: {academy_size}px;
                font-weight: 700;
            }}
            QLabel#Title {{
                font-size: {title_size}px;
                font-weight: 700;
            }}
            QLabel#Subtitle {{
                color: #5f6b7a;
                font-size: {subtitle_size}px;
            }}
            QLabel#Status {{
                color: #526070;
            }}
            QLabel#Authors {{
                color: #334155;
                font-size: {authors_size}px;
                font-weight: 600;
            }}
            QFrame#Header, QFrame#Card, QFrame#Footer {{
                background: #ffffff;
                border: 1px solid #dde5ef;
                border-radius: 8px;
            }}
            QLabel#Logo {{
                background: transparent;
            }}
            QLineEdit, QComboBox {{
                background: #ffffff;
                border: 1px solid #cbd5e1;
                border-radius: 6px;
                padding: {input_padding_v}px {input_padding_h}px;
                color: #111827;
                selection-background-color: #2563eb;
                selection-color: #ffffff;
            }}
            QLineEdit:focus, QComboBox:focus {{
                border-color: #2563eb;
            }}
            QComboBox:disabled {{
                color: #64748b;
                background: #f1f5f9;
            }}
            QComboBox::drop-down {{
                border: 0;
                width: {combo_drop_width}px;
            }}
            QComboBox QAbstractItemView {{
                background: #ffffff;
                color: #111827;
                border: 1px solid #cbd5e1;
                selection-background-color: #dbeafe;
                selection-color: #111827;
                outline: 0;
            }}
            QComboBox QAbstractItemView::item {{
                min-height: {combo_item_height}px;
                padding: {self._scaled(6)}px {self._scaled(10)}px;
            }}
            QPushButton {{
                background: #ffffff;
                border: 1px solid #c7d2df;
                border-radius: 6px;
                padding: {button_padding_v}px {button_padding_h}px;
                color: #1f2937;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background: #eef4ff;
                border-color: #93b4e8;
            }}
            QPushButton:disabled {{
                color: #94a3b8;
                background: #f1f5f9;
            }}
            QPushButton#PrimaryButton {{
                background: #2563eb;
                border-color: #2563eb;
                color: white;
            }}
            QPushButton#PrimaryButton:hover {{
                background: #1d4ed8;
            }}
            QPushButton#RecordButton {{
                background: #059669;
                border-color: #059669;
                color: white;
            }}
            QPushButton#RecordButton:hover {{
                background: #047857;
            }}
            QPushButton#RecordButton[recording="true"] {{
                background: #dc2626;
                border-color: #dc2626;
                color: white;
            }}
            QCheckBox {{
                color: #1f2937;
                spacing: {self._scaled(8)}px;
            }}
            QProgressBar {{
                background: #e8eef6;
                border: 0;
                border-radius: 5px;
                height: {progress_height}px;
            }}
            QProgressBar::chunk {{
                background: #2563eb;
                border-radius: 5px;
            }}
            QMessageBox {{
                background: #ffffff;
                color: #111827;
            }}
            QMessageBox QLabel {{
                color: #111827;
                background: transparent;
                padding: {message_padding}px;
            }}
            QMessageBox QPushButton {{
                background: #2563eb;
                border: 1px solid #2563eb;
                border-radius: 6px;
                color: #ffffff;
                min-width: {message_button_width}px;
                padding: {self._scaled(5)}px {self._scaled(10)}px;
            }}
            QMessageBox QPushButton:hover {{
                background: #1d4ed8;
                border-color: #1d4ed8;
            }}
            """
        )

    def _update_widget_sizes(self) -> None:
        field_height = self._scaled(38)
        button_height = self._scaled(40)
        side_button_width = max(
            self._scaled(132),
            min(
                self._scaled(170),
                self.file_button.sizeHint().width(),
                self.model_button.sizeHint().width(),
                self.diarization_model_button.sizeHint().width(),
            ),
        )
        action_button_width = max(
            self._scaled(126),
            min(
                self._scaled(170),
                *(button.sizeHint().width() for button in self.action_buttons),
            ),
        )

        for widget in (
            self.file_edit,
            self.model_combo,
            self.diarization_model_combo,
            self.speaker_count_combo,
            self.language_combo,
            self.format_combo,
        ):
            widget.setMinimumHeight(field_height)
            widget.setMinimumWidth(self._scaled(170))

        for button in (
            self.file_button,
            self.model_button,
            self.diarization_model_button,
            self.start_button,
            self.record_button,
            self.save_button,
            self.open_button,
        ):
            button.setMinimumHeight(button_height)

        self.timestamps_check.setMinimumHeight(self._scaled(32))
        self.file_button.setMinimumWidth(side_button_width)
        self.model_button.setMinimumWidth(side_button_width)
        self.diarization_model_button.setMinimumWidth(side_button_width)
        for button in self.action_buttons:
            button.setMinimumWidth(action_button_width)

        logo_size = self._scaled(74)
        self.logo_label.setFixedSize(logo_size, logo_size)
        if self.logo_pixmap.isNull():
            self.logo_label.setText("")
        else:
            self.logo_label.setPixmap(
                self.logo_pixmap.scaled(
                    logo_size,
                    logo_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
        self.header.setMinimumHeight(logo_size + self._scaled(20))
        self.footer.setMinimumHeight(self._scaled(42))

        row_heights = [
            field_height,
            field_height,
            field_height,
            field_height,
            field_height,
            field_height,
            max(self._scaled(32), self.timestamps_check.sizeHint().height()),
        ]
        for row, height in enumerate(row_heights):
            self.card_layout.setRowMinimumHeight(row, height)

        card_height = (
            self.card_layout.contentsMargins().top()
            + self.card_layout.contentsMargins().bottom()
            + sum(row_heights)
            + (self.card_layout.verticalSpacing() * (len(row_heights) - 1))
        )
        self.card.setMinimumHeight(card_height)

    def _load_models(self) -> None:
        self.model_combo.clear()
        for name, path in self.models.items():
            self.model_combo.addItem(name, str(path))

        default_index = self.model_combo.findText(DEFAULT_MODEL_DIR.name)
        if default_index >= 0:
            self.model_combo.setCurrentIndex(default_index)
        elif self.model_combo.count() == 0:
            self.status_label.setText(
                "Модель Whisper не найдена. Положите её в папку models."
            )

    def _load_diarization_models(self) -> None:
        self.diarization_model_combo.clear()
        for name, path in self.diarization_models.items():
            self.diarization_model_combo.addItem(name, str(path))

        default_index = self.diarization_model_combo.findText(
            DEFAULT_DIARIZATION_MODEL_DIR.name
        )
        if default_index >= 0:
            self.diarization_model_combo.setCurrentIndex(default_index)
        elif self.diarization_model_combo.count() == 0:
            self.status_label.setText(
                "Модель диаризации не найдена. Положите её в папку models."
            )

    def _update_diarization_controls(
        self,
        _checked: bool | None = None,
    ) -> None:
        enabled = (
            self.use_diarization_check.isChecked()
            and self.use_diarization_check.isEnabled()
        )
        self.diarization_model_combo.setEnabled(enabled)
        self.diarization_model_button.setEnabled(enabled)
        self.speaker_count_combo.setEnabled(enabled)

    def select_file(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Выберите аудиофайл",
            str(PROJECT_DIR),
            "Audio files (*.mp3 *.wav *.m4a *.flac *.ogg *.opus *.webm *.mp4);;All files (*.*)",
        )
        if selected:
            self.file_edit.setText(selected)
            self.pending_text = None
            self.pending_rttm = None
            self.result_path = None
            self.save_button.setEnabled(False)
            self.open_button.setEnabled(False)

    def select_model_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку модели faster-whisper",
            str(MODELS_DIR),
        )
        if not selected:
            return

        path = Path(selected)
        missing = missing_model_files(path, WHISPER_REQUIRED_FILES)
        if missing:
            QMessageBox.warning(
                self,
                "Модель не найдена",
                "Локальная модель Whisper неполна.\n\n"
                "Отсутствуют файлы:\n" + "\n".join(missing),
            )
            return

        self.models[path.name] = path
        self._load_models()
        index = self.model_combo.findText(path.name)
        if index >= 0:
            self.model_combo.setCurrentIndex(index)

    def select_diarization_model_dir(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Выберите папку локальной модели диаризации",
            str(MODELS_DIR),
        )
        if not selected:
            return

        path = Path(selected)
        missing = missing_model_files(path, DIARIZATION_REQUIRED_FILES)
        if missing:
            QMessageBox.warning(
                self,
                "Модель не найдена",
                "Локальная модель диаризации неполна.\n\n"
                "Отсутствуют файлы:\n" + "\n".join(missing),
            )
            return

        self.diarization_models[path.name] = path
        self._load_diarization_models()
        index = self.diarization_model_combo.findText(path.name)
        if index >= 0:
            self.diarization_model_combo.setCurrentIndex(index)

    def selected_output_format(self) -> str:
        return str(self.format_combo.currentData() or "txt")

    def current_model_path(self) -> Path | None:
        data = self.model_combo.currentData()
        return Path(data) if data else None

    def current_diarization_model_path(self) -> Path | None:
        data = self.diarization_model_combo.currentData()
        return Path(data) if data else None

    def selected_num_speakers(self) -> int | None:
        data = self.speaker_count_combo.currentData()
        return int(data) if data is not None else None

    def start_file_transcription(self) -> None:
        audio_path = Path(self.file_edit.text())
        model_path = self.current_model_path()
        diarization_model_path = self.current_diarization_model_path()
        use_diarization = self.use_diarization_check.isChecked()

        if not audio_path.is_file():
            QMessageBox.warning(self, "Файл не выбран", "Выберите аудиофайл.")
            return
        if audio_path.suffix.lower() not in AUDIO_EXTENSIONS:
            QMessageBox.warning(self, "Неподдерживаемый файл", "Выберите аудиофайл.")
            return
        if model_path is None:
            QMessageBox.warning(self, "Модель не выбрана", "Выберите модель Whisper.")
            return
        if use_diarization and diarization_model_path is None:
            QMessageBox.warning(
                self, "Модель не выбрана", "Выберите модель диаризации."
            )
            return

        self.start_transcription(
            audio_path=audio_path,
            model_path=model_path,
            diarization_model_path=diarization_model_path,
            use_diarization=use_diarization,
            suggested_name=audio_path.stem,
            cleanup_audio=False,
        )

    def toggle_recording(self) -> None:
        if self.is_recording:
            self.stop_recording_and_transcribe()
        else:
            self.start_recording()

    def start_recording(self) -> None:
        if (
            self.current_model_path() is None
            or (
                self.use_diarization_check.isChecked()
                and self.current_diarization_model_path() is None
            )
        ):
            QMessageBox.warning(
                self,
                "Модель не выбрана",
                "Выберите модели Whisper и диаризации перед записью.",
            )
            return

        try:
            self.recorder = AudioRecorder()
            self.recorder.start()
        except Exception as exc:
            self.recorder = None
            QMessageBox.critical(self, "Ошибка записи", str(exc))
            return

        self.is_recording = True
        self.record_button.setText("Остановить")
        self.record_button.setProperty("recording", "true")
        self.record_button.style().unpolish(self.record_button)
        self.record_button.style().polish(self.record_button)
        self.pending_text = None
        self.pending_rttm = None
        self.result_path = None
        self.save_button.setEnabled(False)
        self.open_button.setEnabled(False)
        self.start_button.setEnabled(False)
        self.file_button.setEnabled(False)
        self.status_label.setText("Идет запись. Нажмите кнопку еще раз, чтобы остановить.")

    def stop_recording_and_transcribe(self) -> None:
        if self.recorder is None:
            self.reset_record_button()
            return

        try:
            audio_path = self.recorder.stop_to_wav()
        except Exception as exc:
            self.reset_record_button()
            QMessageBox.critical(self, "Ошибка записи", str(exc))
            return
        finally:
            self.recorder = None

        self.reset_record_button()

        model_path = self.current_model_path()
        diarization_model_path = self.current_diarization_model_path()
        use_diarization = self.use_diarization_check.isChecked()
        if model_path is None or (
            use_diarization and diarization_model_path is None
        ):
            audio_path.unlink(missing_ok=True)
            QMessageBox.warning(
                self,
                "Модель не выбрана",
                "Выберите модели Whisper и диаризации.",
            )
            return

        self.start_transcription(
            audio_path=audio_path,
            model_path=model_path,
            diarization_model_path=diarization_model_path,
            use_diarization=use_diarization,
            suggested_name="recording",
            cleanup_audio=True,
        )

    def reset_record_button(self) -> None:
        self.is_recording = False
        self.record_button.setText("Начать запись")
        self.record_button.setProperty("recording", "false")
        self.record_button.style().unpolish(self.record_button)
        self.record_button.style().polish(self.record_button)
        self.start_button.setEnabled(True)
        self.file_button.setEnabled(True)

    def start_transcription(
        self,
        audio_path: Path,
        model_path: Path,
        diarization_model_path: Path | None,
        use_diarization: bool,
        suggested_name: str,
        cleanup_audio: bool,
    ) -> None:
        language_text = self.language_combo.currentText()
        language = None if language_text == "auto" else language_text

        self.set_busy(True)
        self.status_label.setText("Проверяю локальные модели...")
        self.pending_text = None
        self.pending_rttm = None
        self.result_path = None

        self.thread = QThread()
        self.worker = TranscriptionWorker(
            audio_path=audio_path,
            model_path=model_path,
            diarization_model_path=diarization_model_path,
            use_diarization=use_diarization,
            num_speakers=self.selected_num_speakers(),
            language=language,
            timestamps=self.timestamps_check.isChecked(),
            suggested_name=suggested_name,
            cleanup_audio=cleanup_audio,
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.finished.connect(self.finish_success)
        self.worker.failed.connect(self.finish_error)
        self.worker.progress.connect(self.status_label.setText)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.finished.connect(self.clear_worker_refs)
        self.thread.start()

    def set_busy(self, busy: bool) -> None:
        self.start_button.setEnabled(not busy)
        self.record_button.setEnabled(not busy)
        self.file_button.setEnabled(not busy)
        self.model_button.setEnabled(not busy)
        self.model_combo.setEnabled(not busy)
        self.use_diarization_check.setEnabled(not busy)
        self._update_diarization_controls()
        self.language_combo.setEnabled(not busy)
        self.format_combo.setEnabled(not busy)
        self.timestamps_check.setEnabled(not busy)
        if busy:
            self.progress.setRange(0, 0)
            self.save_button.setEnabled(False)
            self.open_button.setEnabled(False)
        else:
            self.progress.setRange(0, 1)
            self.progress.setValue(0)

    def finish_success(
        self,
        text: str,
        rttm_text: str,
        suggested_name: str,
        device_label: str,
    ) -> None:
        self.pending_text = text
        self.pending_rttm = rttm_text or None
        self.pending_name = suggested_name
        if self.pending_rttm is not None:
            result_name = "Транскрибация и диаризация готовы"
        else:
            result_name = "Транскрибация готова"
        self.status_label.setText(
            f"{result_name} ({device_label}). Нажмите «Сохранить»."
        )
        self.save_button.setEnabled(True)
        self.open_button.setEnabled(False)
        self.set_busy(False)

    def finish_error(self, message: str) -> None:
        self.status_label.setText("Ошибка распознавания.")
        self.set_busy(False)
        QMessageBox.critical(self, "Ошибка", message)

    def clear_worker_refs(self) -> None:
        self.thread = None
        self.worker = None

    def save_result(self) -> None:
        if self.pending_text is None:
            QMessageBox.warning(self, "Нет результата", "Сначала распознайте аудио.")
            return

        output_format = self.selected_output_format()
        extension = extension_for_format(output_format)
        suggested = OUTPUT_DIR / f"{self.pending_name}{extension}"
        file_filter = "Word documents (*.docx)" if output_format == "docx" else "Text files (*.txt)"
        output_name, _ = QFileDialog.getSaveFileName(
            self,
            "Куда сохранить распознанный текст",
            str(suggested),
            f"{file_filter};;All files (*.*)",
        )
        if not output_name:
            return

        output_path = normalize_output_path(Path(output_name), output_format)
        try:
            write_transcript(output_path, self.pending_text, output_format)
        except Exception as exc:
            QMessageBox.critical(self, "Ошибка сохранения", str(exc))
            return

        self.result_path = output_path
        self.status_label.setText(f"Сохранено: {self.result_path}")

    def open_result(self) -> None:
        if self.result_path and self.result_path.exists():
            os.startfile(self.result_path)


def run_gui() -> int:
    app = QApplication(sys.argv)
    icon_path = bundled_path(ICON_RELATIVE_PATH)
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))
    window = MainWindow()
    window.show()
    return app.exec()


def main() -> int:
    return run_gui()


if __name__ == "__main__":
    raise SystemExit(main())
