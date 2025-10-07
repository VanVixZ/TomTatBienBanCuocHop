# main.py
# ======================================================================================
# ONE-FILE APP: FastAPI backend + embedded HTML frontend + ASR/Diar pipeline (Windows)
# Run: python main.py  → mở http://127.0.0.1:8000/
# ======================================================================================

import os, sys, json, time, shutil, subprocess, math, gc, warnings, threading, uuid, webbrowser, re, tempfile
from pathlib import Path
from datetime import timedelta
from collections import defaultdict

# ----------------------- ENV & UTF-8 -----------------------
from dotenv import load_dotenv
load_dotenv()

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# ----------------------- CONFIG -----------------------
# gốc dữ liệu & kho giọng mẫu toàn cục (đã set trong .env)
DATA_ROOT       = Path(os.getenv("DATA_ROOT",       r"D:\Test\data_cuochop")).resolve()
SAMPLE_DIR      = Path(os.getenv("SAMPLE_DIR",      r"D:\Test\audio_label")).resolve()
SAMPLE_NORM_DIR = Path(os.getenv("SAMPLE_NORM_DIR", r"D:\Test\audio_label_norm")).resolve()

for p in [DATA_ROOT, SAMPLE_DIR, SAMPLE_NORM_DIR]:
    p.mkdir(parents=True, exist_ok=True)

HF_TOKEN = os.getenv("HF_TOKEN", "")
if not HF_TOKEN or not HF_TOKEN.startswith("hf_"):
    raise SystemExit("Missing HF_TOKEN. Set HF_TOKEN=... trong .env (token đã accept pyannote).")

# ASR
ASR_MODEL   = "medium" 
BEAM_SIZE   = 5
TRY_COMPUTE = ["float16", "float32", "int8_float16", "int8"]

# Diarization & speaker ID
DIAR_PIPELINE_ID       = "pyannote/speaker-diarization-3.1"
EMB_MODEL_ID           = "pyannote/embedding"
NUM_SPEAKERS_HINT      = None
MIN_TURN_SEC           = 1.2
SIM_THRESHOLD          = 0.62
MAJORITY_MIN_VOTES     = 2
MAJORITY_PROP          = 0.5
MERGE_GAP              = 1.2

# ----------------------- SUMMARY with GEMINI -----------------------
ENABLE_SUMMARY            = os.getenv("ENABLE_SUMMARY", "0") == "1"
GOOGLE_API_KEY            = os.getenv("GOOGLE_API_KEY", "")
GEMINI_MODEL              = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
GEMINI_TEMPERATURE        = float(os.getenv("GEMINI_TEMPERATURE", "0.2"))
GEMINI_TOP_P              = float(os.getenv("GEMINI_TOP_P", "0.9"))
GEMINI_TOP_K              = int(os.getenv("GEMINI_TOP_K", "40"))
GEMINI_MAX_OUTPUT_TOKENS  = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "65000"))

if ENABLE_SUMMARY and not GOOGLE_API_KEY:
    print("[SUM] WARNING: ENABLE_SUMMARY=1 nhưng chưa có GOOGLE_API_KEY trong env.", flush=True)

SYSTEM_INSTRUCTION = (
    "Bạn là thư ký doanh nghiệp. "
    "Mục tiêu: tạo biên bản tóm tắt chuyên nghiệp bằng TIẾNG VIỆT CÓ DẤU, súc tích, rõ ràng, "
    "phục vụ lãnh đạo và đội ngũ ra quyết định.\n"
    "- Không dùng tiếng Anh, TRỪ thuật ngữ xuất hiện trong transcript (ví dụ: sprint, backlog, KPI), "
    "nhưng vẫn diễn giải bằng tiếng Việt.\n"
    "- Không cắt giữa câu. Nếu buộc dừng do giới hạn, hãy dừng ở dấu câu gần nhất.\n"
    "- Không bịa thông tin; nếu thiếu dữ liệu, ghi [chưa rõ]. Không suy đoán tên/chức danh khi transcript không nêu.\n"
    "- Không trích nguyên văn dài (>40 từ). Ưu tiên diễn giải; riêng quyết định/action giữ nguyên tinh thần và mốc thời gian.\n"
    "- Giữ tông trang trọng, chuẩn chính tả, tránh khẩu ngữ."
)

# ----------------------- IMPORTS (after env) -----------------------
import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks, Query, UploadFile, File, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse

# ----------------------- SUMMARIZER (Gemini 2.5 Flash) -----------------------
def build_enterprise_prompt(transcript_text: str) -> str:
    return (
        "Hãy tạo BIÊN BẢN TÓM TẮT CUỘC HỌP NỘI BỘ ở dạng VĂN BẢN THUẦN (TXT). "
        "KHÔNG dùng JSON/Markdown/mã hóa/biểu tượng trang trí. "
        "Chỉ trả về nội dung biên bản, không thêm lời dẫn.\n\n"
        "YÊU CẦU ĐỊNH DẠNG & PHONG CÁCH\n"
        "- Tiêu đề: BIÊN BẢN TÓM TẮT CUỘC HỌP NỘI BỘ\n"
        "- Dòng đầu (nếu suy ra được): Ngày họp; Thời lượng; Thành phần tham dự (ghi [chưa rõ] nếu thiếu).\n"
        "- 1. TÓM TẮT ĐIỀU HÀNH: 3–5 gạch đầu dòng, ≤ 20 từ/dòng, nêu mục tiêu, tổng thể, kết quả chính.\n"
        "- 2. NỘI DUNG CHÍNH THEO MỐC THỜI GIAN:\n"
        "  + Mỗi ý một dòng ngắn; nhắc mốc thời gian dạng [HH:MM:SS] khi nêu sự kiện/ý quan trọng.\n"
        "  + Gom theo chủ đề (kỹ thuật, tiến độ, rủi ro, chi phí…) thay vì liệt kê dàn trải.\n"
        "- 3. QUYẾT ĐỊNH/PHÊ DUYỆT: mỗi dòng gồm [HH:MM:SS] – Quyết định – Lý do ngắn.\n"
        "- 4. VIỆC CẦN LÀM (ACTION ITEMS): mỗi dòng theo mẫu:\n"
        "  • [HH:MM:SS] | Phụ trách: <tên hoặc “Người Nói i”> | Việc: <mô tả ngắn> | Hạn: <dd/mm/yyyy hoặc [chưa rõ]>\n"
        "- 5. RỦI RO & PHỤ THUỘC: liệt kê rủi ro/điều kiện phụ thuộc; nêu biện pháp giảm thiểu nếu có.\n"
        "- 6. VẤN ĐỀ MỞ/CẦN QUYẾT: câu hỏi còn treo, nội dung cần phê duyệt tiếp.\n"
        "Kết thúc: Không thêm ghi chú ngoài phạm vi cuộc họp.\n\n"
        "TRANSCRIPT NGUỒN (giữ nguyên dòng):\n"
        f"{transcript_text}"
    )

def _extract_text(resp):
    finish = None
    out = ""
    if getattr(resp, "candidates", None):
        cand = resp.candidates[0]
        finish = getattr(cand, "finish_reason", None)
        parts = getattr(cand, "content", None).parts if getattr(cand, "content", None) else []
        out = "".join(getattr(p, "text", "") for p in parts).strip()
    else:
        out = (getattr(resp, "text", "") or "").strip()
    return out, finish

def _ends_with_sentence_punct(s: str) -> bool:
    return bool(re.search(r'[\.!\?…]$', s.strip()))

def summarize_with_gemini(transcript_text: str, log_cb=lambda s: None) -> str:
    import google.generativeai as genai
    genai.configure(api_key=GOOGLE_API_KEY)
    model = genai.GenerativeModel(
        model_name=GEMINI_MODEL,
        system_instruction=SYSTEM_INSTRUCTION,
    )
    GEN_CFG = {
        "temperature": GEMINI_TEMPERATURE,
        "top_p": GEMINI_TOP_P,
        "top_k": GEMINI_TOP_K,
        "max_output_tokens": GEMINI_MAX_OUTPUT_TOKENS,
    }
    log_cb("[SUM] Bat dau tao tom tat (Gemini 2.5 Flash)…")
    full = ""
    followup_prefix = ""
    for _ in range(3):
        resp = model.generate_content(
            followup_prefix + build_enterprise_prompt(transcript_text if not full else transcript_text),
            generation_config=GEN_CFG,
            request_options={"timeout": 180},
        )
        chunk, finish = _extract_text(resp)
        if chunk:
            full += (("\n" if full and not full.endswith("\n") else "") + chunk)
        if finish == 2:  # MAX_TOKENS
            tail = full[-500:]
            followup_prefix = (
                "Tiep tuc ngay sau doan truoc, khong lap lai. Hoan tat cau dang do. "
                f"Canh gan nhat:\n```{tail}```\n\n"
            )
            time.sleep(1.0)
            continue
        break
    if full and not _ends_with_sentence_punct(full):
        full += "."
    log_cb("[SUM] Hoan tat tom tat.")
    return full.strip()

# ----------------------- UTILS ------------------------
def ts_hhmmss(sec: float | None) -> str:
    if sec is None or (isinstance(sec, float) and math.isnan(sec)):
        sec = 0.0
    return str(timedelta(seconds=int(sec))).rjust(8, "0")

def ensure_ffmpeg():
    if not shutil.which("ffmpeg"):
        raise SystemExit("FFmpeg chưa có trong PATH.")

def to_wav_16k_mono(inp: Path, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["ffmpeg", "-y", "-i", str(inp), "-ac", "1", "-ar", "16000", str(out)],
        capture_output=True
    )
    if r.returncode != 0:
        raise RuntimeError(f"FFmpeg loi khi chuan hoa:\n{r.stderr.decode('utf-8', 'replace')}")

def get_duration_sec(path: Path) -> float:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", str(path)],
        capture_output=True, text=True
    )
    if r.returncode != 0:
        return 0.0
    try:
        return float(json.loads(r.stdout)["format"]["duration"])
    except Exception:
        return 0.0

def unit_norm(v: np.ndarray):
    n = np.linalg.norm(v)
    return v / n if n > 0 else v

def ensure_sample_norm(mp3_path: Path) -> Path:
    """
    Từ file mp3 trong SAMPLE_DIR, tạo/lấy wav 16k mono tương ứng trong SAMPLE_NORM_DIR.
    Trả về đường dẫn wav đã chuẩn hoá.
    """
    mp3_path = mp3_path.resolve()
    if mp3_path.suffix.lower() != ".mp3":
        raise RuntimeError(f"Chi ho tro .mp3 cho giong mau: {mp3_path.name}")
    out_wav = SAMPLE_NORM_DIR / (mp3_path.stem + ".wav")
    if not out_wav.exists():
        out_wav.parent.mkdir(parents=True, exist_ok=True)
        to_wav_16k_mono(mp3_path, out_wav)
    return out_wav

# ===================================================================
#                           PIPELINE
# ===================================================================
def run_pipeline(input_dir: Path, log_cb=lambda s: None, prog_cb=lambda p: None):
    """
    Full pipeline:
      - Tìm file audio trong input_dir + label.txt (tùy chọn)
      - Chuẩn hóa wav 16k mono
      - ASR (faster-whisper) với auto fallback compute (GPU -> CPU)
      - Diarization + speaker embedding (pyannote) + auto gán tên theo mẫu giọng (SAMPLE_DIR/SAMPLE_NORM_DIR)
      - Áp file thư ký cho các "Nguoi Noi i" (không đụng tên đã auto-match)
      - Ghi transcript TXT + mapping JSON
      - (Tùy chọn) Tóm tắt bằng Gemini 2.5 Flash -> _summary.txt
    Trả về: (output_txt_path, output_summary_path or None)
    """
    import torch
    from faster_whisper import WhisperModel
    from intervaltree import IntervalTree
    from pyannote.audio import Inference, Pipeline
    from pyannote.core import Segment

    def LOG(x: str):
        log_cb(str(x))

    # --------- I/O xác định đường dẫn ---------
    if not input_dir.is_dir():
        raise SystemExit(f"Khong thay thu muc input: {input_dir}")
    LOG(f"[IO] Input dir: {input_dir}")

    audio_exts = (".mp3", ".wav", ".m4a", ".flac", ".aac", ".wma", ".ogg", ".opus", ".mp4", ".mkv")
    audio_files = [p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in audio_exts]
    if not audio_files:
        raise SystemExit("Khong thay file audio trong input_dir")
    audio_files.sort(key=lambda p: p.stat().st_size, reverse=True)
    AUDIO_IN = audio_files[0]

    NAMES_PATH = input_dir / "label.txt"

    OUTPUT_DIR = input_dir.parent / "output"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    WAV_16K  = OUTPUT_DIR / f"{AUDIO_IN.stem}_16k_mono.wav"
    OUT_TXT  = OUTPUT_DIR / f"{AUDIO_IN.stem}_full_transcript.txt"
    OUT_JSON = OUTPUT_DIR / f"{AUDIO_IN.stem}_mapping.json"

    LOG(f"[IO] Audio: {AUDIO_IN}")
    LOG(f"[IO] Label: {NAMES_PATH if NAMES_PATH.exists() else '(khong co)'}")
    LOG(f"[IO] Output folder: {OUTPUT_DIR}")

    ensure_ffmpeg()
    to_wav_16k_mono(AUDIO_IN, WAV_16K)
    prog_cb(5)

    # --------- ASR: faster-whisper (auto-fallback compute) ---------
    try:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = None
    segments = None
    info = None

    def try_init(ct: str) -> bool:
        nonlocal model
        try:
            LOG(f"[ASR] Init: {ASR_MODEL} | compute_type={ct} | device={device}")
            model = WhisperModel(ASR_MODEL, device=device, compute_type=ct)
            LOG("[ASR] OK")
            return True
        except Exception as e:
            LOG(f"[ASR] Fail: {e}")
            return False

    if device == "cuda":
        for ct in TRY_COMPUTE:
            if not try_init(ct):
                continue
            try:
                t0 = time.time()
                seg_iter, info = model.transcribe(
                    str(WAV_16K),
                    task="transcribe", language="vi",
                    condition_on_previous_text=False,
                    vad_filter=True, vad_parameters={"min_silence_duration_ms": 300},
                    beam_size=BEAM_SIZE, temperature=[0.0, 0.2],
                    word_timestamps=False,
                    initial_prompt="Cuoc hop noi bo, ke hoach sprint, backlog, phan cong nhiem vu, moc thoi gian."
                )
                segments = list(seg_iter)
                LOG(f"[ASR] Done on GPU in {time.time()-t0:.1f}s, segments={len(segments)}")
                break
            except Exception as e:
                LOG(f"[ASR] Runtime fail: {e}")
                model = None
                continue

    if segments is None:
        LOG("[ASR] Fallback CPU (float32)")
        if not try_init("float32"):
            raise SystemExit("ASR init that bai (CPU)")
        t0 = time.time()
        seg_iter, info = model.transcribe(
            str(WAV_16K),
            task="transcribe", language="vi",
            condition_on_previous_text=False,
            vad_filter=True, vad_parameters={"min_silence_duration_ms": 300},
            beam_size=BEAM_SIZE, temperature=[0.0, 0.2],
            word_timestamps=False,
            initial_prompt="Cuoc hop noi bo, ke hoach sprint, backlog, phan cong nhiem vu, moc thoi gian."
        )
        segments = list(seg_iter)
        LOG(f"[ASR] Done on CPU in {time.time()-t0:.1f}s, segments={len(segments)}")

    LOG(f"[ASR] Language: {getattr(info, 'language', '?')} ({getattr(info, 'language_probability', 0.0):.1%})")
    prog_cb(40)

    # Giải phóng VRAM
    del model
    gc.collect()
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    # --------- Diarization + Embedding (pyannote) ---------
    from intervaltree import IntervalTree
    from pyannote.audio import Inference, Pipeline
    from pyannote.core import Segment

    if not HF_TOKEN or not HF_TOKEN.startswith("hf_"):
        raise SystemExit("HF_TOKEN khong hop le. Vui long set token da accept pyannote.")

    PYANNOTE_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.environ["PYANNOTE_AUDIO_DEVICE"] = "cuda" if PYANNOTE_DEVICE.type == "cuda" else "cpu"

    speaker_infer = Inference(
        EMB_MODEL_ID,
        use_auth_token=HF_TOKEN,
        device=PYANNOTE_DEVICE,
        window="whole"
    )

    # ---- NẠP GIỌNG MẪU TOÀN CỤC (SAMPLE_DIR + SAMPLE_NORM_DIR) ----
    known_speakers = {}
    sample_files = [p for p in SAMPLE_DIR.glob("*.mp3")]
    if not sample_files:
        LOG(f"[Canh bao] Khong thay mp3 mau trong {SAMPLE_DIR}. Se gan 'Nguoi Noi i' cho tat ca.")
    else:
        for mp3 in sample_files:
            name = mp3.stem.strip()
            try:
                wav_norm = ensure_sample_norm(mp3)   # cache dùng chung tại SAMPLE_NORM_DIR
                dur = get_duration_sec(wav_norm)
                if dur <= 0: 
                    continue
                vec = speaker_infer.crop(str(wav_norm), Segment(0.0, float(dur)))
                vec = unit_norm(np.asarray(vec).reshape(-1))
                known_speakers[name] = vec
            except Exception as e:
                LOG(f"[Warn] Loi chuan hoa/nap mau {mp3.name}: {e}")
    LOG(f"[Embed] Loaded {len(known_speakers)} voice samples from {SAMPLE_DIR}")

    dia = Pipeline.from_pretrained(DIAR_PIPELINE_ID, use_auth_token=HF_TOKEN)

    def run_diar_with_fallback():
        try:
            dia.to(PYANNOTE_DEVICE)
            t0 = time.time()
            kw = {}
            if NUM_SPEAKERS_HINT is not None:
                kw["num_speakers"] = int(NUM_SPEAKERS_HINT)
            diarization = dia(str(WAV_16K), **kw)
            LOG(f"[Diar] Done on {PYANNOTE_DEVICE.type.upper()} in {time.time()-t0:.1f}s")
            return diarization
        except Exception as e:
            warnings.warn(f"[Diar] GPU error, fallback CPU: {e}")
            dia.to(torch.device("cpu"))
            try:
                torch.set_num_threads(1)
            except Exception:
                pass
            t0 = time.time()
            kw = {}
            if NUM_SPEAKERS_HINT is not None:
                kw["num_speakers"] = int(NUM_SPEAKERS_HINT)
            diarization = dia(str(WAV_16K), **kw)
            LOG(f"[Diar] Done on CPU in {time.time()-t0:.1f}s")
            return diarization

    diar = run_diar_with_fallback()

    raw_turns = []  # (start, end, local_label)
    for turn, _, spk in diar.itertracks(yield_label=True):
        s, e = float(turn.start), float(turn.end)
        if e - s > 0.05:
            raw_turns.append((s, e, str(spk)))
    raw_turns.sort(key=lambda x: x[0])

    # ----- Label naming by embedding votes -----
    def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b))

    by_label = defaultdict(list)
    for s, e, lbl in raw_turns:
        if (e - s) >= MIN_TURN_SEC:
            by_label[lbl].append((s, e))
    for lbl in by_label:
        by_label[lbl].sort(key=lambda x: (x[1] - x[0]), reverse=True)
        by_label[lbl] = by_label[lbl][:5]

    label_votes = {}
    for lbl, segs in by_label.items():
        votes = defaultdict(int)
        for s, e in segs:
            vec = speaker_infer.crop(str(WAV_16K), Segment(float(s), float(e)))
            vec = unit_norm(np.asarray(vec).reshape(-1))
            best_name, best_sim = None, -1.0
            for name, ref in known_speakers.items():
                sim = cosine_sim(vec, ref)
                if sim > best_sim:
                    best_name, best_sim = name, sim
            if best_name is not None and best_sim >= SIM_THRESHOLD:
                votes[best_name] += 1
        label_votes[lbl] = dict(votes)

    label2name = {}
    unknown_counter = 0
    labels_in_order = []
    seen = set()
    for _, _, lbl in raw_turns:
        if lbl not in seen:
            labels_in_order.append(lbl)
            seen.add(lbl)

    for lbl in labels_in_order:
        votes = label_votes.get(lbl, {})
        if votes:
            best_name = max(votes, key=votes.get)
            total_votes = sum(votes.values())
            if votes[best_name] >= MAJORITY_MIN_VOTES and (votes[best_name] / max(1, total_votes)) >= MAJORITY_PROP:
                label2name[lbl] = best_name
                continue
        unknown_counter += 1
        label2name[lbl] = f"Nguoi Noi {unknown_counter}"

    LOG("[Map] Label -> Final name:")
    for lbl in labels_in_order:
        LOG(f"  {lbl} -> {label2name[lbl]} (votes={label_votes.get(lbl, {})})")

    named_turns = [(s, e, label2name[lbl]) for (s, e, lbl) in raw_turns]

    # ----- Assign speaker to ASR segments + merge -----
    tree = IntervalTree()
    if not named_turns:
        tree.addi(0.0, 10**9, "Nguoi Noi 1")
    else:
        for s, e, spk in named_turns:
            tree.addi(float(s), float(e), spk)

    def assign_speaker(st, ed):
        cand = tree.overlap(st, ed)
        if cand:
            best = max(cand, key=lambda iv: min(ed, iv.end) - max(st, iv.begin))
            return best.data
        window = 0.5
        neigh = [iv for iv in tree if (abs(iv.begin - ed) <= window or abs(iv.end - st) <= window)]
        if neigh:
            best = min(neigh, key=lambda iv: min(abs(iv.end - st), abs(iv.begin - ed)))
            return best.data
        return "Nguoi Noi 1"

    asr_segments = []
    for seg in segments:
        st = float(seg.start if seg.start is not None else 0.0)
        ed = float(seg.end if seg.end is not None else st)
        txt = (seg.text or "").strip()
        if not txt:
            continue
        asr_segments.append({"start": st, "end": ed, "text": txt})

    aligned = []
    for seg in asr_segments:
        st, ed, txt = seg["start"], seg["end"], seg["text"]
        spk = assign_speaker(st, ed)
        if aligned and aligned[-1]["speaker"] == spk and (st - aligned[-1]["end"]) < MERGE_GAP:
            aligned[-1]["end"] = ed
            aligned[-1]["text"] = (aligned[-1]["text"] + " " + txt).strip()
        else:
            aligned.append({"start": st, "end": ed, "speaker": spk, "text": txt})

    # ----- Áp file thư ký (chỉ thay "Nguoi Noi i") -----
    secretary_names = []
    if NAMES_PATH.exists():
        with open(NAMES_PATH, "r", encoding="utf-8") as f:
            secretary_names = [ln.strip() for ln in f if ln.strip()]

    def collect_auto_and_generic(rows):
        auto_names, auto_set = [], set()
        generics, seen_generic = [], set()
        for a in rows:
            spk = a["speaker"]
            if spk.startswith("Nguoi Noi "):
                if spk not in seen_generic:
                    generics.append(spk); seen_generic.add(spk)
            else:
                if spk not in auto_set:
                    auto_names.append(spk); auto_set.add(spk)
        return auto_names, generics

    auto_names, ordered_generic = collect_auto_and_generic(aligned)

    remaining_names = []
    for nm in secretary_names:
        if nm in auto_names:
            LOG(f"[Info] Bo qua ten da auto-match: {nm}")
        else:
            remaining_names.append(nm)

    manual_map = {}
    min_len = min(len(ordered_generic), len(remaining_names))
    for i in range(min_len):
        manual_map[ordered_generic[i]] = remaining_names[i]

    if len(remaining_names) > len(ordered_generic):
        extra = remaining_names[len(ordered_generic):]
        if extra:
            LOG(f"[Canh bao] Ten du (khong dung): {extra}")
    elif len(ordered_generic) > len(remaining_names):
        missing = ordered_generic[len(remaining_names):]
        if missing:
            LOG(f"[Canh bao] Con thieu ten cho: {missing}")

    def display_name(spk: str) -> str:
        return manual_map.get(spk, spk)

    # ----- Ghi TXT + JSON -----
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        for a in aligned:
            st = a["start"]; ed = a["end"]; spk = display_name(a["speaker"]); txt = a["text"]
            f.write(f"[{ts_hhmmss(st)} - {ts_hhmmss(ed)}] {spk}: {txt}\n")
    LOG(f"[Out] TXT: {OUT_TXT}")

    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "auto_names_in_transcript": auto_names,
            "generic_in_order": ordered_generic,
            "secretary_names_raw": secretary_names,
            "manual_map_applied": manual_map
        }, f, ensure_ascii=False, indent=2)
    LOG(f"[Out] JSON: {OUT_JSON}")

    prog_cb(85)

    # --------- (Tùy chọn) Tóm tắt bằng Gemini 2.5 Flash ---------
    OUT_SUMMARY = OUTPUT_DIR / f"{AUDIO_IN.stem}_summary.txt"
    output_summary_path = None
    if ENABLE_SUMMARY:
        try:
            if not GOOGLE_API_KEY:
                LOG("[SUM] Bo qua: thieu GOOGLE_API_KEY")
            else:
                transcript_text = Path(OUT_TXT).read_text(encoding="utf-8")
                summary_text = summarize_with_gemini(transcript_text, log_cb=LOG)
                OUT_SUMMARY.write_text(summary_text, encoding="utf-8")
                LOG(f"[Out] SUMMARY: {OUT_SUMMARY}")
                output_summary_path = str(OUT_SUMMARY)
        except Exception as e:
            LOG(f"[SUM] Loi tao tom tat: {e}")

    prog_cb(100)
    return str(OUT_TXT), output_summary_path

# ===================================================================
#                           WEB APP
# ===================================================================
app = FastAPI(title="ASR + Diarization + Summary (one-file)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

JOBS = {}  # job_id -> {"state","progress","log","output_txt","output_summary","error","thread"}
JOBS_LOCK = threading.Lock()

INDEX_HTML = r"""
<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8"/>
  <title>ASR + Diarization + (Gemini) Summary</title>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>

  <style>
    /* ========== Layout responsive, mượt mà ========== */
    :root{
      --container-max: 1280px;
      --container-pad: 20px;
      --gap: 16px;
    }
    html, body{height:100%}
    body{
      font-family: system-ui, -apple-system, Segoe UI, Roboto, Arial;
      margin:0; color:#222; background:#f6f8fb;
      display:flex; align-items:flex-start; justify-content:center;
    }
    .page{
      width: min(var(--container-max), 96vw);
      padding: clamp(12px, 2.2vw, var(--container-pad));
    }
    .card{
      border:1px solid #ddd; border-radius:14px;
      padding: clamp(12px, 2vw, 18px);
      background:#fff; box-shadow: 0 6px 24px rgba(16,24,40,.04);
    }

    /* Grid 2 cột → 1 cột khi hẹp; tránh tràn bằng minmax(0,1fr) + min-width:0 */
    .grid{
      display:grid;
      grid-template-columns: minmax(0,1fr) minmax(0,1fr);
      gap: var(--gap);
    }
    .grid > div{ min-width:0; }

    /* Hàng linh hoạt */
    .row{display:flex; gap:12px; align-items:center; flex-wrap:wrap;}
    .row > *{flex:1 1 220px; min-width:0;}

    h2{margin: 4px 0 12px 0; font-size: clamp(20px, 2.4vw, 26px);}
    .label{font-weight:600; margin-bottom:4px; font-size:14px;}
    .mt8{margin-top:8px;} .mt12{margin-top:12px;} .mt16{margin-top:16px;}

    /* Input luôn co trong khối, không đẩy tràn lưới */
    input[type="text"], input[type="file"]{
      box-sizing:border-box;
      width:100%; max-width:100%;
      padding:10px 12px;
      border:1px solid #ccd3e0; border-radius:8px;
      font-size:14px; background:#fff;
      overflow:hidden; text-overflow:ellipsis;
    }

    button{
      padding:10px 14px; border:0; border-radius:10px;
      background:#1976d2; color:white; cursor:pointer;
      font-size:14px;
    }
    button.ghost{background:#eef3fb; color:#1b4d91;}
    button:disabled{background:#9ab9da; cursor:not-allowed;}

    .logbox{
      background:#fafafa; border:1px solid #eee;
      border-radius:10px; padding:10px;
      height: min(34vh, 420px);
      overflow:auto; white-space:pre-wrap;
      font-family: ui-monospace,Consolas,monospace; font-size:13px;
    }
    .progress{height:12px; background:#edf0f6; border-radius:999px; overflow:hidden;}
    .progress > div{height:100%; background:#1976d2; width:0%; transition:width .2s ease;}

    .badge{display:inline-block; padding:4px 10px; border-radius:999px; font-size:12px; background:#eee;}
    .badge-idle{background:#999; color:#fff;}
    .badge-running{background:#1976d2; color:#fff;}
    .badge-done{background:#2e7d32; color:#fff;}
    .badge-error{background:#c62828; color:#fff;}

    .result-line{line-height:1.9; word-break:break-word;}
    .summary-panel{display:none; border:1px solid #e4e7ee; border-radius:10px; background:#fbfcff; padding:12px;}
    .toolbar{display:flex; gap:8px; align-items:center; justify-content:flex-end; flex-wrap:wrap;}
    .summary-meta{font-size:12px; color:#666;}
    .summary-content{
      white-space:pre-wrap; font-family: ui-monospace,Consolas,monospace; font-size:14px; line-height:1.55;
      max-height: min(50vh, 560px); overflow:auto; background:#fff; border:1px solid #e9edf5; border-radius:10px; padding:10px;
    }

    details > summary {cursor:pointer; user-select:none; font-weight:600;}
    a{color:#0b61d6; text-decoration:none}
    a:hover{text-decoration:underline}

    .list{
      border:1px solid #e6e6e6; border-radius:10px; padding:10px; max-height:240px;
      overflow:auto; background:#fafafa; font-size:14px;
    }
    .pill{display:inline-block; padding:6px 10px; background:#eef3fb; color:#1b4d91;
          border-radius:999px; margin:4px 6px 0 0; font-size:13px;}

    /* Mobile breakpoint */
    @media (max-width: 980px){
      .grid{ grid-template-columns: 1fr; }
    }
  </style>
</head>
<body>
  <div class="page">
    <div class="card">
      <h2>ASR + Diarization + (Gemini) Summary</h2>

      <div class="grid">
        <!-- Cột trái: chọn input -->
        <div>
          <div class="label">Chọn thư mục <u>input</u> (upload):</div>
          <input id="folder" type="file" webkitdirectory directory multiple />
          <div class="mt8">
            <button id="btnRunUpload" onclick="startUpload()">Chạy từ thư mục đã chọn</button>
          </div>

          <div class="mt16">
            <details>
              <summary>Hoặc dùng đường dẫn cục bộ</summary>
              <div class="mt8 row">
                <input id="inp" type="text" placeholder="VD: D:/Test/data_cuochop/Hop01/input"/>
                <button id="btnRun" onclick="startPath()">Chạy</button>
              </div>
            </details>
          </div>
        </div>

        <!-- Cột phải: thêm giọng mẫu -->
        <div>
          <div class="label">Thêm giọng mẫu (mp3) + Tên hiển thị</div>
          <div class="row">
            <input id="sampleName" type="text" placeholder="VD: Nguyen Van A"/>
            <input id="sampleFile" type="file" accept=".mp3"/>
          </div>
          <div class="mt8">
            <button class="ghost" onclick="uploadSample()">Thêm giọng mẫu</button>
            <button class="ghost" onclick="refreshSamples()">Làm mới danh sách</button>
          </div>
          <div class="mt8 list" id="sampleList"><i>— chưa tải danh sách —</i></div>
        </div>
      </div>

      <div class="mt16">
        <span class="badge badge-idle" id="stateBadge">IDLE</span>
      </div>

      <div class="label mt12">Tiến độ</div>
      <div class="progress"><div id="bar"></div></div>

      <div class="label mt12">Log</div>
      <div class="logbox" id="log"></div>

      <div class="label mt12">Kết quả</div>
      <div id="result" class="mt8 result-line"><i>— chưa có —</i></div>

      <details id="summaryWrap" class="mt12">
        <summary>Xem nhanh tóm tắt</summary>
        <div class="summary-panel mt8" id="summaryPanel">
          <div class="toolbar">
            <span class="summary-meta" id="summaryMeta">Độ dài: 0 ký tự</span>
            <button class="ghost" onclick="copySummary()" title="Sao chép tóm tắt">Sao chép</button>
            <button class="ghost" id="btnDownloadSummary" style="display:none">Tải file</button>
          </div>
          <div class="summary-content mt8" id="summaryContent"></div>
        </div>
      </details>
    </div>
  </div>

<script>
let JOB_ID = null;

function setBadge(state){
  const el = document.getElementById("stateBadge");
  el.textContent = state.toUpperCase();
  el.className = "badge " + (
    state === "running" ? "badge-running" :
    state === "done" ? "badge-done" :
    state === "error" ? "badge-error" : "badge-idle"
  );
}
function setProgress(p){ document.getElementById("bar").style.width = (p||0) + "%"; }
function appendLog(lines){
  const box = document.getElementById("log");
  if(!lines || !lines.length){ box.textContent=""; return; }
  box.textContent = lines.join("\n");
  box.scrollTop = box.scrollHeight;
}
function linkTo(path){ return '/file?path=' + encodeURIComponent(path||""); }
function setResult(txtPath, sumPath){
  const el = document.getElementById("result");
  let html = "";
  if (txtPath) html += '• <b>Transcript (TXT):</b> <a href="'+linkTo(txtPath)+'" target="_blank" rel="noreferrer noopener">'+txtPath+'</a><br/>';
  if (sumPath){
    html += '• <b>Tóm tắt:</b> <a href="'+linkTo(sumPath)+'" target="_blank" rel="noreferrer noopener">'+sumPath+'</a> ';
    html += '<button class="ghost" onclick="loadSummary(\''+String(sumPath).replace(/\\/g,'\\\\')+'\')">Xem nhanh</button>';
  }
  el.innerHTML = html || "<i>— chưa có —</i>";
}
function clearSummary(){
  const panel = document.getElementById("summaryPanel");
  const content = document.getElementById("summaryContent");
  const meta = document.getElementById("summaryMeta");
  const btn = document.getElementById("btnDownloadSummary");
  content.textContent = "";
  meta.textContent = "Độ dài: 0 ký tự";
  panel.style.display = "none";
  btn.style.display = "none";
  btn.onclick = null;
}

async function startPath(){
  const input = document.getElementById("inp").value.trim();
  if (!input) { alert("Nhập đường dẫn thư mục input"); return; }
  document.getElementById("btnRun").disabled = true;
  setBadge("queued"); setProgress(0); appendLog([]); setResult(null, null); clearSummary();
  try{
    const res = await fetch("/process?input_dir="+encodeURIComponent(input), {method:"POST"});
    if(!res.ok) throw new Error("Failed to start job");
    const data = await res.json(); JOB_ID = data.job_id; poll();
  }catch(err){
    alert("Lỗi: "+(err?.message||err)); document.getElementById("btnRun").disabled = false;
  }
}

async function startUpload(){
  const input = document.getElementById("folder");
  const files = Array.from(input.files || []);
  if (!files.length){ alert("Chọn thư mục input (chứa audio + label.txt)"); return; }
  document.getElementById("btnRunUpload").disabled = true;
  setBadge("queued"); setProgress(0); appendLog([]); setResult(null, null); clearSummary();
  try{
    const fd = new FormData();
    files.forEach(f => fd.append("files", f, f.webkitRelativePath || f.name));
    const res = await fetch("/process-upload", { method:"POST", body: fd });
    if(!res.ok) throw new Error("Failed to upload & start job");
    const data = await res.json(); JOB_ID = data.job_id; poll();
  }catch(err){
    alert("Lỗi: "+(err?.message||err));
    document.getElementById("btnRunUpload").disabled = false;
  }
}

async function poll(){
  if(!JOB_ID) return;
  try{
    const res = await fetch("/status/"+JOB_ID);
    if(!res.ok) throw new Error("status HTTP "+res.status);
    const data = await res.json();

    setBadge(data.state || "idle");
    setProgress(data.progress || 0);
    appendLog(data.log || []);
    if (data.output_txt || data.output_summary){
      setResult(data.output_txt, data.output_summary);
      const d = document.getElementById("summaryWrap");
      const isOpen = d && d.open;
      if (isOpen && data.output_summary && !document.getElementById("summaryContent").textContent.trim()){
        loadSummary(data.output_summary);
      }
    }

    if (data.state === "done" || data.state === "error"){
      document.getElementById("btnRun").disabled = false;
      document.getElementById("btnRunUpload").disabled = false;
      return;
    }
  }catch(err){
    appendLog(["[client] "+(err?.message||err)]);
    document.getElementById("btnRun").disabled = false;
    document.getElementById("btnRunUpload").disabled = false;
    return;
  }
  setTimeout(poll, 1500);
}

async function loadSummary(path){
  clearSummary();
  if(!path){ return; }
  const panel = document.getElementById("summaryPanel");
  const content = document.getElementById("summaryContent");
  const meta = document.getElementById("summaryMeta");
  const btn = document.getElementById("btnDownloadSummary");
  try{
    const url = linkTo(path);
    const res = await fetch(url);
    if(!res.ok) throw new Error("Không đọc được tóm tắt");
    const text = await res.text();
    content.textContent = text;
    meta.textContent = "Độ dài: " + (text?.length || 0) + " ký tự";
    panel.style.display = "block";
    btn.style.display = "inline-block";
    btn.onclick = () => { window.open(url, "_blank", "noreferrer"); };
    const wrap = document.getElementById("summaryWrap");
    if (wrap && !wrap.open) wrap.open = true;
  }catch(err){
    content.textContent = "Lỗi: " + (err?.message || err);
    panel.style.display = "block";
  }
}

async function uploadSample(){
  const name = (document.getElementById("sampleName").value || "").trim();
  const file = document.getElementById("sampleFile").files[0];
  if(!name){ alert("Nhập tên giọng mẫu"); return; }
  if(!file){ alert("Chọn file mp3"); return; }
  if(!/\.mp3$/i.test(file.name)){ alert("Vui lòng chọn file .mp3"); return; }
  const fd = new FormData();
  fd.append("name", name);
  fd.append("file", file, file.name);
  try{
    const res = await fetch("/upload-sample", {method:"POST", body: fd});
    if(!res.ok){
      const t = await res.text(); throw new Error(t || "Upload lỗi");
    }
    await refreshSamples();
    alert("Đã thêm giọng mẫu.");
    document.getElementById("sampleName").value = "";
    document.getElementById("sampleFile").value = "";
  }catch(err){
    alert("Lỗi: "+(err?.message||err));
  }
}

async function refreshSamples(){
  try{
    const res = await fetch("/samples");
    if(!res.ok) throw new Error("HTTP "+res.status);
    const data = await res.json();
    const box = document.getElementById("sampleList");
    const items = data.items || [];
    if(!items.length){ box.innerHTML = "<i>— chưa có mẫu —</i>"; return; }
    box.innerHTML = items.map(x => '<span class="pill">'+x.name+'</span>').join(" ");
  }catch(err){
    document.getElementById("sampleList").innerHTML = "Lỗi tải danh sách: "+(err?.message||err);
  }
}

function copySummary(){
  const content = document.getElementById("summaryContent").textContent || "";
  if(!content){ return; }
  navigator.clipboard.writeText(content).then(
    () => alert("Đã sao chép tóm tắt."),
    async () => {
      const ta = document.createElement("textarea");
      ta.value = content; document.body.appendChild(ta);
      ta.select(); document.execCommand("copy");
      document.body.removeChild(ta);
      alert("Đã sao chép tóm tắt.");
    }
  );
}
refreshSamples();
</script>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(INDEX_HTML)

@app.get("/health")
def health():
    return {"ok": True}

def _start_worker(job_id: str, input_dir: Path):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job: return
        job["state"] = "running"
        job["progress"] = 1

    def log_cb(line: str):
        with JOBS_LOCK:
            j = JOBS.get(job_id)
            if not j: return
            j["log"].append(line)
            if len(j["log"]) > 500:
                j["log"] = j["log"][-400:]

    def prog_cb(p: float):
        with JOBS_LOCK:
            j = JOBS.get(job_id)
            if not j: return
            j["progress"] = max(0, min(100, float(p)))

    try:
        out_txt, out_summary = run_pipeline(input_dir, log_cb=log_cb, prog_cb=prog_cb)
        with JOBS_LOCK:
            j = JOBS.get(job_id)
            if not j: return
            j["output_txt"] = out_txt
            j["output_summary"] = out_summary
            j["progress"] = 100
            j["state"] = "done"
    except SystemExit as e:
        with JOBS_LOCK:
            j = JOBS.get(job_id); 
            if j: j["state"] = "error"; j["error"] = str(e)
    except Exception as e:
        with JOBS_LOCK:
            j = JOBS.get(job_id); 
            if j: j["state"] = "error"; j["error"] = str(e)

@app.post("/process")
def process(input_dir: str = Query(..., description="Thư mục input (chứa audio + label.txt)")):
    jid = str(uuid.uuid4())
    with JOBS_LOCK:
        JOBS[jid] = {"state":"queued","progress":0,"log":[],
                     "output_txt":None,"output_summary":None,"error":None,"thread":None}
    t = threading.Thread(target=_start_worker, args=(jid, Path(input_dir)), daemon=True)
    t.start()
    with JOBS_LOCK:
        JOBS[jid]["thread"] = t
    return {"job_id": jid}

@app.get("/status/{job_id}")
def status(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "Job not found")
        return JSONResponse({
            "state": job["state"],
            "progress": job["progress"],
            "log": job["log"][-200:],
            "output_txt": job["output_txt"],
            "output_summary": job.get("output_summary"),
            "error": job["error"],
        })

@app.post("/cancel/{job_id}")
def cancel(job_id: str):
    with JOBS_LOCK:
        job = JOBS.get(job_id)
        if not job:
            raise HTTPException(404, "job_id khong ton tai")
        job["state"] = "error"
        job["error"] = "Da huy"
    return {"ok": True}

# Cho phép tải file (whitelist 2 gốc: cwd & uploads)
OUTPUT_ROOT = Path.cwd()
UPLOADS_ROOT = Path.cwd() / "uploads"
UPLOADS_ROOT.mkdir(parents=True, exist_ok=True)

@app.get("/file")
def get_file(path: str):
    p = Path(path).resolve()
    roots = [OUTPUT_ROOT.resolve(), UPLOADS_ROOT.resolve(), DATA_ROOT.resolve()]
    if not any(root == p or root in p.parents for root in roots):
        raise HTTPException(403, "Path not allowed")
    if not p.is_file():
        raise HTTPException(404, "File not found")
    mt = "text/plain"
    if p.suffix.lower() == ".json":
        mt = "application/json"
    return FileResponse(str(p), media_type=mt, filename=p.name)

def _slugify_name(s: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in " -_." else "_" for ch in s).strip()
    safe = safe.replace(" ", "_")
    return safe or "unknown"

@app.post("/upload-sample")
async def upload_sample(name: str = Form(...), file: UploadFile = File(...)):
    # chỉ nhận mp3 (pipeline đang quét *.mp3)
    ext = (Path(file.filename).suffix or "").lower()
    if ext != ".mp3":
        raise HTTPException(400, "Vui lòng upload file .mp3")
    stem = _slugify_name(name)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    dest = SAMPLE_DIR / f"{stem}.mp3"
    try:
        with dest.open("wb") as f:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        # chuẩn hoá ngay sang SAMPLE_NORM_DIR (cache dùng chung)
        try:
            ensure_sample_norm(dest)
        except Exception as e:
            print(f"[Warn] Chuan hoa mau loi: {e}", flush=True)
    except Exception as e:
        raise HTTPException(500, f"Lỗi lưu file: {e}")
    return {"ok": True, "path": str(dest)}

@app.get("/samples")
def list_samples():
    if not SAMPLE_DIR.exists():
        return {"items": []}
    items = [{"name": p.stem, "path": str(p)} for p in SAMPLE_DIR.glob("*.mp3")]
    return {"items": sorted(items, key=lambda x: x["name"].lower())}

@app.post("/process-upload")
async def process_upload(files: list[UploadFile] = File(...)):
    jid = str(uuid.uuid4())
    with JOBS_LOCK:
        JOBS[jid] = {"state":"queued","progress":0,"log":[],
                     "output_txt":None,"output_summary":None,"error":None,"thread":None}

    # Lưu thư mục input vào uploads/<jid>/input
    job_root = UPLOADS_ROOT / jid
    input_dir = job_root / "input"
    input_dir.mkdir(parents=True, exist_ok=True)

    try:
        for uf in files:
            relname = uf.filename
            relpath = Path(relname)
            relpath = Path(*[p for p in relpath.parts if p not in ("..", "/", "\\")])
            dst = input_dir / relpath.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            with dst.open("wb") as f:
                while True:
                    chunk = await uf.read(1024*1024)
                    if not chunk:
                        break
                    f.write(chunk)
    except Exception as e:
        with JOBS_LOCK:
            JOBS[jid]["state"]="error"; JOBS[jid]["error"]=f"Lỗi lưu upload: {e}"
        raise HTTPException(500, f"Lỗi lưu upload: {e}")

    t = threading.Thread(target=_start_worker, args=(jid, input_dir), daemon=True)
    t.start()
    with JOBS_LOCK:
        JOBS[jid]["thread"] = t
    return {"job_id": jid}

# ----------------------- RUN SERVER -----------------------
if __name__ == "__main__":
    import uvicorn
    url = "http://127.0.0.1:8000/"
    print(f"Open: {url}", flush=True)
    try:
        webbrowser.open(url)
    except Exception:
        pass
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", "8000")))
