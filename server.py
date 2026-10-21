"""AutoTuner 엔진 서버. Mac 앱(AutoTuner.app)이 켜질 때 띄우고, 앱을 끄면 같이 꺼진다.

실행: python server.py  →  http://127.0.0.1:8766  (화면은 Mac 앱)
앱이 띄울 때는 빈 포트(TETOTUNE_PORT)와 실행마다 새로 만든 비밀값(TETOTUNE_TOKEN)을 넘기고,
비밀값을 X-TetoTune-Token 머리글로 보낸 요청만 받는다 (같은 Mac의 다른 사용자 · 다른 프로그램이 못 쓰게).
"""
import hmac
import json
import os
import shutil
import threading
import time
import traceback
import uuid
from dataclasses import asdict, fields
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse

from tetotune import __version__, pitch, stemmixer, voicebank
from tetotune.tune import Options, tune

# 설정 값마다 허용 범위 (숫자) 또는 고를 수 있는 값 (글자)
LIMITS = {
    "bpm": (0.0, 400.0), "pitch_follow": (0.0, 1.0), "vibrato": (0.0, 2.0), "correct": (0.0, 1.0),
    "dynamics": (0.0, 2.0), "breathiness": (0.0, 2.0), "tension": (0.0, 2.0), "breath_volume": (0.0, 3.0),
}
CHOICES = {"breaths": ("auto", "voicebank", "original", "off"), "pitch_method": ("auto", "harvest", "rmvpe")}
MAX_JOBS = 50
INCOMPLETE = ".incomplete"  # 조교 중인 폴더 표시 (엔진 pid). 끝나면 지우고, 엔진이 중간에 꺼졌으면 다음 실행 때 폴더를 치운다

PORT = int(os.environ.get("TETOTUNE_PORT", 8766))
TOKEN = os.environ.get("TETOTUNE_TOKEN") or None
OUTPUT = Path(os.environ.get("TETOTUNE_OUTPUT") or Path.home() / "Music" / "AutoTuner")
OLD_OUTPUT = Path.home() / "Music" / "TetoTune"  # 앱 이름을 바꾸기 전 결과 폴더: 처음 켤 때 옮긴다
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}

app = FastAPI()
jobs = {}  # job_id -> {"status", "step", "progress", "result", "error"}
work_lock = threading.Lock()  # 한 번에 한 곡씩


@app.middleware("http")
async def only_this_computer(request, call_next):
    """이 컴퓨터에서 보낸 요청만 받는다 (다른 웹사이트가 몰래 부르지 못하게). Mac 앱의 요청에는 Origin이 없다."""
    host = request.headers.get("host", "")
    name = host if host.endswith("]") else host.rsplit(":", 1)[0]
    origin = request.headers.get("origin")
    if name not in LOCAL_HOSTS or (origin is not None and origin != f"http://{host}"):
        return JSONResponse({"detail": "이 컴퓨터의 AutoTuner에서 보낸 요청만 받아요"}, status_code=403)
    # 머리글은 latin-1로 풀려 오니 바이트로 비교한다 (글자로 비교하면 ASCII가 아닌 값에서 오류가 난다)
    sent = request.headers.get("x-tetotune-token", "").encode("latin-1")
    if TOKEN and not hmac.compare_digest(sent, TOKEN.encode()):
        return JSONResponse({"detail": "이 엔진을 켠 AutoTuner에서 보낸 요청만 받아요"}, status_code=403)
    return await call_next(request)


@app.get("/api/status")
def status(singer: str = ""):
    """singer: 앱에서 고른 가수 폴더 (비우면 테토를 찾는다)."""
    singer_id, info = voicebank.find(singer if singer and (Path(singer) / "character.txt").exists() else None)
    return {
        "version": __version__,
        "singer": info and {k: info[k] for k in ("name", "path", "kind", "breath_in", "breath_out", "phonemizer")},
        "singers": voicebank.list_singers(),
        "rmvpe": pitch.rmvpe_model_path() is not None,
        "stemmixer": stemmixer.library() is not None,
        "output": str(OUTPUT),
    }


@app.get("/api/stemmixer/songs")
def stemmixer_songs():
    return stemmixer.songs()


@app.get("/api/options")
def default_options():
    return asdict(Options())


def _folder_name(title):
    """결과 폴더 이름 "<제목> 날짜 시각". 경로에 못 쓰는 글자 · 앞의 점(숨김 폴더)은 빼고, 이미 있으면 번호를 붙인다."""
    safe = "".join(c for c in title if c not in '/\\:*?"<>|' and c >= " ").strip().lstrip(".").strip() or "노래"
    base = f"{safe[:60]} {time.strftime('%Y-%m-%d %H%M%S')}"
    name, k = base, 2
    while (OUTPUT / name).exists():
        name, k = f"{base} ({k})", k + 1
    return name


def _options(raw):
    """앱에서 온 설정 → Options. 모르는 키는 버리고, 숫자는 범위 안으로, 잘못된 값은 알기 쉬운 오류로."""
    if not isinstance(raw, dict):
        raise ValueError
    out = {}
    for f in fields(Options):
        if f.name not in raw:
            continue
        v = raw[f.name]
        if f.name in LIMITS:
            lo, hi = LIMITS[f.name]
            v = min(hi, max(lo, float(v)))
        elif f.name == "transpose":
            v = "auto" if str(v) == "auto" else str(int(min(24, max(-24, round(float(v))))))
        elif f.name in CHOICES:
            if v not in CHOICES[f.name]:
                raise ValueError
        else:
            v = str(v)
        out[f.name] = v
    if out.get("singer") and not (Path(out["singer"]) / "character.txt").exists():
        raise HTTPException(400, "고른 가수 폴더를 찾지 못했어요")
    return Options(**out)


def _input_file(path, what):
    if path and not Path(path).is_file():
        raise HTTPException(400, f"{what}을 찾지 못했어요")
    return path or None


@app.post("/api/tune")
def start(vocal: str = Form(None), song: str = Form(None), lyrics: str = Form(""), notes: str = Form(None),
          backing: str = Form(None), options: str = Form("{}")):
    if not vocal and not song:
        raise HTTPException(400, "보컬 파일이나 StemMixer 곡을 골라 주세요")
    vocal = _input_file(vocal, "보컬 파일")
    notes = _input_file(notes, "음표 파일")
    backing = _input_file(backing, "반주 파일")
    if song and not stemmixer.has_song(song):
        raise HTTPException(400, "StemMixer 곡을 찾지 못했어요")
    try:
        opts = _options(json.loads(options or "{}"))
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "설정을 읽지 못했어요")
    for old in [k for k, j in jobs.items() if j["status"] != "running"][:-MAX_JOBS]:
        jobs.pop(old, None)
    job_id = uuid.uuid4().hex[:12]
    jobs[job_id] = {"status": "running", "step": "대기 중", "progress": 0.0}
    threading.Thread(target=_run, args=(job_id, vocal, song, lyrics, notes, backing, opts), daemon=True).start()
    return {"job_id": job_id}


def _run(job_id, vocal, song, lyrics, notes, backing, opts):
    job = jobs[job_id]
    out = None
    try:
        with work_lock:
            title = Path(vocal).stem if vocal else song
            if song:
                job.update(step="StemMixer 곡 불러오는 중", progress=0.01)
                tmp = OUTPUT / ".work" / job_id
                _mark(tmp)
                vocal, sm_notes, sm_backing, title = stemmixer.prepare(song, tmp)
                notes = notes or (str(sm_notes) if sm_notes else None)
                backing = backing or (str(sm_backing) if sm_backing else None)
            out = OUTPUT / _folder_name(title)
            _mark(out)
            res = tune(vocal, out, lyrics or "", notes or None, backing or None, opts,
                       progress=lambda p, s: job.update(step=s, progress=round(p, 3)), title=title)
            (out / INCOMPLETE).unlink(missing_ok=True)
            job.update(status="done", progress=1.0, step="끝", result=asdict(res))
    except Exception as e:
        traceback.print_exc()
        if out is not None and out.exists() and not (out / "project.ustx").exists():
            shutil.rmtree(out, ignore_errors=True)
        elif out is not None:
            (out / INCOMPLETE).unlink(missing_ok=True)
        job.update(status="error", error=str(e) if isinstance(e, ValueError) else f"조교하지 못했어요: {e}")
    finally:
        shutil.rmtree(OUTPUT / ".work" / job_id, ignore_errors=True)


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404)
    return jobs[job_id]


def _result_folder(name):
    folder = (OUTPUT / name).resolve()
    if folder.parent != OUTPUT.resolve() or not (folder / "report.json").exists():
        raise HTTPException(404, "결과를 찾지 못했어요")
    return folder


@app.get("/api/results")
def results():
    out = []
    for p in OUTPUT.glob("*/report.json") if OUTPUT.exists() else []:
        try:
            r = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
        out.append({"id": p.parent.name, "folder": str(p.parent), "ustx": str(p.parent / "project.ustx"),
                    "notes": r.get("notes"), "breaths": r.get("breaths"), "warnings": r.get("warnings") or [],
                    "singer": r.get("singer"), "transpose": r.get("transpose"), "created": p.stat().st_mtime})
    out.sort(key=lambda r: -r["created"])
    return out


@app.get("/api/results/{name}/preview")
def preview(name: str):
    return json.loads((_result_folder(name) / "preview.json").read_text())


# 결과 지우기는 앱이 직접 휴지통으로 옮긴다 (OpenUtau에서 고친 project.ustx도 되살릴 수 있게)


def _mark(folder):
    folder.mkdir(parents=True, exist_ok=True)
    (folder / INCOMPLETE).write_text(str(os.getpid()))


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:  # 다른 사용자의 프로그램이 그 번호를 쓰는 중
        return True
    return True


def sweep_unfinished():
    """앱을 끄거나 강제로 꺼서 끝나지 못한 조교의 폴더(결과 · StemMixer 작업)를 치운다.
    표시(INCOMPLETE)가 있고 그 엔진이 이미 꺼진 폴더만: 사용자가 만든 폴더 · 다른 엔진이 쓰는 중인 폴더는 그대로."""
    for m in list(OUTPUT.glob(f"*/{INCOMPLETE}")) + list((OUTPUT / ".work").glob(f"*/{INCOMPLETE}")):
        try:
            pid = int(m.read_text().strip() or 0)
        except (OSError, ValueError):
            pid = 0
        if pid != os.getpid() and not (pid and _alive(pid)):
            shutil.rmtree(m.parent, ignore_errors=True)


def _exit_with_parent(pid):
    """앱이 강제로 꺼지거나 멈춰도 엔진이 혼자 남지 않게: 부모가 바뀌면(launchd가 받으면) 끝낸다."""
    while True:
        time.sleep(1)
        if os.getppid() != pid:
            os._exit(0)


if __name__ == "__main__":
    import uvicorn

    if not os.environ.get("TETOTUNE_OUTPUT") and OLD_OUTPUT.is_dir() and not OUTPUT.exists():
        try:
            OLD_OUTPUT.rename(OUTPUT)
        except OSError:
            pass
    OUTPUT.mkdir(parents=True, exist_ok=True)
    sweep_unfinished()
    parent = os.environ.get("TETOTUNE_PARENT_PID")
    if parent and parent.isdigit():
        threading.Thread(target=_exit_with_parent, args=(int(parent),), daemon=True).start()
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
