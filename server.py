"""TetoTune 엔진 서버. Mac 앱(TetoTune.app)이 켜질 때 띄우고, 앱을 끄면 같이 꺼진다.

실행: python server.py  →  http://127.0.0.1:8766  (화면은 Mac 앱)
"""
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

PORT = int(os.environ.get("TETOTUNE_PORT", 8766))
OUTPUT = Path(os.environ.get("TETOTUNE_OUTPUT") or Path.home() / "Music" / "TetoTune")
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
        return JSONResponse({"detail": "이 컴퓨터의 TetoTune에서 보낸 요청만 받아요"}, status_code=403)
    return await call_next(request)


@app.get("/api/status")
def status():
    singer_id, info = voicebank.find()
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
    safe = "".join(c for c in title if c not in '/\\:*?"<>|').strip() or "노래"
    return f"{safe[:60]} {time.strftime('%Y-%m-%d %H%M%S')}"


@app.post("/api/tune")
def start(vocal: str = Form(None), song: str = Form(None), lyrics: str = Form(""), notes: str = Form(None),
          backing: str = Form(None), options: str = Form("{}")):
    if not vocal and not song:
        raise HTTPException(400, "보컬 파일이나 StemMixer 곡을 골라 주세요")
    if vocal and not Path(vocal).exists():
        raise HTTPException(400, "보컬 파일을 찾지 못했어요")
    try:
        raw = json.loads(options or "{}")
        known = {f.name for f in fields(Options)}
        opts = Options(**{k: v for k, v in raw.items() if k in known})
    except Exception:
        raise HTTPException(400, "설정을 읽지 못했어요")
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
                vocal, sm_notes, sm_backing, title = stemmixer.prepare(song, tmp)
                notes = notes or (str(sm_notes) if sm_notes else None)
                backing = backing or (str(sm_backing) if sm_backing else None)
            out = OUTPUT / _folder_name(title)
            res = tune(vocal, out, lyrics or "", notes or None, backing or None, opts,
                       progress=lambda p, s: job.update(step=s, progress=round(p, 3)), title=title)
            job.update(status="done", progress=1.0, step="끝", result=asdict(res))
    except Exception as e:
        traceback.print_exc()
        if out is not None and out.exists() and not (out / "project.ustx").exists():
            shutil.rmtree(out, ignore_errors=True)
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


@app.delete("/api/results/{name}")
def delete_result(name: str):
    shutil.rmtree(_result_folder(name))
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    OUTPUT.mkdir(parents=True, exist_ok=True)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
