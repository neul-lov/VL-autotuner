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

from tetotune import __version__, align_fa, clean, edit, f0models, game, openutau, pitch, refine, retune, stemmixer, voicebank
from tetotune import tune as tune_mod
from tetotune.tune import Options, tune

# 설정 값마다 허용 범위 (숫자) 또는 고를 수 있는 값 (글자)
LIMITS = {
    "bpm": (0.0, 400.0), "pitch_follow": (0.0, 1.0), "vibrato": (0.0, 2.0), "correct": (0.0, 1.0),
    "dynamics": (0.0, 2.0), "breathiness": (0.0, 2.0), "tension": (0.0, 2.0), "breath_volume": (0.0, 3.0),
}
CHOICES = {"breaths": ("auto", "voicebank", "original", "off"), "pitch_method": ("auto", "harvest", "rmvpe", "fusion"),
           "refine": ("auto", "off"), "renderer": ("auto", "worldline")}
MAX_JOBS = 50  # 끝난 작업 기록은 이만큼만 (그리고 JOB_TTL초가 지나면 지운다)
JOB_TTL = 3600
MAX_ACTIVE = 4  # 돌고 있거나 차례를 기다리는 작업 (조교 · 구간 다시 조교 · 렌더) 수의 상한: 넘으면 새 요청을 거절한다
MAX_INPUT_BYTES = {"보컬 파일": 2 << 30, "반주 파일": 2 << 30, "음표 파일": 20 << 20}
INCOMPLETE = ".incomplete"  # 조교 중인 폴더 표시 (엔진 pid). 끝나면 지우고, 엔진이 중간에 꺼졌으면 다음 실행 때 폴더를 치운다

PORT = int(os.environ.get("TETOTUNE_PORT", 8766))
TOKEN = os.environ.get("TETOTUNE_TOKEN") or None
OUTPUT = Path(os.environ.get("TETOTUNE_OUTPUT") or Path.home() / "Music" / "AutoTuner")
OLD_OUTPUT = Path.home() / "Music" / "TetoTune"  # 앱 이름을 바꾸기 전 결과 폴더: 처음 켤 때 옮긴다
LOCAL_HOSTS = {"127.0.0.1", "localhost", "[::1]"}

app = FastAPI()
jobs = {}  # job_id -> {"status", "step", "progress", "result", "error"} (+ 끝난 시각 "_ended", 앱에는 보내지 않는다)
jobs_lock = threading.Lock()  # jobs를 고치는 곳 (요청마다 다른 스레드)
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
        "openutau": openutau.available(),  # 테토로 불러 보고 다듬기 · 미리 듣기를 할 수 있는지
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
    if path and Path(path).stat().st_size > MAX_INPUT_BYTES.get(what, 2 << 30):
        raise HTTPException(400, f"{what}이 너무 커요")
    return path or None


def _new_job(step="대기 중", progress=0.0):
    """새 작업 기록 → job_id. 오래된 끝난 기록을 치우고, 쌓인 작업이 MAX_ACTIVE개면 거절한다 (429)."""
    now = time.time()
    with jobs_lock:
        done = [k for k, j in jobs.items() if j["status"] != "running"]
        for k in done[:-MAX_JOBS] + [k for k in done[-MAX_JOBS:] if now - jobs[k].get("_ended", now) > JOB_TTL]:
            jobs.pop(k, None)
        if sum(j["status"] == "running" for j in jobs.values()) >= MAX_ACTIVE:
            raise HTTPException(429, "작업이 너무 많이 쌓였어요. 지금 하는 작업이 끝나면 다시 해 주세요")
        job_id = uuid.uuid4().hex[:12]
        jobs[job_id] = {"status": "running", "step": step, "progress": progress}
    return job_id


def _finish(job, **kw):
    job.update(_ended=time.time(), **kw)


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
    job_id = _new_job()
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
            if not notes:  # 음표 파일이 없으면 GAME으로 음표를 나눈다: 모델이 없으면 처음 한 번 받는다
                try:
                    game.ensure_model(lambda s: job.update(step=s, progress=0.01))
                except Exception as e:
                    print(f"GAME 모델을 받지 못했어요: {e}", flush=True)
                if (lyrics or "").strip():  # 가사 맞추기 모델 (HubertFA): 가사가 있을 때만, 처음 한 번 받는다
                    try:  # 한글 가사면 한국어 모델도
                        align_fa.ensure_model(lambda s: job.update(step=s, progress=0.01),
                                              korean=any("가" <= c <= "힣" for c in lyrics))
                    except Exception as e:
                        print(f"가사 맞추기 모델을 받지 못했어요: {e}", flush=True)
            for name in ("vocals", "dereverb"):  # 분석용 보컬 깨끗하게 하는 모델 (보컬 다시 분리는 반주가 있을 때만)
                if clean.wanted(name, backing):
                    try:
                        clean.ensure(name, lambda s: job.update(step=s, progress=0.01))
                    except Exception as e:
                        print(f"{name} 모델을 받지 못했어요: {e}", flush=True)
            if opts.pitch_method in ("auto", "fusion"):  # 음 높이 읽기 모델 (RMVPE): 없으면 처음 한 번 받는다
                try:
                    f0models.ensure_rmvpe(lambda s: job.update(step=s, progress=0.01))
                except Exception as e:
                    print(f"RMVPE 모델을 받지 못했어요: {e}", flush=True)
            hifi_note = tune_mod.prepare_renderer(lambda s: job.update(step=s, progress=0.01), opts.renderer)
            polish = opts.refine != "off" and openutau.available()
            scale = 0.6 if polish else 1.0  # 다듬기가 남은 시간의 대부분
            res = tune(vocal, out, lyrics or "", notes or None, backing or None, opts,
                       progress=lambda p, s: job.update(step=s, progress=round(p * scale, 3)), title=title)
            if hifi_note:
                res.warnings.append(hifi_note)
            if polish:
                w = refine.for_result(out, lambda p, s: job.update(step=s, progress=round(scale + (1 - scale) * p, 3)))
                if w:
                    res.warnings.append(w)
            (out / INCOMPLETE).unlink(missing_ok=True)
            _finish(job, status="done", progress=1.0, step="끝", result=asdict(res))
    except Exception as e:
        traceback.print_exc()
        # 결과가 다 만들어지기 전(report.json이 없으면 결과 목록에 안 보인다)에 멈췄으면 새 폴더를 치운다.
        # 다 만든 뒤(다듬기 중 오류 등)면 결과는 남긴다
        if out is not None and out.exists() and not _complete_result(out):
            shutil.rmtree(out, ignore_errors=True)
        elif out is not None:
            (out / INCOMPLETE).unlink(missing_ok=True)
        _finish(job, status="error", error=str(e) if isinstance(e, ValueError) else f"조교하지 못했어요: {e}")
    finally:
        shutil.rmtree(OUTPUT / ".work" / job_id, ignore_errors=True)


def _complete_result(folder):
    return all((folder / f).is_file() for f in ("project.ustx", "preview.json", "report.json"))


@app.get("/api/jobs/{job_id}")
def job_status(job_id: str):
    job = jobs.get(job_id)
    if job is None:
        raise HTTPException(404)
    return {k: v for k, v in job.items() if not k.startswith("_")}


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
        folder = p.parent
        teto = folder / (r.get("teto_preview") or "teto_preview.wav")
        backing = next((f for f in sorted(folder.glob("backing.*"))), None)
        out.append({"id": folder.name, "folder": str(folder), "ustx": str(folder / "project.ustx"),
                    "notes": r.get("notes"), "breaths": r.get("breaths"), "warnings": r.get("warnings") or [],
                    "singer": r.get("singer"), "transpose": r.get("transpose"), "created": p.stat().st_mtime,
                    "guide": str(folder / "guide_vocals.wav") if (folder / "guide_vocals.wav").exists() else None,
                    "teto": str(teto) if teto.is_file() else None, "backing": str(backing) if backing else None})
    out.sort(key=lambda r: -r["created"])
    return out


@app.get("/api/results/{name}/preview")
def preview(name: str):
    folder = _result_folder(name)
    _sync_external_edit(folder)
    prev = json.loads((folder / "preview.json").read_text())
    prev["history"] = edit.history_count(folder)
    try:  # 구간 다시 조교하기 창이 처음 조교한 설정에서 시작하게
        prev["options"] = json.loads((folder / "report.json").read_text()).get("options")
    except (OSError, ValueError):
        pass
    return prev


# AutoTuner가 project.ustx를 쓸 때는 언제나 preview.json을 그 뒤에 쓴다. 그래서 project.ustx가 preview.json보다
# 확실히 늦게 바뀌었으면 다른 앱(EasyUtau · OpenUtau)에서 고치고 저장한 것이다.
EXTERNAL_EDIT_SLACK = 1.0


def _sync_external_edit(folder):
    """다른 앱에서 고친 project.ustx로 미리 보기 음표를 맞추고, 테토 미리 듣기는 "다시 불러 보기"로 표시한다."""
    ustx, prev = folder / "project.ustx", folder / "preview.json"
    try:
        if ustx.stat().st_mtime <= prev.stat().st_mtime + EXTERNAL_EDIT_SLACK:
            return
    except OSError:
        return
    if not work_lock.acquire(blocking=False):
        return  # 조교 중이면 다음에 (미리 보기는 기다리지 않는다)
    try:
        edit.refresh_preview(folder, stale=True)
    except Exception:  # 읽을 수 없는 파일이면 예전 미리 보기를 그대로 보여 준다
        traceback.print_exc()
    finally:
        work_lock.release()


@app.post("/api/results/{name}/edit")
def edit_result(name: str, op: str = Form(...), index: int = Form(...), text: str = Form(None), time: float = Form(None)):
    """음절 옮기기 · 늘여 부르기 · 나누기 · 합치기 · 가사 고치기 → 새 미리 보기."""
    folder = _result_folder(name)
    kw = {}
    if op == "lyric":
        kw["text"] = text or ""
    if op == "split" and time is not None:
        kw["time"] = time
    with work_lock:
        try:
            return edit.apply(folder, op, index, **kw)
        except edit.EditError as e:
            raise HTTPException(400, str(e))


@app.post("/api/results/{name}/undo")
def undo_result(name: str):
    folder = _result_folder(name)
    with work_lock:
        try:
            return edit.undo(folder)
        except edit.EditError as e:
            raise HTTPException(400, str(e))


@app.post("/api/results/{name}/retune")
def retune_result(name: str, start: float = Form(...), end: float = Form(...), options: str = Form("{}"),
                  lyrics: str = Form(None), renote: bool = Form(False)):
    """고른 구간만 다시 조교 (뒤에서 돌리고 job으로 따라간다). 결과는 새 미리 보기."""
    folder = _result_folder(name)
    try:
        opts = asdict(_options(json.loads(options or "{}")))
        raw = json.loads(options or "{}")
        opts = {k: v for k, v in opts.items() if k in raw}  # 앱이 보낸 것만 바꾼다 (나머지는 처음 조교한 설정)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(400, "설정을 읽지 못했어요")
    if not end > start:
        raise HTTPException(400, "구간을 다시 골라 주세요")
    job_id = _new_job()

    def run():
        job = jobs[job_id]
        try:
            with work_lock:
                prev = retune.retune(folder, start, end, opts, lyrics, renote,
                                     progress=lambda p, s: job.update(step=s, progress=round(p, 3)))
            prev["history"] = edit.history_count(folder)
            _finish(job, status="done", progress=1.0, step="끝", result=prev)
        except Exception as e:
            traceback.print_exc()
            _finish(job, status="error", error=str(e) if isinstance(e, ValueError) else f"다시 조교하지 못했어요: {e}")
    threading.Thread(target=run, daemon=True).start()
    return {"job_id": job_id}


@app.post("/api/results/{name}/render")
def render_result(name: str):
    """지금 project.ustx를 테토로 다시 불러서 미리 듣기(teto_preview.wav)를 새로 만든다 (고친 뒤에). job으로 따라간다."""
    folder = _result_folder(name)
    if not openutau.available():
        raise HTTPException(400, "OpenUtau를 찾지 못했어요")
    job_id = _new_job("테토로 불러 보는 중", 0.1)

    def run():
        job = jobs[job_id]
        work = folder / ".render"
        try:
            with work_lock:
                wav = openutau.render(folder / "project.ustx", work)
                prev_path = folder / "preview.json"
                prev = json.loads(prev_path.read_text(encoding="utf-8"))
                prev.pop("teto_stale", None)
                os.replace(wav, folder / "teto_preview.wav")  # 같은 폴더 안이라 한 번에 바뀐다
                edit.write_text_atomic(prev_path, json.dumps(prev, ensure_ascii=False))
                prev["history"] = edit.history_count(folder)
            _finish(job, status="done", progress=1.0, step="끝", result=prev)
        except Exception as e:
            traceback.print_exc()
            _finish(job, status="error", error=str(e))
        finally:
            shutil.rmtree(work, ignore_errors=True)
    threading.Thread(target=run, daemon=True).start()
    return {"job_id": job_id}


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
    if not TOKEN:  # 앱 밖에서 직접 켰을 때도 같은 Mac의 다른 프로그램이 부르지 못하게: 비밀값을 만들어 알려 준다
        import secrets

        TOKEN = secrets.token_urlsafe(32)
        print(f"TETOTUNE_TOKEN이 없어서 새로 만들었어요. 요청마다 머리글 X-TetoTune-Token: {TOKEN}", flush=True)
    uvicorn.run(app, host="127.0.0.1", port=PORT, log_level="warning")
