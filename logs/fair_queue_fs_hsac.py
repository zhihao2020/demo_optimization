
import json, subprocess, time
from pathlib import Path

ROOT = r"D:\xuzh\demo_optimization"
MAX_LIVE = 5
QUEUE = json.loads(Path(ROOT, "logs", "fair_queue_fs_hsac.json").read_text(encoding="utf-8"))
state_path = Path(ROOT, "logs", "fair_queue_fs_hsac_state.json")
logp = Path(ROOT, "logs", "fair_queue_fs_hsac.log")

def load_started():
    if state_path.is_file():
        try:
            return set(json.loads(state_path.read_text(encoding="utf-8")).get("started", []))
        except Exception:
            return set()
    return set()

def save_started(started):
    state_path.write_text(json.dumps({"started": sorted(started)}, indent=2), encoding="utf-8")

def live_jobs():
    out = subprocess.check_output(
        "wmic process where name='python.exe' get CommandLine /FORMAT:LIST",
        shell=True, text=True, errors="replace",
    )
    dirs = set()
    for ln in out.splitlines():
        if "train_seasonal.py" not in ln.lower():
            continue
        if "seasonal_v1" not in ln.lower():
            continue
        if "--run-dir" not in ln.lower():
            continue
        part = ln.split("--run-dir", 1)[1].strip().strip('"').strip()
        part = part.split(" --")[0].strip().strip('"')
        dirs.add(part.lower())
    return dirs

def start_bat(bat):
    cmdline = 'cmd.exe /c call "%s"' % bat
    cmd = 'wmic process call create "%s"' % cmdline.replace('"', '\\"')
    subprocess.check_call(cmd, shell=True)

def log(msg):
    print(msg, flush=True)
    with logp.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")

started = load_started()
for j in QUEUE:
    rd = str(j.get("run_dir", "")).lower()
    if rd and rd in live_jobs():
        started.add(j["name"])
save_started(started)
log("FS_HSAC_QUEUE start n=%d max=%d started=%d" % (len(QUEUE), MAX_LIVE, len(started)))

while True:
    pending = [j for j in QUEUE if j["name"] not in started]
    live = live_jobs()
    n_live = len(live)
    log("live=%d pending=%d started=%d" % (n_live, len(pending), len(started)))
    if not pending and n_live == 0:
        log("QUEUE_DONE")
        break
    # If only non-queue seasonal jobs remain and our queue is done, exit.
    if not pending:
        ours = {str(j.get("run_dir", "")).lower() for j in QUEUE}
        if not (live & ours):
            log("QUEUE_DONE_OUR_JOBS")
            break
        time.sleep(45)
        continue
    if n_live < MAX_LIVE:
        job = pending[0]
        try:
            start_bat(job["bat"])
            started.add(job["name"])
            save_started(started)
            log("STARTED " + job["name"])
        except Exception as exc:
            log("FAIL " + job["name"] + " " + str(exc))
        time.sleep(12)
        continue
    time.sleep(45)
