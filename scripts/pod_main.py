#!/usr/bin/env python3
"""Pure-Python pod entrypoint. No git / apt needed.
Heartbeats + results go to a GitHub branch via the GitHub API when credentials are
available. Self-terminates after the selected run."""
import os, sys, time, json, subprocess, threading, traceback, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)

from trust_geometry import github_io as gh

RUN_STAGE = os.environ.get("TG_RUN_STAGE", "R0").upper()
RUN_BRANCH = os.environ.get("TG_RUN_BRANCH", f"{RUN_STAGE.lower()}-run")
DRIVERS = {
    "R0": ("run_r0.py", "r0_log.txt", ["r0_results.json", "r0_log.txt", "driver_stdout.txt", "r0_arrays.npz"]),
    "R1": ("run_r1.py", "r1_log.txt", ["r1_results.json", "r1_log.txt", "driver_stdout.txt", "r1_directions.npz"]),
}
HB_LOG = []


def gh_available():
    return bool(os.environ.get("GH_TOKEN") or os.environ.get("GITHUB_TOKEN"))


def gh_put_text(path, text, message):
    if not gh_available():
        return False
    return gh.put_text(path, text, message, branch=RUN_BRANCH)


def gh_put_file(path, content_bytes, message):
    if not gh_available():
        return False, "no-github-token"
    return gh.put_file(path, content_bytes, message, branch=RUN_BRANCH)


def hb(stage, extra=""):
    line = f"{time.strftime('%H:%M:%S')} STAGE={stage} {extra}".strip()
    HB_LOG.append(line); print(line, flush=True)
    try:
        gh_put_text("results/heartbeat.txt", "\n".join(HB_LOG) + "\n", f"hb: {stage}")
    except Exception as e:
        print("hb push err", repr(e)[:120], flush=True)

def self_terminate():
    pid = os.environ.get("RUNPOD_POD_ID"); key = os.environ.get("RUNPOD_API_KEY")
    if pid and key:
        try:
            urllib.request.urlopen(urllib.request.Request(
                f"https://rest.runpod.io/v1/pods/{pid}",
                headers={"Authorization": f"Bearer {key}", "User-Agent": "tg"}, method="DELETE"), timeout=25)
        except Exception:
            pass

def main():
    if RUN_STAGE not in DRIVERS:
        raise ValueError(f"unknown TG_RUN_STAGE={RUN_STAGE!r}; expected one of {sorted(DRIVERS)}")
    driver, progress_log, artifacts = DRIVERS[RUN_STAGE]
    if gh_available():
        gh.ensure_branch(RUN_BRANCH)
    hb("STARTED", f"{os.uname().nodename} stage={RUN_STAGE} branch={RUN_BRANCH}")
    hb("PIP_INSTALLING")
    # GPU check
    try:
        import torch
        hb("GPU_OK", f"torch={torch.__version__} cuda={torch.cuda.is_available()} {torch.cuda.get_device_name(0) if torch.cuda.is_available() else ''}")
    except Exception as e:
        hb("GPU_ERR", repr(e)[:100])
    # deps (torch already present in image)
    pip = [sys.executable, "-m", "pip", "install", "-q",
           "transformers>=4.56.0", "accelerate>=1.0.0", "huggingface_hub>=0.25",
           "scikit-learn", "scipy", "numpy", "safetensors", "hf_transfer", "kernels>=0.4.0"]
    r = subprocess.run(pip, capture_output=True, text=True)
    hb("DEPS_DONE", f"pip_rc={r.returncode} {r.stderr[-200:] if r.returncode else ''}")

    # Run the selected driver as a subprocess; stream heartbeats of its log while it runs.
    env = dict(os.environ, HF_HUB_ENABLE_HF_TRANSFER="1", PYTHONUNBUFFERED="1")
    logf = os.path.join(ROOT, "results", progress_log)
    proc = subprocess.Popen([sys.executable, os.path.join(HERE, driver)],
                            cwd=ROOT, env=env,
                            stdout=open(os.path.join(ROOT, "results", "driver_stdout.txt"), "w"),
                            stderr=subprocess.STDOUT)
    t0 = time.time()
    while proc.poll() is None:
        time.sleep(12)
        tail = ""
        if os.path.exists(logf):
            tail = open(logf).read()[-400:]
        try:
            gh_put_text("results/progress.txt",
                        f"stage={RUN_STAGE}\nelapsed={int(time.time()-t0)}s\n\n...{tail}", "progress")
        except Exception:
            pass
    hb("DRIVER_EXIT", f"rc={proc.returncode} elapsed={int(time.time()-t0)}s")

    # upload result artifacts
    for name in artifacts:
        p = os.path.join(ROOT, "results", name)
        if os.path.exists(p):
            try:
                ok, code = gh_put_file(f"results/{name}", open(p, "rb").read(), f"result: {name}")
                print("uploaded", name, ok, code, flush=True)
            except Exception as e:
                print("upload err", name, repr(e)[:120], flush=True)
    hb(f"{RUN_STAGE}_DONE")

if __name__ == "__main__":
    try:
        main()
    except Exception:
        tb = traceback.format_exc()
        print(tb, flush=True)
        try: hb("FATAL", tb[-300:])
        except Exception: pass
    finally:
        self_terminate()
