"""Helpers for the blog's follow-up experiments: start / stop 8-device engines on a chosen
device set and drive them with the bench's own `vllm bench serve` cell (sonnet, greedy, 128 out)."""
import ctypes, json, os, signal, subprocess, sys, time, urllib.request
from pathlib import Path

TEST = Path(os.environ.get("DS41_TEST", "/home/ubuntu/ds41_staging_wt/contrib/models/deepseek_v41_8d_trn3/test"))
sys.path.insert(0, str(TEST))
from performance import run_bench as rb  # noqa: E402  (loads the port, sets the host env)

MODEL = os.environ.get("MODEL", "/mnt/nvme/hf")
SONNET = str(TEST / "performance" / "sonnet.txt")
HERE = Path(__file__).resolve().parent


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def serve(name, port, ctx, devices=None, max_num_seqs=1, extra_env=None, out=None):
    out = Path(out or HERE / "results"); out.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    if devices:   # NEURON_VISIBLE_DEVICES counts logical cores (4 a device at LNC=2): "0-31" = devices 0-7
        env["NEURON_VISIBLE_DEVICES"] = devices
    env.update(extra_env or {})
    cmd = [sys.executable, "-m", "performance._serve_main", "--model", MODEL,
           "--served-model-name", rb.SERVED_NAME, "--tensor-parallel-size", "32",
           "--max-model-len", str(ctx), "--max-num-seqs", str(max_num_seqs), "--port", str(port),
           "--no-enable-prefix-caching"]
    def child():
        os.setsid()
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL, 0, 0, 0)
    slog = out / f"server_{name}.log"
    proc = subprocess.Popen(cmd, stdout=slog.open("w"), stderr=subprocess.STDOUT, preexec_fn=child,
                            cwd=str(TEST), env=env)
    proc.t_start = time.time(); proc.slog = slog; proc.port = port; proc.name = name
    log(f"engine {name}: devices {devices or 'default 0-7'} port {port} ctx {ctx} pid {proc.pid}")
    return proc


def wait_healthy(proc, timeout=3600):
    rb._wait_healthy(proc, proc.port, timeout)
    proc.t_healthy = time.time()
    log(f"engine {proc.name}: healthy after {proc.t_healthy - proc.t_start:.0f} s")


def stop(proc):
    if proc.poll() is None:
        try:
            pg = os.getpgid(proc.pid)
            os.killpg(pg, signal.SIGINT)
            proc.wait(timeout=180)
        except Exception:
            try:
                os.killpg(pg, signal.SIGKILL); proc.wait(timeout=60)
            except Exception:
                pass
    log(f"engine {proc.name}: stopped")


def wait_devices_free(timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout and rb._device_holders():
        time.sleep(3)
    return not rb._device_holders()


def bench(proc, tag, isl=512, out_len=128, batch=1, n=8, warmups=None, out=None):
    """One `vllm bench serve` run with the bench's own flags. Returns the metrics + acceptance."""
    out = Path(out or HERE / "results"); out.mkdir(parents=True, exist_ok=True)
    flags = rb._sonnet_flags(isl, SONNET)(out_len)
    c0 = rb._scrape_spec_counters(proc.port)
    t0 = time.time()
    raw = rb._bench_serve(MODEL, proc.port, flags, batch, n, out, f"{tag}.json", 1800, server=proc,
                          server_log=proc.slog, num_warmups=n if warmups is None else warmups)
    t1 = time.time()
    c1 = rb._scrape_spec_counters(proc.port)
    if raw is None:
        return None
    m = rb._metrics(raw)
    d = {k: c1[k] - c0[k] for k in c0}
    m["median_itl_ms"] = raw.get("median_itl_ms")
    m["accept_pct"] = 100.0 * d["accepted_tokens"] / d["draft_tokens"] if d["draft_tokens"] else None
    m["tokens_per_step"] = 1 + d["accepted_tokens"] / d["num_drafts"] if d["num_drafts"] else None
    m["t0"], m["t1"], m["tag"], m["engine"] = t0, t1, tag, proc.name
    return m


def metric(port, name):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=30) as r:
        tot = 0.0
        for line in r.read().decode().splitlines():
            if line.startswith(name):
                tot += float(line.rpartition(" ")[2])
        return tot
