#!/usr/bin/env python3
"""
Load-aware task-sharded launcher for the truth sprint benchmark.

Goal:
- run multiple truth sprint shards in parallel when the machine has spare capacity
- avoid overcommitting the M4 Max when other heavy jobs are already running
- shard over finer-grained work items than configs alone

Environment variables:
  PMO_BUDGET                default 10000
  PMO_RUNS                  default 1
  PMO_TASKS                 comma-separated task list
  PMO_CONFIGS               comma-separated config list
  TRUTH_SPRINT_LOGDIR       root logdir for this campaign
  TASK_SHARD_GROUPS         number of task groups to create (default 1)
  MAX_WORKERS               hard cap on worker count (default = number of work items)
  TARGET_THREADS_PER_WORKER desired threads per worker (default 4)
  MIN_THREADS_PER_WORKER    default 2
  RESERVE_CORES             keep this many logical cores free (default max(2, ncpu//4))
  WAIT_FOR_CAPACITY         1 to wait for capacity, 0 to launch with computed minimum (default 0)
  POLL_SECONDS              wait polling interval (default 60)
  LAUNCH_DRY_RUN            1 to print the launch plan without starting workers

Notes:
- This launcher shards over (config, task-group) work items rather than configs alone.
- Each work item gets its own TRUTH_SPRINT_LOGDIR shard subdirectory.
- Workers may process multiple work items sequentially if capacity is lower than work item count.
"""

from __future__ import annotations
import math
import os
import shlex
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = ROOT / "checkpoints" / "truth_sprint_sharded"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def run(cmd: list[str]) -> str:
    return subprocess.check_output(cmd, text=True).strip()


def parse_list_env(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name, "")
    if not raw.strip():
        return default
    return [x.strip() for x in raw.split(",") if x.strip()]


def get_int_sysctl(name: str, default: int) -> int:
    try:
        return int(run(["sysctl", "-n", name]))
    except Exception:
        return default


def logical_cores() -> int:
    return get_int_sysctl("hw.ncpu", os.cpu_count() or 8)


def busy_cores_estimate() -> float:
    try:
        out = run(["ps", "-Ao", "pid,%cpu,command"])
    except Exception:
        return 0.0

    busy = 0.0
    this_pid = os.getpid()
    for line in out.splitlines()[1:]:
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            cpu = float(parts[1])
        except Exception:
            continue
        if pid == this_pid:
            continue
        if cpu < 5.0:
            continue
        busy += cpu / 100.0
    return busy


def available_cores(reserve: int) -> int:
    ncpu = logical_cores()
    busy = busy_cores_estimate()
    avail = math.floor(max(1.0, ncpu - reserve - busy))
    return max(1, avail)


def chunk_list(items: list, n_chunks: int) -> list[list]:
    n_chunks = max(1, min(n_chunks, len(items)))
    chunks = [[] for _ in range(n_chunks)]
    for i, item in enumerate(items):
        chunks[i % n_chunks].append(item)
    return [c for c in chunks if c]


def choose_parallel_plan(n_work_items: int) -> tuple[int, int, dict]:
    ncpu = logical_cores()
    reserve = int(os.getenv("RESERVE_CORES", max(2, ncpu // 4)))
    target_threads = int(os.getenv("TARGET_THREADS_PER_WORKER", "4"))
    min_threads = int(os.getenv("MIN_THREADS_PER_WORKER", "2"))
    max_workers = int(os.getenv("MAX_WORKERS", str(n_work_items)))

    avail = available_cores(reserve)
    workers = max(1, min(max_workers, n_work_items, max(1, avail // max(min_threads, 1))))
    threads = max(min_threads, min(target_threads, max(1, avail // workers)))

    while workers > 1 and workers * threads > avail:
        workers -= 1
        threads = max(min_threads, min(target_threads, max(1, avail // workers)))

    meta = {
        "logical_cores": ncpu,
        "reserve_cores": reserve,
        "busy_cores_estimate": round(busy_cores_estimate(), 2),
        "available_cores": avail,
        "workers": workers,
        "threads_per_worker": threads,
        "n_work_items": n_work_items,
    }
    return workers, threads, meta


def maybe_wait_for_capacity(n_work_items: int):
    wait_mode = os.getenv("WAIT_FOR_CAPACITY", "0") == "1"
    poll_seconds = int(os.getenv("POLL_SECONDS", "60"))
    if not wait_mode:
        return
    while True:
        workers, threads, meta = choose_parallel_plan(n_work_items)
        if workers >= 1 and threads >= int(os.getenv("MIN_THREADS_PER_WORKER", "2")):
            print(f"[launcher] capacity ok: {meta}")
            return
        print(f"[launcher] waiting for capacity... {meta}")
        time.sleep(poll_seconds)


def safe_slug(parts: list[str]) -> str:
    raw = "-".join(parts)
    return "".join(ch if ch.isalnum() or ch in "._-" else "-" for ch in raw)


def build_task_groups(tasks: list[str], n_groups: int) -> list[list[str]]:
    return chunk_list(tasks, max(1, min(n_groups, len(tasks))))


def build_work_items(configs: list[str], tasks: list[str], n_task_groups: int, root_logdir: Path) -> list[dict]:
    task_groups = build_task_groups(tasks, n_task_groups)
    items: list[dict] = []
    for config in configs:
        for group_idx, task_group in enumerate(task_groups, start=1):
            group_slug = safe_slug([config, f"g{group_idx}", *task_group])
            shard_logdir = root_logdir / f"shard_{group_slug}"
            items.append({
                "config": config,
                "tasks": task_group,
                "group_index": group_idx,
                "group_slug": group_slug,
                "logdir": str(shard_logdir),
            })
    return items


def make_worker_script(items: list[dict], threads: int, budget: str, runs: str, base_env: dict[str, str]) -> str:
    lines = [
        "set -euo pipefail",
        f"cd {shlex.quote(str(ROOT))}",
        f"export JULIA_NUM_THREADS={threads}",
        "export OPENBLAS_NUM_THREADS=1",
        "export VECLIB_MAXIMUM_THREADS=1",
    ]

    passthrough_keys = [
        "HE_BUDGET_FRACTION",
        "HE_WARMUP_EPISODES",
        "HE_EPISODES_PER_SEGMENT",
        "HE_HORIZON",
        "HE_TOPK_TRACKING",
        "HE_MAX_OPERATOR_CANDIDATES",
        "HE_MAX_STEP_ATTEMPTS",
        "HE_MIN_EXPLORATION_PER_OPERATOR",
        "HE_MULTI_CHILD_MIN_REWARD_RATIO",
        "HE_OPERATOR_PRIOR_STRENGTH",
        "HE_ALLOW_CROSSOVER",
        "PMO_BATCH_SIZE",
        "PMO_REPLAY_RATIO",
        "PMO_N_ITERATIONS",
        "PMO_GA_PER_STEP",
        "PMO_GA_CROSSOVER",
        "PMO_GA_MUTATION",
        "FRONTIER_BOOTSTRAP_SAMPLES",
        "FRONTIER_BOOTSTRAP_MIN_ENTRIES",
    ]
    for key in passthrough_keys:
        if key in base_env:
            lines.append(f"export {key}={shlex.quote(base_env[key])}")

    for item in items:
        lines.extend([
            f"export TRUTH_SPRINT_LOGDIR={shlex.quote(item['logdir'])}",
            f"export PMO_BUDGET={shlex.quote(budget)}",
            f"export PMO_RUNS={shlex.quote(runs)}",
            f"export PMO_TASKS={shlex.quote(','.join(item['tasks']))}",
            f"export PMO_CONFIGS={shlex.quote(item['config'])}",
            "julia --project test/smiles_gflownet/run_truth_sprint_benchmark.jl",
        ])
    return "\n".join(lines) + "\n"


def main():
    configs = parse_list_env("PMO_CONFIGS", ["tb", "rwmle", "tb_seeded", "rwmle_seeded"])
    tasks = parse_list_env("PMO_TASKS", ["qed", "drd2", "gsk3b", "jnk3", "albuterol_similarity", "celecoxib_rediscovery"])
    budget = os.getenv("PMO_BUDGET", "10000")
    runs = os.getenv("PMO_RUNS", "1")
    root_logdir = Path(os.getenv("TRUTH_SPRINT_LOGDIR", str(ROOT / "checkpoints" / "truth_sprint"))).resolve()
    root_logdir.mkdir(parents=True, exist_ok=True)
    task_shard_groups = int(os.getenv("TASK_SHARD_GROUPS", "1"))

    work_items = build_work_items(configs, tasks, task_shard_groups, root_logdir)
    maybe_wait_for_capacity(len(work_items))
    workers, threads, meta = choose_parallel_plan(len(work_items))
    worker_assignments = chunk_list(work_items, workers)

    print("[launcher] plan:")
    print(meta)
    print(f"[launcher] task_shard_groups={task_shard_groups}")
    print(f"[launcher] work_items={work_items}")

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    manifest_path = root_logdir / f"task_shard_manifest_{timestamp}.txt"
    manifest_lines = [
        f"timestamp={timestamp}",
        f"root_logdir={root_logdir}",
        f"configs={','.join(configs)}",
        f"tasks={','.join(tasks)}",
        f"budget={budget}",
        f"runs={runs}",
        f"task_shard_groups={task_shard_groups}",
        f"workers={workers}",
        f"threads_per_worker={threads}",
        f"logical_cores={meta['logical_cores']}",
        f"reserve_cores={meta['reserve_cores']}",
        f"busy_cores_estimate={meta['busy_cores_estimate']}",
        f"available_cores={meta['available_cores']}",
        f"n_work_items={meta['n_work_items']}",
        "",
    ]
    for idx, items in enumerate(worker_assignments, start=1):
        manifest_lines.append(f"[worker {idx}]")
        for item in items:
            manifest_lines.append(f"config={item['config']} tasks={','.join(item['tasks'])} logdir={item['logdir']}")
        manifest_lines.append("")
    manifest_path.write_text("\n".join(manifest_lines))

    if os.getenv("LAUNCH_DRY_RUN", "0") == "1":
        print(f"[launcher] dry run manifest: {manifest_path}")
        return

    launcher_meta = CHECKPOINT_DIR / f"launch_{timestamp}.meta"
    launcher_meta.write_text(
        "\n".join([
            f"timestamp={timestamp}",
            f"configs={','.join(configs)}",
            f"tasks={','.join(tasks)}",
            f"budget={budget}",
            f"runs={runs}",
            f"task_shard_groups={task_shard_groups}",
            f"workers={workers}",
            f"threads_per_worker={threads}",
            f"logical_cores={meta['logical_cores']}",
            f"reserve_cores={meta['reserve_cores']}",
            f"busy_cores_estimate={meta['busy_cores_estimate']}",
            f"available_cores={meta['available_cores']}",
            f"root_logdir={root_logdir}",
            f"manifest={manifest_path}",
        ]) + "\n"
    )

    base_env = os.environ.copy()
    for idx, items in enumerate(worker_assignments, start=1):
        log_path = CHECKPOINT_DIR / f"truth_sprint_worker{idx}_{timestamp}.log"
        pid_path = CHECKPOINT_DIR / f"truth_sprint_worker{idx}_{timestamp}.pid"
        meta_path = CHECKPOINT_DIR / f"truth_sprint_worker{idx}_{timestamp}.meta"
        script_path = CHECKPOINT_DIR / f"truth_sprint_worker{idx}_{timestamp}.sh"

        script = make_worker_script(items, threads, budget, runs, base_env)
        script_path.write_text(script)
        script_path.chmod(0o755)

        meta_lines = [
            f"worker={idx}",
            f"threads={threads}",
            f"log={log_path}",
            f"script={script_path}",
        ]
        for item in items:
            meta_lines.append(f"config={item['config']} tasks={','.join(item['tasks'])} logdir={item['logdir']}")
        meta_path.write_text("\n".join(meta_lines) + "\n")

        with open(log_path, "ab") as f:
            proc = subprocess.Popen(["/bin/bash", str(script_path)], cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
        pid_path.write_text(str(proc.pid) + "\n")
        print(f"[launcher] started worker {idx} pid={proc.pid} threads={threads}")
        for item in items:
            print(f"[launcher]   item: config={item['config']} tasks={item['tasks']} logdir={item['logdir']}")
        print(f"[launcher] log: {log_path}")


if __name__ == "__main__":
    main()
