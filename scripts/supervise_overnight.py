#!/usr/bin/env python3
"""Finish stage one, prepare one verified continuation, and enforce its deadline.

This local supervisor never publishes or starts a third training run. Its status
file is the handoff point for the scheduled review and publication follow-up.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
EXECUTION = ROOT / ".execution"
PYTHON = ROOT / ".venv/bin/python"
STATUS = EXECUTION / "overnight_status.json"
STOP = EXECUTION / "overnight_stop"
PUBLIC = "https://raw.githubusercontent.com/sbardacosta-code/class-3-pacman-dqn/main/"


def utc_now():
    return datetime.now(timezone.utc)


def parse_utc(value):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if result.tzinfo is None:
        raise ValueError("A timezone is required for a deadline")
    return result.astimezone(timezone.utc)


def sha(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read(path):
    return json.loads(path.read_text())


def save_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def update(state, stage, **fields):
    state.update(stage=stage, updated_utc=utc_now().isoformat(), **fields)
    save_json(STATUS, state)
    print(f"{state['updated_utc']} {stage}", flush=True)


def run_step(arguments, log_name):
    with (EXECUTION / log_name).open("w") as log:
        subprocess.run([str(PYTHON), *map(str, arguments)], cwd=ROOT,
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def archive_phase1(run):
    destination = ROOT / "experiments/overnight_phase1"
    if destination.exists():
        raise RuntimeError("Stage-one archive already exists; inspect status before retrying")
    temporary = ROOT / "experiments/.overnight_phase1.tmp"
    if temporary.exists():
        raise RuntimeError("Incomplete archive needs inspection before retrying")
    temporary.mkdir(parents=True)
    shutil.copytree(ROOT / "results", temporary / "results")
    notebook = read(ROOT / "pacman_dqn.ipynb")
    for cell in notebook["cells"]:
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if "text/html" in data:
                value = data["text/html"]
                if isinstance(value, list):
                    value = "".join(value)
                data["text/html"] = value.replace(
                    PUBLIC + "results/demos/",
                    PUBLIC + "experiments/overnight_phase1/results/demos/")
    save_json(temporary / "pacman_dqn.ipynb", notebook)
    for name in ("experiment_plan.md", "overnight_plan.md"):
        if name == "experiment_plan.md":
            shutil.copy2(ROOT / name, temporary / "frozen_experiment_plan.md")
        text = (ROOT / name).read_text()
        for previous in ("scaled_1069", "clipped_2000", "original_100"):
            text = text.replace(f"(experiments/{previous}/README.md)", f"(../{previous}/README.md)")
        (temporary / name).write_text(text)
    verification = read(temporary / "results/verification.json")
    verification["notebook_sha256"] = sha(temporary / "pacman_dqn.ipynb")
    verification["archive_display_note"] = "Only HTML GIF URLs changed to this preserved archive; embedded outputs and sources retained."
    save_json(temporary / "results/verification.json", verification)
    report = read(temporary / "results/report_data.json")
    report["source_hashes"]["notebook_after_html_update"] = verification["notebook_sha256"]
    save_json(temporary / "results/report_data.json", report)
    summary, comparison = read(run / "training_summary.json"), read(run / "comparison.json")
    lines = ["# Overnight experiment: preserved first stage", "",
             "This is the completed first stage of the [overnight continuation](../../README.md). "
             "Its selected model is an intermediate result. The continuation recipe was fixed before "
             "these classroom scores and does not use them to choose training settings.", "",
             f"Run `{run.name}`: status `{summary['status']}`, {summary['completed_episodes']:,} completed "
             f"games, {summary['total_decisions']:,} decisions, {summary['learning_updates']:,} recorded "
             f"updates, and {summary['elapsed_seconds_including_periodic_demos']:.3f} training seconds.", "",
             "[Executed notebook](pacman_dqn.ipynb) · [Config](results/config.json) · "
             "[Training CSV](results/training.csv) · [Summary](results/training_summary.json) · "
             "[Comparison](results/comparison.json) · [Selection](results/model_selection.json) · "
             "[Initial plan](experiment_plan.md) · [Overnight extension](overnight_plan.md)", "",
             "| Seed | Fresh baseline | Stage-one selected |", "| --- | ---: | ---: |"]
    for seed, before, after in zip(comparison["before"]["seeds"], comparison["before"]["scores"], comparison["after"]["scores"]):
        lines.append(f"| {seed} | {before:g} | {after:g} |")
    lines += [f"| Mean | {comparison['before']['mean']:g} | {comparison['after']['mean']:g} |", "",
              "![Training dashboard](results/training_dashboard.png)", "",
              "The GIFs show at most the first 20 seconds. The final GIF is the best of five games; "
              "it does not demonstrate typical performance. The full stage-one ZIP and all large "
              "checkpoints remain local under `pacman_runs/`.", "",
              "![Fresh baseline](results/demos/episode_0000.gif)", "",
              "![Stage-one best selected game](results/demos/final_best.gif)", "",
              "<details><summary>Every stage-one intermediate gameplay GIF</summary>", ""]
    for path in sorted((temporary / "results/demos").glob("episode_*.gif")):
        if path.name != "episode_0000.gif":
            lines.extend([f"![{path.stem}](results/demos/{path.name})", ""])
    lines.extend(["</details>", ""])
    (temporary / "README.md").write_text("\n".join(lines))
    save_json(temporary / "archive_manifest.json", {
        "run_id": run.name, "archived_utc": utc_now().isoformat(),
        "original_executed_notebook_sha256": sha(ROOT / "pacman_dqn.ipynb"),
        "archived_notebook_sha256": sha(temporary / "pacman_dqn.ipynb"),
        "original_zip_sha256": sha(run.with_suffix(".zip")),
        "frozen_original_plan": "frozen_experiment_plan.md",
        "frozen_original_plan_sha256": sha(temporary / "frozen_experiment_plan.md"),
    })
    temporary.replace(destination)
    return destination


def create_bundle(run):
    completed = read(run / "training_summary.json")["completed_episodes"]
    names = ["learner_state.pt", "config.json", "training_summary.json", "training.csv",
             "model_selection.json", "demo_scores.json", "validation_best.pt", "last_trained.pt"]
    for episode in range(25, completed + 1, 25):
        names.extend([f"episode_{episode:04d}.pt", f"demos/episode_{episode:04d}.gif"])
    directory = ROOT / "pacman_runs/overnight_assets"
    directory.mkdir(exist_ok=True)
    destination = directory / "phase1-resume.zip"
    if destination.exists():
        raise RuntimeError("Resume bundle already exists; inspect before retrying")
    temporary = destination.with_suffix(".zip.tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as output:
        for name in names:
            output.write(run / name, name)
    temporary.replace(destination)
    return destination


def interrupt_owned_kernel(process):
    """Interrupt only the Jupyter kernel belonging to this supervisor's child."""
    import psutil
    matches = [child for child in psutil.Process(process.pid).children(recursive=True)
               if any("ipykernel_launcher" in item for item in child.cmdline())]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one owned notebook kernel, found {len(matches)}")
    matches[0].send_signal(signal.SIGINT)


def remember_owned_processes(process, known):
    import psutil
    try:
        parent = psutil.Process(process.pid)
        for item in [parent, *parent.children(recursive=True)]:
            known[(item.pid, item.create_time())] = item
    except psutil.NoSuchProcess:
        pass


def stop_owned_processes(process, known):
    """Jupyter kernels have separate sessions, so stop the actual owned tree.

    psutil Process signal methods check process identity against PID reuse.
    Remembered handles also cover a kernel orphaned by an executor failure.
    """
    import psutil
    remember_owned_processes(process, known)
    alive = []
    for (pid, born), item in known.items():
        try:
            if item.is_running() and item.create_time() == born:
                alive.append(item)
                item.terminate()
        except psutil.NoSuchProcess:
            pass
    _, remaining = psutil.wait_procs(alive, timeout=10)
    for item in remaining:
        try:
            item.kill()
        except psutil.NoSuchProcess:
            pass
    _, remaining = psutil.wait_procs(remaining, timeout=5)
    if remaining:
        raise RuntimeError(f"Owned notebook processes did not stop: {[item.pid for item in remaining]}")
    process.wait(timeout=5)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase1", type=Path, required=True)
    parser.add_argument("--training-deadline", required=True)
    parser.add_argument("--finish-by", required=True)
    parser.add_argument("--resume-url", required=True)
    args = parser.parse_args()
    deadline, finish = parse_utc(args.training_deadline), parse_utc(args.finish_by)
    if not utc_now() < deadline < finish:
        raise ValueError("Deadlines must be future, with a finalization reserve")
    EXECUTION.mkdir(exist_ok=True)
    with (EXECUTION / "overnight_supervisor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock.write(str(os.getpid())); lock.flush()
        state = {"schema_version": 1, "supervisor_pid": os.getpid(),
                 "phase1_run": str(args.phase1.resolve()), "training_deadline_utc": deadline.isoformat(),
                 "finish_by_utc": finish.isoformat(), "resume_bundle_url": args.resume_url,
                 "started_utc": utc_now().isoformat()}
        process, known_processes = None, {}
        try:
            update(state, "waiting_for_phase1")
            first_log = EXECUTION / "notebook_advanced.log"
            while "Executed notebook saved with all outputs." not in first_log.read_text():
                if STOP.exists():
                    update(state, "stopped_before_continuation"); return
                if utc_now() >= deadline:
                    raise RuntimeError("Deadline reached before the first stage finished")
                if "Traceback (most recent call last):" in first_log.read_text():
                    raise RuntimeError("Stage-one executor reported an error; preserve its evidence")
                time.sleep(30)
            time.sleep(1)
            if STOP.exists():
                update(state, "stopped_before_continuation"); return
            run = args.phase1.resolve()
            update(state, "verifying_phase1")
            run_step([ROOT / "scripts/build_advanced_submission.py", "--run", run], "overnight_phase1_assembly.log")
            archive = archive_phase1(run)
            update(state, "packing_resume_bundle", phase1_archive=str(archive))
            bundle = create_bundle(run)
            update(state, "preparing_continuation", resume_bundle=str(bundle), resume_bundle_sha256=sha(bundle))
            prepared = EXECUTION / "overnight_prepared.ipynb"
            manifest = EXECUTION / "overnight_pretraining_manifest.json"
            run_step([ROOT / "scripts/prepare_overnight_continuation.py", "--phase1", run,
                      "--deadline-utc", deadline.isoformat(), "--output", prepared,
                      "--manifest-output", manifest, "--resume-bundle", bundle,
                      "--resume-url", args.resume_url], "overnight_preparation.log")
            if STOP.exists() or (deadline - utc_now()).total_seconds() < 180:
                update(state, "stopped_before_continuation"); return
            shutil.copy2(prepared, ROOT / "pacman_dqn.ipynb")
            log_path = EXECUTION / "notebook_overnight.log"
            with log_path.open("w") as log:
                process = subprocess.Popen([str(PYTHON), str(ROOT / "run_notebook.py")], cwd=ROOT,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                update(state, "continuation_running", notebook_executor_pid=process.pid,
                       continuation_log=str(log_path), prepared_manifest=str(manifest))
                interrupted = False
                while process.poll() is None:
                    remember_owned_processes(process, known_processes)
                    content = log_path.read_text()
                    # These markers bracket the interruptible training try,
                    # excluding restoration and final evaluation/serialization.
                    training = (content.rfind("OVERNIGHT_COLLECTION_STARTED") >
                                content.rfind("OVERNIGHT_COLLECTION_FINISHED"))
                    if not interrupted and training and (STOP.exists() or utc_now() >= deadline):
                        interrupt_owned_kernel(process)
                        interrupted = True
                        update(state, "continuation_saving_after_interrupt", interruption_reason="user_stop" if STOP.exists() else "absolute_deadline")
                    if utc_now() >= finish:
                        stop_owned_processes(process, known_processes)
                        raise RuntimeError("Final deadline reached; stopped the owned notebook process tree")
                    time.sleep(15)
                if process.returncode != 0:
                    raise RuntimeError(f"Continuation executor exited with status {process.returncode}; inspect its saved notebook/log")
            notebook = read(ROOT / "pacman_dqn.ipynb")
            run_ids = []
            for cell in notebook["cells"]:
                for output in cell.get("outputs", []):
                    value = output.get("text", "")
                    value = "".join(value) if isinstance(value, list) else value
                    if "Saved:" in value:
                        run_ids.extend(line.split("Saved:", 1)[1].strip() for line in value.splitlines() if line.startswith("Saved:"))
            update(state, "ready_for_final_review", saved_run_paths=run_ids,
                   notebook_completed_utc=utc_now().isoformat())
            while utc_now() < finish and not (EXECUTION / "overnight_publication_complete.json").exists():
                time.sleep(30)
            update(state, "publication_complete" if (EXECUTION / "overnight_publication_complete.json").exists() else "training_finished_review_pending")
        except Exception as error:
            if process is not None:
                try:
                    stop_owned_processes(process, known_processes)
                except Exception as cleanup_error:
                    state["cleanup_error"] = f"{type(cleanup_error).__name__}: {cleanup_error}"
            update(state, "failed", error=f"{type(error).__name__}: {error}")
            raise


if __name__ == "__main__":
    main()
