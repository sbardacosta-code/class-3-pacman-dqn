#!/usr/bin/env python3
"""Benchmark disposable DQN copies; never train or modify a saved experiment.

The archived clipped notebook supplies the exact architecture, environment,
single-action helper, and legacy Double DQN learning update. Only timing JSON is
written. The main training notebook and all existing checkpoints stay untouched.
"""

from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import random
import time


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments" / "clipped_2000" / "pacman_dqn.ipynb"
WARMUP = 20
ITERATIONS = 100


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / ".execution/device_benchmark.json")
    parser.add_argument("--require-mps", action="store_true")
    args = parser.parse_args()
    started = time.perf_counter()
    deadline = started + 110
    import ale_py
    import gymnasium as gym
    import numpy as np
    import torch
    from torch import nn

    torch.set_num_threads(min(4, torch.get_num_threads()))
    if args.require_mps and not torch.backends.mps.is_available():
        raise RuntimeError("MPS_UNAVAILABLE: GPU access is unavailable in this process; "
                           "rerun with approved unsandboxed device access if sandbox blocked it.")
    gym.register_envs(ale_py)
    namespace = {"gym": gym, "torch": torch, "nn": nn, "np": np, "random": random}
    source_bytes = SOURCE.read_bytes()
    notebook = json.loads(source_bytes)
    for index in (10, 15, 17, 21, 23):
        cell = notebook["cells"][index]
        code = "".join(cell["source"])
        ast.parse(code)
        exec(compile(code, f"archived_notebook:cell-{index}", "exec"), namespace)
    environments = []
    timings = []

    def synchronize(device):
        if device == "mps":
            torch.mps.synchronize()

    def measure(label, operation, device="cpu", batch=1, is_learning=False):
        for _ in range(WARMUP):
            if time.perf_counter() >= deadline:
                raise TimeoutError("Benchmark reached its 110-second limit.")
            operation()
        synchronize(device)
        begin = time.perf_counter()
        for _ in range(ITERATIONS):
            if time.perf_counter() >= deadline:
                raise TimeoutError("Benchmark reached its 110-second limit.")
            operation()
        synchronize(device)
        duration = time.perf_counter() - begin
        entry = {"operation": label, "device": device, "batch_size": batch,
                 "warmup_iterations": WARMUP, "timed_iterations": ITERATIONS,
                 "elapsed_seconds": duration,
                 "milliseconds_per_call": duration * 1000 / ITERATIONS}
        if not is_learning:
            entry["milliseconds_per_decision"] = duration * 1000 / ITERATIONS / batch
            entry["decisions_per_second"] = ITERATIONS * batch / duration
        timings.append(entry)
        print(f"{label} {device} batch={batch}: {entry['milliseconds_per_call']:.3f} ms/call",
              flush=True)
        return entry

    result = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_notebook": str(SOURCE.relative_to(ROOT)),
        "source_notebook_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "cpu_threads": torch.get_num_threads(),
        "mps_available": torch.backends.mps.is_available(),
        "disposable_models_only": True, "saved_checkpoints_loaded_or_modified": False,
        "timings": timings,
        "scope": "Synthetic updates benchmark speed only; they are not an agent experiment. "
                 "Batched greedy action timing includes host/device transfer and reading all actions.",
    }
    try:
        observations = []
        for index in range(8):
            env = namespace["make_env"]()
            environments.append(env)
            obs, _ = env.reset(seed=70001 + index)
            observations.append(np.asarray(obs))
        n_actions = int(environments[0].action_space.n)
        env_timings = {}
        env_rng = random.Random(112358)
        for batch_size in (1, 4, 8):
            def step_group(batch=batch_size):
                for index in range(batch):
                    obs, _, ended, truncated, _ = environments[index].step(env_rng.randrange(n_actions))
                    if ended or truncated:
                        obs, _ = environments[index].reset(seed=env_rng.randrange(100000, 200000))
                    observations[index] = np.asarray(obs)
            env_timings[batch_size] = measure("environment_step", step_group, batch=batch_size)

        torch.manual_seed(4242)
        initial_model = namespace["DQN"](n_actions)
        state = {key: value.clone() for key, value in initial_model.state_dict().items()}
        del initial_model
        rng = np.random.default_rng(12345)
        screens = rng.integers(0, 256, size=(32, 4, 84, 84), dtype=np.uint8)
        next_screens = rng.integers(0, 256, size=(32, 4, 84, 84), dtype=np.uint8)
        training_batch = (screens, rng.integers(0, n_actions, size=32, dtype=np.int64),
                          rng.random(32, dtype=np.float32), next_screens,
                          np.zeros(32, dtype=bool), np.zeros(32, dtype=bool))
        projections = []
        devices = ["cpu"] + (["mps"] if torch.backends.mps.is_available() else [])
        for device in devices:
            model = namespace["DQN"](n_actions).to(device)
            model.load_state_dict(state)
            model.eval()
            action_timings = {}
            for batch_size in ((1,) if device == "cpu" else (1, 4, 8)):
                action_rng = random.Random(13579)
                if batch_size == 1:
                    def choose_one():
                        return namespace["choose_action"](model, observations[0], 0., action_rng, n_actions)
                    operation = choose_one
                else:
                    @torch.no_grad()
                    def choose_batch(batch=batch_size):
                        tensor = torch.as_tensor(np.stack(observations[:batch]), device=device)
                        return model(tensor).argmax(dim=1).cpu().tolist()
                    operation = choose_batch
                action_timings[batch_size] = measure("greedy_action", operation,
                                                     device=device, batch=batch_size)
            target = namespace["DQN"](n_actions).to(device)
            target.load_state_dict(model.state_dict())
            target.eval()
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.00005)
            def learning_update():
                return namespace["learn"](model, target, optimizer, training_batch)
            learning = measure("disposable_learning_update", learning_update,
                               device=device, batch=32, is_learning=True)
            for batch_size, action in action_timings.items():
                projected_ms = (action["milliseconds_per_decision"]
                                + env_timings[batch_size]["milliseconds_per_decision"]
                                + learning["milliseconds_per_call"] / 4)
                projections.append({"device": device, "action_batch_size": batch_size,
                                    "estimated_milliseconds_per_decision": projected_ms,
                                    "estimated_decisions_per_second": 1000 / projected_ms})
            del model, target, optimizer
            if device == "mps":
                torch.mps.empty_cache()
        result["projected_training_cost"] = projections
        result["projection_assumptions"] = (
            "One learning update per four collected decisions; greedy actions; sequential CPU "
            "environments; excludes replay sampling, n-step/PER processing, logging and saving. "
            "These are timing estimates, not measured full training throughput."
        )
        result["preferred_configuration"] = min(projections,
            key=lambda entry: entry["estimated_milliseconds_per_decision"])
        result["status"] = "completed"
    except TimeoutError as error:
        result["status"] = "time_limit"
        result["error"] = str(error)
    finally:
        for env in environments:
            env.close()
        result["total_elapsed_seconds"] = time.perf_counter() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"status": result["status"],
                      "preferred_configuration": result.get("preferred_configuration"),
                      "total_elapsed_seconds": result["total_elapsed_seconds"],
                      "json": str(args.output)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
