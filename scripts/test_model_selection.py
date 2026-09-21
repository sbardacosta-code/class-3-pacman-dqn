"""Exercise notebook selection logic with disposable stubs; no ML or gameplay."""

import copy
import json
from pathlib import Path
import random
import shutil
import tempfile
from types import SimpleNamespace

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def main():
    notebook = json.loads((ROOT / "pacman_dqn.ipynb").read_text())
    namespace = {}
    exec("".join(notebook["cells"][10]["source"]), namespace)
    classroom = set(namespace["EVAL_SEEDS"])
    validation = set(namespace["VALIDATION_SEEDS"])
    heldout = set(range(30001, 30021))
    assert len(validation) == 10 and not (classroom & validation or classroom & heldout or validation & heldout)

    class FakeModel:
        def __init__(self, token):
            self.token = token

        def state_dict(self):
            return {"token": SimpleNamespace(detach=lambda: SimpleNamespace(cpu=lambda: self.token))}

    class FakeReplay:
        def __len__(self):
            return 17

        def state_dict(self, **kwargs):
            return {"disposable": True}

    def evaluate(model, seeds):
        assert seeds == namespace["VALIDATION_SEEDS"]
        scores = list(namespace["fake_scores"])
        return {"seeds": list(seeds), "scores": scores,
                "mean": sum(scores) / len(scores), "steps": [10] * len(scores),
                "time_limited": [False] * len(scores)}

    def save_checkpoint(model, path, episode, steps):
        path.write_text(json.dumps({"token": model.token, "episode": episode, "steps": steps}))

    def fake_torch_save(payload, path, **kwargs):
        assert kwargs.get("pickle_protocol") == 5
        namespace["saved_learner_snapshot"] = copy.deepcopy(payload)
        path.write_text("disposable learner snapshot")

    namespace.update({"evaluate": evaluate, "save_checkpoint": save_checkpoint,
                      "np": SimpleNamespace(std=np.std, random=SimpleNamespace(get_state=lambda: [])),
                      "json": json, "shutil": shutil, "random": random,
                      "time": SimpleNamespace(time=lambda: 100., monotonic=lambda: 50.),
                      "torch": SimpleNamespace(get_rng_state=lambda: [], save=fake_torch_save),
                      "DEVICE": SimpleNamespace(type="cpu"), "target": FakeModel(999),
                      "optimizer": SimpleNamespace(state_dict=lambda: {}),
                      "replay": FakeReplay(), "train_rng": random.Random(42),
                      "save_metrics": lambda rows, path: path.write_text("disposable csv"),
                      "plot_training": lambda rows, path: path.write_text("disposable plot"),
                      "history": [], "training_states": [], "training_lives_lost": 0,
                      "started": 0., "active_started": 0.})
    exec("".join(notebook["cells"][49]["source"]), namespace)
    exec("".join(notebook["cells"][51]["source"]), namespace)
    with tempfile.TemporaryDirectory(prefix="pacman_selection_test_") as temporary:
        namespace["RUN_DIR"] = Path(temporary)

        def reset():
            namespace.update(selection_history=[], best_validation_metric=-float("inf"),
                             selected_episode=0, selected_updates=0, total_steps=0, updates=0)

        def candidate(episode, token, scores, reason="periodic"):
            namespace.update(model=FakeModel(token), fake_scores=scores,
                             total_steps=episode * 10, updates=episode * 2)
            namespace["assess_candidate"](episode, reason)

        reset()
        candidate(0, 10, [100.] * 10, "inherited starting model")
        candidate(250, 20, [90.] * 10)
        namespace.update(model=FakeModel(30), fake_scores=[100.] * 10,
                         total_steps=4000, updates=800, completed_episodes=400, episodes_started=400)
        namespace["save_training_result"]("time_budget")
        selected = json.loads((Path(temporary) / "trained.pt").read_text())
        latest = json.loads((Path(temporary) / "last_trained.pt").read_text())
        summary = json.loads((Path(temporary) / "training_summary.json").read_text())
        assert selected["token"] == 10 and selected["episode"] == 0
        assert latest["token"] == 30 and latest["episode"] == 400
        assert summary["selected_additional_episodes"] == 0 and summary["selected_recorded_updates"] == 0
        assert summary["completed_episodes"] == 400 and summary["learning_updates"] == 800
        assert namespace["saved_learner_snapshot"]["model"]["token"] == 30

        reset()
        candidate(0, 40, [100.] * 10, "inherited starting model")
        candidate(250, 50, [200.] * 10)
        candidate(500, 60, [500.] * 5 + [0.] * 5)  # Higher mean, worse selected consistency metric.
        assert namespace["selected_episode"] == 250
        candidate(750, 70, [200.] * 10)  # Exact tie must retain earlier checkpoint.
        assert namespace["selected_episode"] == 250
        namespace.update(model=FakeModel(80), fake_scores=[300.] * 10,
                         total_steps=10000, updates=2000, completed_episodes=1000, episodes_started=1000)
        namespace["save_training_result"]("completed")
        selected = json.loads((Path(temporary) / "trained.pt").read_text())
        record = json.loads((Path(temporary) / "model_selection.json").read_text())
        assert selected["token"] == 80 and selected["episode"] == 1000
        assert record["selected_metric"] == 300. and record["selected_updates"] == 2000
        assert [row["selected_when_evaluated"] for row in record["candidate_evaluations"]] == [True, True, False, False, True]
        assert record["candidate_evaluations"][-1]["reason"] == "final candidate"
    print("PASS: disjoint seeds; initial winner retained; earlier wins ties; variance penalty; "
          "final improvement selected; final learner and selected checkpoint kept distinct. No ML ran.")


if __name__ == "__main__":
    main()
