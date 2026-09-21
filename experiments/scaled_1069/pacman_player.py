"""A floating GIF player for local notebooks. The learning code stays in the notebook."""

import os
from pathlib import Path
import subprocess
import sys

_player = None
PREVIEW_PLAYS = 2


def _advance(state, frame_count):
    """Move to the next frame and stop after two complete plays."""
    if state["index"] == frame_count - 1:
        state["completed"] += 1
        if state["completed"] >= PREVIEW_PLAYS:
            state["paused"] = True
            return True
        state["index"] = 0
    else:
        state["index"] += 1
    return False


def show_popup(gif_path, title="Pac-Man sample · 4× speed"):
    """Start a separate GUI process so playback never holds up training.

    Return False on machines without Tk or a desktop; the notebook keeps its
    inline GIF as a fallback. A new sample replaces the previous popup.
    """
    global _player
    try:
        import tkinter  # noqa: F401: check the current kernel's GUI support
    except ImportError:
        return False
    if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
        return False
    path = Path(gif_path).resolve(strict=True)
    if _player is not None and _player.poll() is None:
        try:
            _player.terminate()
            _player.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            # A Tk window can ignore termination briefly on macOS. Never let a
            # preview window pause or fail the training loop.
            _player.kill()
        except OSError:
            pass
    # Keep GUI errors available without filling the notebook with subprocess logs.
    with path.with_suffix(".player.log").open("a") as log:
        _player = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), str(path), title],
            stdout=log, stderr=log,
        )
    return True


def play(gif_path, title):
    """Play the saved timing, with pause, replay, and an always-on-top toggle."""
    import tkinter as tk
    from PIL import Image, ImageTk, ImageSequence

    root = tk.Tk()
    root.title(title)
    root.configure(bg="#101522")
    root.attributes("-topmost", True)
    root.geometry("380x590+70+70")
    root.resizable(False, False)
    tk.Label(root, text=title, bg="#101522", fg="#ffe260",
             font=("Helvetica", 15, "bold"), pady=12).pack()

    # GIF timing already includes the notebook's speed multiplier.
    with Image.open(gif_path) as gif:
        frames = []
        delays = []
        for frame in ImageSequence.Iterator(gif):
            picture = frame.convert("RGB").resize((320, 420), Image.Resampling.NEAREST)
            frames.append(ImageTk.PhotoImage(picture, master=root))
            delays.append(max(20, frame.info.get("duration", 60)))
    screen = tk.Label(root, bg="#101522")
    screen.pack()
    state = {"index": 0, "paused": False, "completed": 0}

    def tick():
        index = state["index"]
        screen.configure(image=frames[index])
        if not state["paused"]:
            if _advance(state, len(frames)):
                pause.configure(text="Play again")
        root.after(delays[index], tick)

    def toggle_pause():
        if state["paused"] and state["completed"] >= PREVIEW_PLAYS:
            state.update(index=0, completed=0)
        state["paused"] = not state["paused"]
        pause.configure(text="Play" if state["paused"] else "Pause")

    def replay():
        state.update(index=0, paused=False, completed=0)
        pause.configure(text="Pause")

    controls = tk.Frame(root, bg="#101522", pady=10)
    controls.pack()
    pause = tk.Button(controls, text="Pause", command=toggle_pause)
    pause.pack(side="left", padx=5)
    tk.Button(controls, text="Replay", command=replay).pack(side="left", padx=5)
    on_top = tk.BooleanVar(value=True)
    tk.Checkbutton(root, text="Keep above other windows", variable=on_top,
                   command=lambda: root.attributes("-topmost", on_top.get()),
                   bg="#101522", fg="white", selectcolor="#101522").pack()
    tk.Label(root, text=f"Recorded sample · plays {PREVIEW_PLAYS} times",
             bg="#101522", fg="#b9c4d6", font=("Helvetica", 10)).pack(pady=8)
    root.bind("<Escape>", lambda event: root.destroy())
    root.lift()
    tick()
    root.mainloop()


if __name__ == "__main__":
    play(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "Pac-Man sample")
