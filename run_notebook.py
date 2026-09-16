"""Execute the supplied notebook in order and preserve its actual outputs."""
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parent
PATH = ROOT / "pacman_dqn.ipynb"
notebook = nbformat.read(PATH, as_version=4)

class LoggedClient(NotebookClient):
    def process_message(self, msg, cell, cell_index):
        if msg["msg_type"] == "stream":
            print(msg["content"]["text"], end="", flush=True)
        return super().process_message(msg, cell, cell_index)

def save(**kwargs):
    temporary = PATH.with_suffix(".ipynb.tmp")
    nbformat.write(notebook, temporary)
    temporary.replace(PATH)

def starting(cell, cell_index, **kwargs):
    print(f"\n--- Executing cell {cell_index} ---", flush=True)

client = LoggedClient(notebook, timeout=None, kernel_name="python3",
                      resources={"metadata": {"path": str(ROOT)}},
                      on_cell_start=starting, on_cell_executed=save)
try:
    client.execute()
finally:
    save()
print("Executed notebook saved with all outputs.", flush=True)
