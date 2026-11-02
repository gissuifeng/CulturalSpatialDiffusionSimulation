from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parent

def main():
    parser = argparse.ArgumentParser(description="CSDS figure notebooks")
    parser.add_argument("action", choices=["list", "run"])
    parser.add_argument("--figure", help="Notebook stem, for example figure16")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    files = sorted((ROOT / "notebooks").glob("figure*.ipynb"))
    if args.action == "list":
        for path in files: print(path.stem)
        return
    matches = [path for path in files if path.stem == args.figure]
    if len(matches) != 1: parser.error("Use --figure with a name shown by list")
    import nbformat
    from nbclient import NotebookClient
    path = matches[0]
    notebook = nbformat.read(path, as_version=4)
    NotebookClient(notebook, timeout=1200, kernel_name="python3",
                   resources={"metadata": {"path": str(path.parent)}}).execute()
    args.output.mkdir(parents=True, exist_ok=True)
    nbformat.write(notebook, args.output / path.name)

if __name__ == "__main__": main()
