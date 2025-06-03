import nbformat
from nbformat.validator import normalize

paths = ["sketch.ipynb", "eda.ipynb"]
for path in paths:
    # Load the notebook
    with open(path, "r", encoding="utf-8") as f:
        nb = nbformat.read(f, as_version=4)

    # Normalize to regenerate unique IDs
    normalize(nb)

    # Save the fixed notebook back to disk
    with open(path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)

    print(f"Fixed and saved: {path}")
