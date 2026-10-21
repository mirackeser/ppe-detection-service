"""Package source only for Colab; never include env files, data, models or virtualenvs."""
from pathlib import Path
import zipfile


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "output" / "ppe-colab-source.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    files = [root / "pyproject.toml", root / "README.md"]
    files.extend(sorted((root / "config").glob("data-readiness-*.json")))
    for directory in ("ppe", "scripts"):
        files.extend(sorted((root / directory).glob("*.py")))
    files.extend(sorted(path for path in (root / "ppe/static").iterdir()
                        if path.suffix in {".html", ".css", ".js"}))
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in files:
            bundle.write(path, path.relative_to(root))
    print(output)


if __name__ == "__main__":
    main()
