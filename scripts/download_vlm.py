"""Download public model artifacts only, without running remote model code."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["HuggingFaceTB/SmolVLM-256M-Instruct", "HuggingFaceTB/SmolVLM-500M-Instruct"],
                        default="HuggingFaceTB/SmolVLM-256M-Instruct")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download
    destination = snapshot_download(args.model, cache_dir="models/huggingface", token=False,
                                    allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model", "*.jinja", "LICENSE*", "README.md"],
                                    max_workers=2)
    provenance = {"model": args.model, "snapshot": str(Path(destination).resolve()),
                  "revision": Path(destination).name,
                  "source": f"https://huggingface.co/{args.model}", "remote_code_enabled": False}
    output = Path("models") / (args.model.split("/")[-1] + "-provenance.json")
    output.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Downloaded public model snapshot: {destination}")


if __name__ == "__main__":
    main()
