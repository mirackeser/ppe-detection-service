"""Build the project-backed Colab notebook; no remote source repository required."""
import json
from pathlib import Path


def main():
    cells = []
    def add(kind, source):
        cell = {"cell_type": kind, "id": f"ppe-{len(cells):02}", "metadata": {},
                "source": source.splitlines(keepends=True)}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add("markdown", "# PPE training — Colab GPU\nGPU çalışma zamanını seçin. Yerelde `python -m scripts.colab_bundle` ile oluşturulan `ppe-colab-source.zip` dosyasını aşağıdaki hücrede yükleyin. Veri, ağırlık ve gizli ayarlar pakete dahil değildir.\n")
    add("code", "%pip install ultralytics==8.4.183\n")
    add("code", 'from google.colab import files\nfrom pathlib import Path\nimport zipfile\nuploaded = files.upload()\narchive = next(name for name in uploaded if name.endswith(".zip"))\nroot = Path("/content/PPE")\nroot.mkdir(exist_ok=True)\nwith zipfile.ZipFile(archive) as bundle:\n    for member in bundle.namelist():\n        assert (root / member).resolve().is_relative_to(root), "Unsafe archive path"\n    bundle.extractall(root)\n%cd /content/PPE\n%pip install -e ".[test]"\n')
    add("code", 'import torch\nassert torch.cuda.is_available(), "GPU çalışma zamanını seçin"\nprint(torch.cuda.get_device_name(0))\n')
    add("markdown", "## İncelenmiş veri sürümünü yükle\nYerelde oluşturulan ve verify_release kontrolünü geçen veri sürümünün ZIP dosyasını yükleyin. Eski Construction-PPE kaynağı artık otomatik eğitime alınmaz. Pilot yalnız baret/baş/kişi içerir; yelek ve nihai test başarısı iddiası taşımaz.\n")
    add("code", 'import subprocess, sys, yaml\nrelease_upload = files.upload()\nrelease_zip = next(name for name in release_upload if name.endswith(".zip"))\nrelease_root = Path("/content/PPE/datasets/reviewed-pilot")\nfrom scripts.prepare_shel5k import safe_extract\nsafe_extract(Path(release_zip), release_root)\nconfig_path = release_root / "data.yaml"\nconfig = yaml.safe_load(config_path.read_text())\nconfig["path"] = str(release_root)  # Only relocate the local dataset root.\nconfig_path.write_text(yaml.safe_dump(config))\nfrom scripts.verify_release import verify_release\nverification = verify_release(release_root)\nassert verification["passed"], verification["errors"]\nprint(verification)\ndata_path = str(config_path)\n')
    add("markdown", "## Kısa deneme\n3 epoch yalnızca eğitim yolunun çalıştığını sınar. Üretilen train_batch ve val_batch görsellerini kontrol edin.\n")
    add("code", 'subprocess.run([sys.executable, "-m", "scripts.train", "--data", data_path, "--device", "0", "--epochs", "3", "--name", "ppe-smoke"], check=True)\n')
    add("markdown", "## Pilot sınırı\nBu küçük sürüm en fazla 3 epoch eğitim hattı denemesine izin verir. Baret örnekleri ve sahne kapsamı artırılmadan uzun eğitim veya model başarısı karşılaştırması yapılmaz.\n")
    add("markdown", "## Sonuçları yedekle\nSon hücreler Google Drive izni ister ve eğitim çıktılarıyla paket sürümlerini saklar. Oturum kapanmadan yerel kopya da indirin.\n")
    add("code", 'from google.colab import drive\nimport shutil\ndrive.mount("/content/drive")\nrun = Path("runs/train/ppe-smoke")\nassert (run / "weights/best.pt").exists()\n(run / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True))\narchive = shutil.make_archive("/content/ppe-pilot-smoke", "zip", run)\nshutil.copy2(archive, "/content/drive/MyDrive/ppe-pilot-smoke.zip")\nfiles.download(archive)\n')
    notebook = {"cells": cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}}, "nbformat": 4, "nbformat_minor": 5}
    path = Path(__file__).resolve().parents[1] / "notebooks/train_colab.ipynb"
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
