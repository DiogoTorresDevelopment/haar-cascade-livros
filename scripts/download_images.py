# scripts/download_images.py

import os
import requests
from app.utils.file_utils import ensure_dir
from app.services.logger import get_logger

logger = get_logger(__name__)

# Caminho raiz do projeto (3 níveis acima)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

def download_images_from_txt(txt_path, save_folder):
    ensure_dir(save_folder)

    with open(txt_path, "r", encoding="utf-8") as f:
        urls = [line.strip() for line in f if line.strip()]

    prefix = "pos" if "positiv" in txt_path.lower() else "neg"

    for i, url in enumerate(urls):
        try:
            ext = os.path.splitext(url)[-1].lower()
            ext = ext if ext in [".jpg", ".jpeg", ".png"] else ".jpg"
            filename = os.path.join(save_folder, f"{prefix}_{i + 1:03d}{ext}")

            logger.info(f"Baixando imagem: {url}")
            response = requests.get(url, timeout=10)
            response.raise_for_status()

            with open(filename, "wb") as f:
                f.write(response.content)

            logger.info(f"Imagem salva como: {filename}")
        except Exception as e:
            logger.error(f"Erro ao baixar {url}: {e}")

if __name__ == "__main__":
    tipo = input("Baixar imagens (positivas/negativas)? ").strip().lower()
    if tipo not in ["positivas", "negativas"]:
        print("Tipo inválido. Use 'positivas' ou 'negativas'.")
    else:
        txt_path = os.path.join(BASE_DIR, "scripts", f"{tipo}.txt")
        folder_name = "positives" if tipo == "positivas" else "negatives"
        txt_path = os.path.join(BASE_DIR, "scripts", f"{tipo}.txt")
        folder = os.path.join(BASE_DIR, "dataset", folder_name)
        download_images_from_txt(txt_path, folder)
