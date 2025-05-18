# app/utils/file_utils.py
import os
import glob
from PIL import Image


def ensure_dir(path: str):
    """Cria o diretório se ele não existir."""
    if not os.path.exists(path):
        os.makedirs(path)


def list_images(directory: str, extensions: tuple = (".jpg", ".jpeg", ".png")) -> list:
    """Lista arquivos de imagem válidos em um diretório."""
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(directory, f"*{ext}")))
    return sorted(files)


def is_image_file(filepath: str) -> bool:
    """Verifica se o arquivo é uma imagem válida."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        return True
    except Exception:
        return False


def save_text_file(filepath: str, content: str):
    """Salva conteúdo de texto em um arquivo."""
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)


def read_text_file(filepath: str) -> str:
    """Lê conteúdo de um arquivo de texto."""
    with open(filepath, "r", encoding="utf-8") as f:
        return f.read()
