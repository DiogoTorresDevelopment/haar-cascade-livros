# app/core/trainer.py

import os
import subprocess
import cv2
from app.services.logger import get_logger
from app.utils.file_utils import ensure_dir
from app.config.settings import (
    POSITIVE_PATH,
    NEGATIVE_PATH,
    ANNOTATIONS_PATH,
    VEC_FILE_PATH,
    MODEL_DIR,
    TRAINING_PARAMS,
    CREATESAMPLES_EXE,
    TRAINCASCADE_EXE,
)

logger = get_logger(__name__)


def find_image(base_name):
    """Tenta localizar uma imagem com base no nome base e extensões comuns."""
    for ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        path = os.path.join(POSITIVE_PATH, base_name + ext)
        if os.path.exists(path):
            return path
    return None


def generate_vec():
    """
    Gera o arquivo .vec usando opencv_createsamples.exe.
    """
    ensure_dir(os.path.dirname(VEC_FILE_PATH))
    positives_txt = os.path.join(ANNOTATIONS_PATH, "positives.txt")
    logger.info("Gerando arquivo de amostras positivas: positives.txt")

    with open(positives_txt, "w") as outfile:
        for file in os.listdir(ANNOTATIONS_PATH):
            if file.endswith(".txt") and file != "positives.txt":
                base_name = os.path.splitext(file)[0]
                image_path = find_image(base_name)

                if not image_path:
                    logger.warning(f"Imagem não encontrada: {base_name}.[jpg/png/jpeg/bmp]")
                    continue

                annotation_path = os.path.join(ANNOTATIONS_PATH, file)

                try:
                    with open(annotation_path, "r") as ann_file:
                        coords = ann_file.read().strip()

                    # Validação do bounding box com a imagem
                    img = cv2.imread(image_path)
                    if img is None:
                        logger.warning(f"Imagem inválida: {image_path}")
                        continue

                    h_img, w_img = img.shape[:2]
                    values = coords.split()
                    num_rects = int(values[0])
                    boxes = values[1:]

                    valid = True
                    for i in range(num_rects):
                        x = int(boxes[i * 4])
                        y = int(boxes[i * 4 + 1])
                        w = int(boxes[i * 4 + 2])
                        h = int(boxes[i * 4 + 3])
                        if x + w > w_img or y + h > h_img:
                            logger.error(f"⚠️ Bounding box fora da imagem: {image_path}")
                            valid = False
                            break

                    if not valid:
                        continue

                    relative_path = os.path.relpath(image_path, start=os.path.dirname(positives_txt))
                    line = f"{relative_path} {coords}"
                    outfile.write(line + "\n")

                except Exception as e:
                    logger.error(f"Erro ao processar anotação: {file} — {e}")

    command = [
        CREATESAMPLES_EXE,
        "-info", positives_txt,
        "-num", str(TRAINING_PARAMS["num_positive"]),
        "-w", str(TRAINING_PARAMS["width"]),
        "-h", str(TRAINING_PARAMS["height"]),
        "-vec", VEC_FILE_PATH
    ]

    logger.info("Executando opencv_createsamples.exe para gerar o .vec...")
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        logger.error("❌ Erro ao gerar .vec:")
        logger.error(result.stderr)
    else:
        logger.info("✅ Arquivo .vec gerado com sucesso.")


def train_classifier():
    """
    Executa o treinamento com opencv_traincascade.exe
    """
    ensure_dir(MODEL_DIR)

    bg_txt = os.path.join(NEGATIVE_PATH, "bg.txt")
    if not os.path.exists(bg_txt):
        logger.error(f"Arquivo bg.txt não encontrado em {NEGATIVE_PATH}.")
        return

    command = [
        TRAINCASCADE_EXE,
        "-data", MODEL_DIR,
        "-vec", VEC_FILE_PATH,
        "-bg", bg_txt,
        "-numPos", str(TRAINING_PARAMS["num_positive"]),
        "-numNeg", str(TRAINING_PARAMS["num_negative"]),
        "-numStages", str(TRAINING_PARAMS["num_stages"]),
        "-minHitRate", str(TRAINING_PARAMS["min_hit_rate"]),
        "-maxFalseAlarmRate", str(TRAINING_PARAMS["max_false_alarm_rate"]),
        "-featureType", TRAINING_PARAMS["feature_type"],
        "-w", str(TRAINING_PARAMS["width"]),
        "-h", str(TRAINING_PARAMS["height"]),
        "-mode", "ALL"
    ]

    logger.info("🔧 Iniciando treinamento do classificador Haar Cascade...")
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    if result.returncode != 0:
        logger.error("❌ Erro no treinamento:")
        logger.error(result.stderr)
    else:
        logger.info("✅ Treinamento finalizado com sucesso. Modelo salvo em model/cascade.xml")


def run_trainer():
    generate_vec()
    train_classifier()


if __name__ == "__main__":
    run_trainer()
