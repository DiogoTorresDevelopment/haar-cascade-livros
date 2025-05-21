# app/core/trainer.py  – configurado p/ 10 estágios
import os, re, struct, subprocess
from typing import Tuple
import cv2

from app.services.logger import get_logger
from app.utils.file_utils import list_images
from app.config.settings import (
    BASE_DIR, POSITIVE_PATH, NEGATIVE_PATH, ANNOTATIONS_PATH,
    VEC_FILE_PATH, MODEL_DIR, TRAINING_PARAMS,
    CREATESAMPLES_EXE, TRAINCASCADE_EXE,
)

logger = get_logger(__name__)

# ------------------------------------------------------------------ #
def build_positives_txt() -> Tuple[str, int]:
    info = os.path.join(ANNOTATIONS_PATH, "positives.txt")
    lines = []

    for ann in os.listdir(ANNOTATIONS_PATH):
        if ann == "positives.txt" or not ann.endswith(".txt"):
            continue
        base = ann[:-4]

        img_abs = next(
            (os.path.join(POSITIVE_PATH, f)
             for f in os.listdir(POSITIVE_PATH)
             if os.path.splitext(f)[0] == base and
                os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png")),
            None
        )
        if not img_abs:
            logger.warning(f"Ignorada '{ann}': imagem não encontrada")
            continue

        ann_line = open(os.path.join(ANNOTATIONS_PATH, ann), encoding="utf-8").read().strip()
        toks = ann_line.split()
        if len(toks) < 5 or (len(toks) - 1) % 4:
            logger.warning(f"Ignorada '{ann}': formato inválido")
            continue

        img = cv2.imread(img_abs)
        if img is None:
            logger.warning(f"Ignorada '{ann}': imagem corrompida")
            continue
        ih, iw = img.shape[:2]

        valid = True
        for i in range(1, len(toks), 4):
            x, y, w, h = map(int, toks[i:i+4])
            if w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > iw or y + h > ih:
                valid = False
                break
        if not valid:
            logger.warning(f"Ignorada '{ann}': bbox inválida")
            continue

        rel = os.path.relpath(img_abs, start=ANNOTATIONS_PATH).replace("\\", "/")
        lines.append(f"{rel} {ann_line}")

    if not lines:
        raise RuntimeError("Nenhuma anotação válida encontrada")
    with open(info, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    return info, len(lines)

# ------------------------------------------------------------------ #
def create_vec(info_path: str, line_count: int, aug: float = 5.0) -> int:
    """gera o .vec e devolve o nº real (“Created N samples”)"""
    vec_num = max(int(line_count * aug), line_count + 100)

    cmd = [
        CREATESAMPLES_EXE,
        "-info", os.path.basename(info_path),
        "-num", str(vec_num),
        "-w", str(TRAINING_PARAMS["width"]),
        "-h", str(TRAINING_PARAMS["height"]),
        "-vec", VEC_FILE_PATH,
    ]
    res = subprocess.run(cmd, cwd=ANNOTATIONS_PATH,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    logger.info(res.stdout)
    if res.returncode:
        logger.error(res.stderr)
        raise RuntimeError("createsamples falhou")

    m = re.search(r"Created\s+(\d+)\s+samples", res.stdout)
    return int(m.group(1)) if m else line_count

# ------------------------------------------------------------------ #
def vec_count(path: str) -> int:
    with open(path, "rb") as f:
        return struct.unpack("<I", f.read(4))[0]

# ------------------------------------------------------------------ #
def ensure_bg_txt() -> str:
    bg = os.path.join(NEGATIVE_PATH, "bg.txt")
    if os.path.exists(bg):
        return bg
    rels = [os.path.relpath(p, BASE_DIR).replace("\\", "/")
            for p in list_images(NEGATIVE_PATH, (".jpg", ".jpeg", ".png"))]
    with open(bg, "w", encoding="utf-8") as f:
        f.write("\n".join(rels))
    logger.info(f"bg.txt gerado ({len(rels)} imagens negativas)")
    return bg

# ------------------------------------------------------------------ #
def train(vec_real: int):
    bg = ensure_bg_txt()
    neg_count = sum(1 for _ in open(bg) if _.strip())

    stages = TRAINING_PARAMS["num_stages"]
    safety = 3 * stages
    num_pos = int(vec_real * 0.60)
    if num_pos > vec_real - safety:
        num_pos = vec_real - safety
    num_pos = max(safety, num_pos)
    num_neg = min(neg_count, TRAINING_PARAMS["num_negative"])

    logger.info(f"→ Treinando (numPos {num_pos} | numNeg {num_neg} | stages {stages})")

    cmd = [
        TRAINCASCADE_EXE,
        "-data", MODEL_DIR, "-vec", VEC_FILE_PATH, "-bg", bg,
        "-numPos", str(num_pos), "-numNeg", str(num_neg),
        "-numStages", str(stages),
        "-minHitRate", str(TRAINING_PARAMS["min_hit_rate"]),
        "-maxFalseAlarmRate", str(TRAINING_PARAMS["max_false_alarm_rate"]),
        "-featureType", TRAINING_PARAMS["feature_type"],
        "-w", str(TRAINING_PARAMS["width"]), "-h", str(TRAINING_PARAMS["height"]),
        "-precalcValBufSize", str(TRAINING_PARAMS["precalcValBufSize"]),
        "-precalcIdxBufSize", str(TRAINING_PARAMS["precalcIdxBufSize"]),
        "-mode", TRAINING_PARAMS["mode"],
    ]
    res = subprocess.run(cmd, cwd=BASE_DIR,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    logger.info(res.stdout)
    if res.returncode or "ERROR" in res.stderr.upper():
        logger.error(res.stderr)
        raise RuntimeError("traincascade falhou")
    logger.info("✔ Treinamento concluído — cascade.xml e stages em /model")

# ------------------------------------------------------------------ #
def run_trainer():
    info, total_lines = build_positives_txt()
    logger.info(f"Anotações válidas: {total_lines}")

    vec_real = create_vec(info, total_lines)   # amostras geradas
    logger.info(f"Amostras reais no .vec: {vec_real}")

    # opcional: checar cabeçalho
    if vec_count(VEC_FILE_PATH) != vec_real:
        logger.warning("Vector header ≠ contagem capturada")

    train(vec_real)

# ------------------------------------------------------------------ #
if __name__ == "__main__":
    try:
        run_trainer()
    except Exception as exc:
        logger.error(f"⛔ Pipeline abortado: {exc}")
