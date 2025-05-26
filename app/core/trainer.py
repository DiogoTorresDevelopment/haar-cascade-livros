# app/core/trainer.py
import os, re, struct, subprocess, math
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

# ───────────────────────────────────────────────────────── #
def build_positives_txt() -> Tuple[str, int]:
    """Gera positives.txt e valida todas as caixas anotadas."""
    info = os.path.join(ANNOTATIONS_PATH, "positives.txt")
    lines = []

    for ann in os.listdir(ANNOTATIONS_PATH):
        if ann == "positives.txt" or not ann.endswith(".txt"):
            continue

        base = ann[:-4]
        img_abs = next(
            (os.path.join(POSITIVE_PATH, f)
             for f in os.listdir(POSITIVE_PATH)
             if os.path.splitext(f)[0] == base
             and os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png")),
            None
        )
        if not img_abs:
            logger.warning(f"Ignorada '{ann}': imagem não encontrada")
            continue

        ann_vals = open(os.path.join(ANNOTATIONS_PATH, ann),
                        encoding="utf-8").read().strip().split()
        if len(ann_vals) < 5 or (len(ann_vals) - 1) % 4:
            logger.warning(f"Ignorada '{ann}': formato inválido")
            continue

        img = cv2.imread(img_abs)
        if img is None:
            logger.warning(f"Ignorada '{ann}': imagem corrompida")
            continue
        ih, iw = img.shape[:2]

        ok = True
        for i in range(1, len(ann_vals), 4):
            x, y, w, h = map(int, ann_vals[i:i+4])
            if w <= 0 or h <= 0 or x < 0 or y < 0 or x+w > iw or y+h > ih:
                ok = False
                break
        if not ok:
            logger.warning(f"Ignorada '{ann}': bbox fora dos limites")
            continue

        rel = os.path.relpath(img_abs, ANNOTATIONS_PATH).replace("\\", "/")
        lines.append(f"{rel} {' '.join(ann_vals)}")

    if not lines:
        raise RuntimeError("Nenhuma anotação válida em /annotations")

    with open(info, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return info, len(lines)

# ───────────────────────────────────────────────────────── #
def create_vec(info_path: str, n_lines: int) -> int:
    """
    Gera .vec; aumenta fator (aug) automaticamente até ter
    amostras ≥ 3×numStages·1.1 (10 % de folga)
    """
    stages = TRAINING_PARAMS["num_stages"]
    min_required = math.ceil(3 * stages * 1.1)

    # fator inicial
    aug = 8.0
    while True:
        vec_target = max(int(n_lines * aug), n_lines + 200)
        cmd = [
            CREATESAMPLES_EXE,
            "-info", os.path.basename(info_path),
            "-num",  str(vec_target),
            "-w",    str(TRAINING_PARAMS["width"]),
            "-h",    str(TRAINING_PARAMS["height"]),
            "-vec",  VEC_FILE_PATH,
        ]
        res = subprocess.run(cmd, cwd=ANNOTATIONS_PATH,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode:
            logger.error(res.stderr)
            raise RuntimeError("createsamples falhou")

        m = re.search(r"Created\s+(\d+)\s+samples", res.stdout)
        vec_real = int(m.group(1)) if m else vec_target
        logger.info(f"Created {vec_real} samples (aug={aug:.1f})")

        if vec_real >= min_required:
            return vec_real

        aug *= 1.5        # aumenta e tenta novamente
        logger.warning("Poucas amostras. Aumentando aug e recriando .vec…")

# ───────────────────────────────────────────────────────── #
def vec_count(path: str) -> int:
    with open(path, "rb") as f:
        return struct.unpack("<I", f.read(4))[0]

# ───────────────────────────────────────────────────────── #
def ensure_bg_txt() -> str:
    bg = os.path.join(NEGATIVE_PATH, "bg.txt")
    if os.path.exists(bg):
        return bg
    rels = [os.path.relpath(p, BASE_DIR).replace("\\", "/")
            for p in list_images(NEGATIVE_PATH, (".jpg", ".jpeg", ".png"))]
    with open(bg, "w", encoding="utf-8") as f:
        f.write("\n".join(rels))
    logger.info(f"bg.txt criado com {len(rels)} negativos")
    return bg

# ───────────────────────────────────────────────────────── #
def train(vec_real: int):
    stages = TRAINING_PARAMS["num_stages"]
    min_pos = 3 * stages
    if vec_real < min_pos:
        raise RuntimeError(
            f"Apenas {vec_real} amostras positivas – precisam ≥ {min_pos} para {stages} estágios."
        )

    num_pos = max(min_pos, int(vec_real * 0.8))
    num_neg = min(
        sum(1 for _ in open(ensure_bg_txt()) if _.strip()),
        TRAINING_PARAMS["num_negative"]
    )

    logger.info(f"→ Treinando (stages={stages}, numPos={num_pos}, numNeg={num_neg})")

    cmd = [
        TRAINCASCADE_EXE,
        "-data", MODEL_DIR,
        "-vec",  VEC_FILE_PATH,
        "-bg",   ensure_bg_txt(),
        "-numPos", str(num_pos),
        "-numNeg", str(num_neg),
        "-numStages", str(stages),
        "-featureType", TRAINING_PARAMS["feature_type"],
        "-minHitRate",  str(TRAINING_PARAMS["min_hit_rate"]),
        "-maxFalseAlarmRate", str(TRAINING_PARAMS["max_false_alarm_rate"]),
        "-w", str(TRAINING_PARAMS["width"]),
        "-h", str(TRAINING_PARAMS["height"]),
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
    logger.info("✔ Treinamento finalizado — cascade.xml salvo em /model")

# ───────────────────────────────────────────────────────── #
def run_trainer():
    info, lines = build_positives_txt()
    logger.info(f"Positivos válidos: {lines}")

    vec_real = create_vec(info, lines)
    logger.info(f"Amostras geradas: {vec_real}")

    if vec_count(VEC_FILE_PATH) != vec_real:
        logger.warning("Inconsistência no cabeçalho do .vec (ok para LBP)")

    train(vec_real)

# ───────────────────────────────────────────────────────── #
if __name__ == "__main__":
    try:
        run_trainer()
    except Exception as err:
        logger.error(f"⛔ Pipeline abortado: {err}")
