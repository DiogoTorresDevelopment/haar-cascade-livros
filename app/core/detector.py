# app/core/detector.py

import os
import cv2
from app.services.logger import get_logger
from app.utils.file_utils import list_images, ensure_dir
from app.config.settings import (
    CASCADE_XML_PATH,
    RESULTS_DETECTED_PATH,
    RESULTS_NOT_DETECTED_PATH,
    DETECTION_PARAMS,
    BASE_DIR,
)

logger = get_logger(__name__)


def run_detection():
    """Aplica o classificador Haar treinado em imagens da pasta test_images."""
    if not os.path.exists(CASCADE_XML_PATH):
        logger.error(f"Modelo Haar não encontrado: {CASCADE_XML_PATH}")
        return

    cascade = cv2.CascadeClassifier(CASCADE_XML_PATH)
    if cascade.empty():
        logger.error("Falha ao carregar cascade.xml.")
        return

    ensure_dir(RESULTS_DETECTED_PATH)
    ensure_dir(RESULTS_NOT_DETECTED_PATH)

    test_dir = os.path.join(BASE_DIR, "dataset", "test_images")
    images = list_images(test_dir)

    if not images:
        logger.warning("Nenhuma imagem em dataset/test_images.")
        return

    logger.info(f"Detectando em {len(images)} imagens…")

    for img_path in images:
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Falha ao abrir: {filename}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        rects = cascade.detectMultiScale(
            gray,
            scaleFactor=DETECTION_PARAMS["scaleFactor"],
            minNeighbors=DETECTION_PARAMS["minNeighbors"],
            minSize=DETECTION_PARAMS["minSize"],
            maxSize=DETECTION_PARAMS.get("maxSize"),
            flags=DETECTION_PARAMS["flags"],
        )

        # ---------- Non-Maximum Suppression -----------------
        rects = list(rects)                         # converte para lista nativa
        if len(rects) > 0:
            rects, _ = cv2.groupRectangles(
                rects + rects,                     # duplica lista p/ agrupar
                groupThreshold=DETECTION_PARAMS.get("groupThreshold", 1),
                eps=DETECTION_PARAMS.get("eps", 0.3),
            )
            rects = rects.tolist()                 # volta a ser lista de tuplas
        # ----------------------------------------------------

        # Limite opcional de objetos por imagem
        max_obj = DETECTION_PARAMS.get("max_objects")
        if max_obj and len(rects) > max_obj:
            rects = rects[:max_obj]

        if len(rects) > 0:
            for (x, y, w, h) in rects:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            out_path = os.path.join(RESULTS_DETECTED_PATH, filename)
            logger.info(f"{filename}: {len(rects)} livro(s) detectado(s).")
        else:
            out_path = os.path.join(RESULTS_NOT_DETECTED_PATH, filename)
            logger.info(f"{filename}: nenhum livro.")

        cv2.imwrite(out_path, img)

    logger.info("✅ Detecção concluída.")


if __name__ == "__main__":
    run_detection()
