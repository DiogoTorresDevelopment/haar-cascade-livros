import os
import cv2
from app.services.logger import get_logger
from app.utils.file_utils import list_images, ensure_dir
from app.config.settings import (
    CASCADE_XML_PATH,
    RESULTS_DETECTED_PATH,
    RESULTS_NOT_DETECTED_PATH,
    DETECTION_PARAMS,
    BASE_DIR
)

logger = get_logger(__name__)


def run_detection():
    """
    Aplica o classificador Haar treinado em imagens novas
    e salva os resultados nas respectivas pastas.
    """
    if not os.path.exists(CASCADE_XML_PATH):
        logger.error(f"Modelo Haar Cascade não encontrado em {CASCADE_XML_PATH}")
        return

    cascade = cv2.CascadeClassifier(CASCADE_XML_PATH)
    if cascade.empty():
        logger.error("Erro ao carregar o cascade.xml.")
        return

    ensure_dir(RESULTS_DETECTED_PATH)
    ensure_dir(RESULTS_NOT_DETECTED_PATH)

    # Usa a pasta dataset/test_images como entrada
    test_images_path = os.path.join(BASE_DIR, "dataset", "test_images")
    images = list_images(test_images_path)

    if not images:
        logger.warning("Nenhuma imagem encontrada para detecção.")
        return

    logger.info(f"Executando detecção em {len(images)} imagens...")

    for image_path in images:
        filename = os.path.basename(image_path)
        image = cv2.imread(image_path)

        if image is None:
            logger.warning(f"Erro ao carregar imagem: {filename}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        detections = cascade.detectMultiScale(
            gray,
            scaleFactor=DETECTION_PARAMS["scaleFactor"],
            minNeighbors=DETECTION_PARAMS["minNeighbors"],
            minSize=DETECTION_PARAMS["minSize"],
            flags=DETECTION_PARAMS["flags"]
        )

        if len(detections) > 0:
            for (x, y, w, h) in detections:
                cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            output_path = os.path.join(RESULTS_DETECTED_PATH, filename)
            logger.info(f"Detectado: {filename} ({len(detections)} objetos)")
        else:
            output_path = os.path.join(RESULTS_NOT_DETECTED_PATH, filename)
            logger.info(f"Nenhum objeto detectado: {filename}")

        cv2.imwrite(output_path, image)

    logger.info("Processo de detecção concluído.")

if __name__ == "__main__":
    run_detection()
