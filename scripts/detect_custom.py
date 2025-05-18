# scripts/detect_custom.py

import os
import cv2
from app.config.settings import CASCADE_XML_PATH
from app.services.logger import get_logger

logger = get_logger(__name__)

def detect_from_image(image_path):
    if not os.path.exists(CASCADE_XML_PATH):
        logger.error(f"Modelo Haar não encontrado: {CASCADE_XML_PATH}")
        return

    cascade = cv2.CascadeClassifier(CASCADE_XML_PATH)
    if cascade.empty():
        logger.error("Erro ao carregar o modelo Haar.")
        return

    image = cv2.imread(image_path)
    if image is None:
        logger.error(f"Erro ao carregar imagem: {image_path}")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detections = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in detections:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow("Detecção personalizada", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    img_path = input("Digite o caminho da imagem para teste: ")
    detect_from_image(img_path.strip())
