# app/services/image_loader.py

import cv2
import os
from app.services.logger import get_logger

logger = get_logger(__name__)

def load_image(path: str, grayscale: bool = False):
    """
    Carrega uma imagem com OpenCV.
    :param path: Caminho completo da imagem
    :param grayscale: Se True, carrega em tons de cinza
    :return: Imagem carregada ou None se erro
    """
    if not os.path.exists(path):
        logger.warning(f"Imagem não encontrada: {path}")
        return None

    flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(path, flag)

    if image is None:
        logger.error(f"Erro ao carregar imagem: {path}")
    return image


def resize_image(image, width: int = None, height: int = None):
    """
    Redimensiona a imagem para a largura/altura especificada.
    :param image: Imagem original
    :param width: Nova largura (opcional)
    :param height: Nova altura (opcional)
    :return: Imagem redimensionada
    """
    if width is None and height is None:
        return image

    dimensions = (width, height)
    return cv2.resize(image, dimensions, interpolation=cv2.INTER_AREA)
