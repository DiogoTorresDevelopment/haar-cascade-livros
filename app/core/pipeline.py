# app/core/pipeline.py

from app.services.logger import get_logger
from app.core import annotator, trainer, detector

logger = get_logger(__name__)

def run_pipeline(etapas=("annotate", "train", "detect")):
    """
    Executa todas as etapas do pipeline, conforme as opções informadas.
    :param etapas: tupla com as etapas desejadas (annotate, train, detect)
    """
    logger.info("Iniciando pipeline Haar Cascade...")

    if "annotate" in etapas:
        logger.info("Etapa 1: Anotação manual das imagens.")
        annotator.annotate_images()

    if "train" in etapas:
        logger.info("Etapa 2: Treinamento do classificador Haar.")
        trainer.run_trainer()

    if "detect" in etapas:
        logger.info("Etapa 3: Detecção com o classificador treinado.")
        detector.run_detection()

    logger.info("Pipeline finalizado com sucesso.")


if __name__ == "__main__":
    # Altere aqui se quiser rodar apenas partes do pipeline
    run_pipeline(etapas=("train", "detect"))
