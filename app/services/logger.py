import logging
import sys

def get_logger(name: str = "haar-cascade") -> logging.Logger:
    """
    Retorna um logger configurado para console.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Captura tudo a partir de DEBUG

    # Garante que não acumule múltiplos handlers em execuções repetidas
    if logger.hasHandlers():
        logger.handlers.clear()

    # Formato das mensagens
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logger.addHandler(console_handler)

    return logger
