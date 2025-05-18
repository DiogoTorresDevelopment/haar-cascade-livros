import cv2
import os
from app.utils.file_utils import list_images, ensure_dir
from app.services.logger import get_logger
from app.config.settings import POSITIVE_PATH, ANNOTATIONS_PATH

logger = get_logger(__name__)


def annotate_images():
    """
    Permite anotar regiões com livros nas imagens da pasta POSITIVE_PATH.
    Salva anotações no formato x y w h por imagem.
    """
    ensure_dir(ANNOTATIONS_PATH)
    images = list_images(POSITIVE_PATH)

    if not images:
        logger.warning("Nenhuma imagem encontrada em 'positives'.")
        return

    logger.info(f"Iniciando anotação de {len(images)} imagens...")

    for image_path in images:
        filename = os.path.basename(image_path)
        annotation_path = os.path.join(ANNOTATIONS_PATH, f"{os.path.splitext(filename)[0]}.txt")

        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Falha ao carregar imagem: {image_path}")
            continue

        clone = image.copy()
        rects = []

        logger.info(f"Anotando: {filename}")
        cv2.namedWindow("Anotar - pressione [S] para salvar", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Anotar - pressione [S] para salvar", 800, 600)

        coords = {"start": None, "end": None, "drawing": False}

        def draw_rectangle(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                coords["drawing"] = True
                coords["start"] = (x, y)

            elif event == cv2.EVENT_MOUSEMOVE and coords["drawing"]:
                temp_img = clone.copy()
                cv2.rectangle(temp_img, coords["start"], (x, y), (0, 255, 0), 2)
                cv2.imshow("Anotar - pressione [S] para salvar", temp_img)

            elif event == cv2.EVENT_LBUTTONUP:
                coords["drawing"] = False
                coords["end"] = (x, y)
                cv2.rectangle(clone, coords["start"], coords["end"], (0, 255, 0), 2)
                rects.append((coords["start"], coords["end"]))
                cv2.imshow("Anotar - pressione [S] para salvar", clone)

        cv2.setMouseCallback("Anotar - pressione [S] para salvar", draw_rectangle)
        cv2.imshow("Anotar - pressione [S] para salvar", clone)

        while True:
            key = cv2.waitKey(100) & 0xFF
            if key == ord("s"):
                break
            elif key == ord("r"):
                clone = image.copy()
                rects.clear()
                cv2.imshow("Anotar - pressione [S] para salvar", clone)

        cv2.destroyAllWindows()

        if rects:
            # Salva apenas: <num_obj> x y w h ...
            annotation_line = f"{len(rects)}"
            for (start, end) in rects:
                x = min(start[0], end[0])
                y = min(start[1], end[1])
                w = abs(end[0] - start[0])
                h = abs(end[1] - start[1])
                annotation_line += f" {x} {y} {w} {h}"

            with open(annotation_path, "w") as f:
                f.write(annotation_line + "\n")

            logger.info(f"Anotação salva: {annotation_path}")
        else:
            logger.warning(f"Sem retângulos anotados para: {filename}")

    logger.info("✅ Processo de anotação finalizado.")


if __name__ == "__main__":
    annotate_images()
