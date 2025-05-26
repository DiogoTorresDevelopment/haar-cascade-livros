import os, cv2, time
from datetime import datetime
from app.services.logger import get_logger
from app.config.settings import (
    CASCADE_XML_PATH,
    DETECTION_PARAMS,
    BASE_DIR,
)

logger = get_logger(__name__)

# pasta onde os frames detectados serão salvos
WEBCAM_DIR = os.path.join(BASE_DIR, "dataset", "results", "webcam")
os.makedirs(WEBCAM_DIR, exist_ok=True)


def main(max_saved=50, cam_index=0):
    if not os.path.exists(CASCADE_XML_PATH):
        logger.error("cascade.xml não encontrado.")
        return

    cascade = cv2.CascadeClassifier(CASCADE_XML_PATH)
    if cascade.empty():
        logger.error("Falha ao carregar cascade.xml.")
        return

    cap = cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        logger.error("Webcam não disponível.")
        return

    saved = 0
    logger.info("▶️ Pressione ESC para sair.")
    while saved < max_saved:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        rects = cascade.detectMultiScale(
            gray,
            scaleFactor=DETECTION_PARAMS["scaleFactor"],
            minNeighbors=DETECTION_PARAMS["minNeighbors"],
            minSize=DETECTION_PARAMS["minSize"],
            maxSize=DETECTION_PARAMS.get("maxSize"),
            flags=DETECTION_PARAMS["flags"],
        )

        # NMS
        rects = list(rects)
        if rects:
            rects, _ = cv2.groupRectangles(
                rects + rects,
                groupThreshold=DETECTION_PARAMS.get("groupThreshold", 1),
                eps=DETECTION_PARAMS.get("eps", 0.3),
            )
            rects = rects.tolist()

        # Limita quantidade de objetos por frame
        max_obj = DETECTION_PARAMS.get("max_objects")
        if max_obj and len(rects) > max_obj:
            rects = rects[:max_obj]

        # Desenha preview
        for (x, y, w, h) in rects:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Webcam – DETECTOR", frame)

        # Salva se houver pelo menos 1 objeto
        if rects:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
            out_name = f"det_{ts}.jpg"
            cv2.imwrite(os.path.join(WEBCAM_DIR, out_name), frame)
            logger.info(f"[{ts}] {len(rects)} objeto(s) – frame salvo")
            saved += 1

        # ESC = sair
        if cv2.waitKey(1) & 0xFF == 27:
            break

    logger.info(f"➡️ Loop encerrado. Frames salvos: {saved}")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(max_saved=50, cam_index=0)
