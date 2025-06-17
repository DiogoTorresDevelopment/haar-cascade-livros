# app/config/settings.py

import os

# Caminho absoluto para a raiz do projeto
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

# Caminhos para datasets
POSITIVE_PATH = os.path.join(BASE_DIR, "dataset", "positives")
NEGATIVE_PATH = os.path.join(BASE_DIR, "dataset", "negatives")
ANNOTATIONS_PATH = os.path.join(BASE_DIR, "dataset", "annotations")

# Caminhos para resultados da detecção
RESULTS_DETECTED_PATH = os.path.join(BASE_DIR, "dataset", "results", "detected")
RESULTS_NOT_DETECTED_PATH = os.path.join(BASE_DIR, "dataset", "results", "not_detected")

# Caminhos para arquivos do modelo treinado
MODEL_DIR = os.path.join(BASE_DIR, "model")
CASCADE_XML_PATH = os.path.join(MODEL_DIR, "cascade.xml")
VEC_FILE_PATH = os.path.join(MODEL_DIR, "trained_vec.vec")


# Parâmetros de treinamento do classificador Haar Cascade
TRAINING_PARAMS = {
    "num_stages": 15,              # pode manter
    "num_negative": 400,           # se tiver base suficiente
    "num_positive": 200,           # defina conforme a geração do .vec
    "min_hit_rate": 5.990,         # um pouco menos exigente
    "max_false_alarm_rate": 0.4,   # mantém estável
    "feature_type": "HAAR",        # melhor para objetos com arestas
    "width": 80,                   # aumenta riqueza de features
    "height": 80,                  # idem
    "precalcValBufSize": 4096,
    "precalcIdxBufSize": 4096,
    "mode": "ALL"
}


# Parâmetros para o processo de detecção
DETECTION_PARAMS = {
    "scaleFactor": 1.06,      # passo entre escalas
    "minNeighbors": 3,       # exige mais retângulos vizinhos
    "minSize": (80, 100),     # ≥ janela de treino
    "maxSize": (800, 1200),   # opcional, pode omitir
    "flags": 0,
    "groupThreshold": 1,     # NMS: grupos com ≥2 retângulos
    "eps": 0.2,              # NMS: sobreposição máxima
    "max_objects": 2      # queremos máx. 5 livros
}


# Caminho para os binários do OpenCV
OPENCV_BIN_DIR = r"C:\opencv\build\x64\vc15\bin"
CREATESAMPLES_EXE = os.path.join(OPENCV_BIN_DIR, "opencv_createsamples.exe")
TRAINCASCADE_EXE = os.path.join(OPENCV_BIN_DIR, "opencv_traincascade.exe")
