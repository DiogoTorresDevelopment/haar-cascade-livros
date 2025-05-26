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
    "num_stages": 15,          # novo alvo
    "num_negative": 300,      # 2× numPos recomendável
    "min_hit_rate": 0.995,     # menor que 0.999 p/ não travar
    "max_false_alarm_rate": 0.4,
    "feature_type": "HAAR",     # LBP treina ~30 % mais rápido
    "width": 50,
    "height": 50,
    "precalcValBufSize": 4096, # MB
    "precalcIdxBufSize": 4096,
    "mode": "ALL"
}

# Parâmetros para o processo de detecção
DETECTION_PARAMS = {
    "scaleFactor": 1.8,      # passo entre escalas
    "minNeighbors": 10,       # exige mais retângulos vizinhos
    "minSize": (160, 40),     # ≥ janela de treino
    "maxSize": (1000, 400),   # opcional, pode omitir
    "flags": 0,
    "groupThreshold": 1,     # NMS: grupos com ≥2 retângulos
    "eps": 0.3,              # NMS: sobreposição máxima
    "max_objects": 5         # queremos máx. 5 livros
}


# Caminho para os binários do OpenCV
OPENCV_BIN_DIR = r"C:\opencv\build\x64\vc15\bin"
CREATESAMPLES_EXE = os.path.join(OPENCV_BIN_DIR, "opencv_createsamples.exe")
TRAINCASCADE_EXE = os.path.join(OPENCV_BIN_DIR, "opencv_traincascade.exe")
