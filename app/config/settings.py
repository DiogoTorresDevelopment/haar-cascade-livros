# app/config/settings.py

import os

# Caminho absoluto para a raiz do projeto
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))


# Caminhos para datasets
POSITIVE_PATH = os.path.join(BASE_DIR, "dataset", "positives")
NEGATIVE_PATH = os.path.join(BASE_DIR, "dataset", "negatives")
ANNOTATIONS_PATH = os.path.join(BASE_DIR, "dataset", "annotations")

# Caminhos para resultados dos testes
RESULTS_DETECTED_PATH = os.path.join(BASE_DIR, "dataset", "results", "detected")
RESULTS_NOT_DETECTED_PATH = os.path.join(BASE_DIR, "dataset", "results", "not_detected")

# Caminhos para arquivos do modelo treinado
MODEL_DIR = os.path.join(BASE_DIR, "model")
CASCADE_XML_PATH = os.path.join(MODEL_DIR, "cascade.xml")
VEC_FILE_PATH = os.path.join(MODEL_DIR, "trained_vec.vec")

# Parâmetros para o treinamento do classificador Haar Cascade
TRAINING_PARAMS = {
    "num_positive": 200,              # Aumente para pelo menos 200 (quanto mais, melhor)
    "num_negative": 400,              # Negativas devem ser no mínimo o dobro das positivas
    "num_stages": 5,                  # 5 a 10 estágios já são bons para teste
    "min_hit_rate": 0.95,             # Reduza para 0.95 no início. 0.999 é muito restritivo!
    "max_false_alarm_rate": 0.5,      # Deixe como está. 0.5 é um bom valor.
    "feature_type": "HAAR",           # Pode ser "HAAR" ou "LBP"
    "width": 50,                      # Largura da janela de detecção (baseada no tamanho dos objetos)
    "height": 50                      # Altura da janela (ex: se o livro geralmente aparece com 50x50 px)
}


# Parâmetros para o processo de detecção em novas imagens
DETECTION_PARAMS = {
    "scaleFactor": 1.05,
    "minNeighbors": 3,
    "minSize": (24, 24),
    "flags": 0
}

# Caminho para os binários do OpenCV instalados manualmente (.exe)
OPENCV_BIN_DIR = r"C:\opencv\build\x64\vc15\bin"  # ajuste se estiver em outro lugar
CREATESAMPLES_EXE = os.path.join(OPENCV_BIN_DIR, "opencv_createsamples.exe")
TRAINCASCADE_EXE = os.path.join(OPENCV_BIN_DIR, "opencv_traincascade.exe")
