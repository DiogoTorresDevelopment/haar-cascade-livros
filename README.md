# 📘 Projeto: Haar Cascade – Detector de Livros

## 🎯 Objetivo

Este projeto tem como objetivo treinar um **classificador Haar Cascade personalizado** utilizando Python e OpenCV para detectar **livros** em imagens. O modelo foi construído do zero, com organização modular seguindo os princípios de **Clean Code** e uma estrutura inspirada no Spring Framework.

O trabalho foi desenvolvido em dupla para a disciplina de Visão Computacional, aplicando conceitos práticos como anotação de imagens, geração de vetores de amostras e treinamento supervisionado com Haar Cascade.

---

## 🧠 Etapas Realizadas

1. **Coleta de imagens positivas e negativas**
2. **Anotação manual das imagens positivas**
3. **Geração de amostras vetorizadas (.vec)**
4. **Treinamento do classificador com opencv_traincascade**
5. **Teste e validação em novas imagens**
6. **Organização do projeto com estrutura modular**
7. **Publicação e documentação no GitHub**

---

## 🗂️ Estrutura de Pastas

```
haar-cascade-livros/
├── app/
│   ├── core/
│   │   ├── annotator.py
│   │   ├── detector.py
│   │   ├── pipeline.py
│   │   └── trainer.py
│   ├── services/
│   │   ├── image_loader.py
│   │   └── logger.py
│   ├── config/
│   │   └── settings.py
│   └── utils/
│       └── file_utils.py
├── dataset/
│   ├── positives/
│   ├── negatives/
│   ├── annotations/
│   └── results/
│       ├── detected/
│       └── not_detected/
├── model/
│   ├── cascade.xml
│   └── trained_vec.vec
├── scripts/
│   ├── detect_custom.py
│   └── create_samples.sh
├── auto_pipeline.py
├── requirements.txt
└── README.md
```

---

## ▶️ Como Executar

### 1. Instale as dependências:

```bash
pip install -r requirements.txt
```

> ⚠️ O OpenCV 3.4.11 deve estar instalado via `.exe` (não via pip).

### 2. Anotar imagens manualmente (opcional)

```bash
python -m app.core.annotator
```

> Clique e arraste para marcar livros nas imagens. Pressione `S` para salvar, `R` para resetar.

### 3. Rodar pipeline completo (treinamento + detecção)

```bash
python auto_pipeline.py
```

Ou, se quiser apenas as etapas específicas:

```python
from app.core.pipeline import run_pipeline
run_pipeline(etapas=("train", "detect"))
```

### 4. Testar detecção em imagem customizada

```bash
python scripts/detect_custom.py
```

---

## 📦 Detalhes Técnicos

- Linguagem: **Python 3.13**
- Biblioteca principal: **OpenCV 3.4.11**
- Detecção Haar com parâmetros ajustáveis (scaleFactor, minNeighbors etc.)
- Interface de anotação manual com mouse via OpenCV GUI
- Logs estruturados para rastreabilidade de erros e progresso
- Estrutura modular (config, services, core, utils)

---

## 💡 Resultados Esperados

- Detecção precisa de livros em imagens com fundos variados
- Classificador Haar `.xml` gerado pode ser reutilizado
- Organização do projeto facilita manutenções e testes