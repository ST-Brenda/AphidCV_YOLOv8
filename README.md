# AphidCV 3.0
# Versão configurada para comparação com YOLOv8n
# Artigo: A comparative study between deep learning approaches for aphid classification
# Autores: Brenda S Taca, Douglas Lau, Rafael Rieder
# Data: 29 de novembro de 2024
***

## Dependências:

- Python 3.10.12
- TensorFlow 2.8.0
- Albumentations 1.4.21
- OpenCV 4.10

***

## Configurações já definidas em código:

**Treinamento e validação dos modelos:**

- Learning rate: 0.001 (padrão do otimizador Adam)
- Image size: (120, 120)
- Epochs: 150
- Batchsize: 100
- Patience: 10

**Métricas coletadas:**

- accuracy
- precision
- recall
- roc
- prc

**Aumentações aplicadas:**

- RandomRotate90(p=0.5)
- Blur(p=0.25)
- RandomBrightnessContrast(p=0.25)
- Sharpen(p=0.25)
- Emboss(p=0.25)
- ening(p=0.25)
- osing(p=0.25)
- CLAHE(p=0.25)
- Affine(shear=([-45, 45]), scale=(0.5, 1.5), p=0.2)
- Flip(p=0.5)

***

## Passos para execução:

**1.** Considerar o arquivo "PAPER_AphidCV_Albu_Color2kBal_BS100.py", que já possui as configurações supracitadas definidas em código.

**2.** Para gerar um modelo para cada espécie de afídeo, é preciso atualizar o caminho onde estão as imagens. No código, está a marcação "CHANGE HERE BEFORE EACH TRAINING" nos locais onde é preciso alterar o caminho para cada subconjunto do dataset, bem como os acrônimos para definir a pasta e os nomes das documentações de saída.

**3.** Após realizar esses ajustes, basta rodar o script.

**4.** Ao final da execução, o tempo de processamento é exibido, e as seguintes saídas em arquivo são geradas: gráfico PNG da arquitetura do modelo, gráficos PNG das curvas de aprendizado (loss, accuracy, precision, recall, roc, prc), histórico em formato CSV, e modelos em formato H5.

**5.** Para calcular a métrica de F1-Score, considerar as medidas obtidas de precision e recall.


***


# YOLOv8m
# Versão configurada para comparação com AphidCV 3.0
# Artigo: A comparative study between deep learning approaches for aphid classification
# Autores: Brenda S Taca, Douglas Lau, Rafael Rieder
# Data: 29 de novembro de 2024

***

## Dependências:

- Python 3.10.12
- Ultralytics 8.1.45
- Albumentations 1.4.21
- OpenCV 4.10

***

## Configurações definidas em linha de comando:

- Image size: (120, 120)
- Epochs: 150
- Batchsize: 100
- Patience: 10

***

## Configurações já definidas em código:

**Treinamento e validação dos modelos:**
- Learning rate: 0.01 (padrão YOLOv8)

**Métricas coletadas:**
- confusion matrix
- precision (P)
- recall (R)
- mAP50
- mAP50-95
  
***

## Alterações a fazer no código original:

**Adicionar aumentações (arquivo "augment.py")**
- RandomRotate90(p=0.5)
- Blur(p=0.25)
- RandomBrightnessContrast(p=0.25)
- Sharpen(p=0.25)
- Emboss(p=0.25)
- ening(p=0.25)
- osing(p=0.25)
- CLAHE(p=0.25)
- Affine(shear=([-45, 45]), scale=(0.5, 1.5), p=0.2)
- Flip(p=0.5)
  
***

## Passos para execução:

**1.** Clonar localmente a YOLO:
```bash
git clone https://github.com/ultralytics/ultralytics.git -b v8.1.45
```
**3.** Adicionar no arquivo "ultralytics/ultralytics/data/augment.py" as aumentações necessárias para o processo comparativo: class Albumentations, variável T (# Transforms).
Para tanto, você pode:

- Substituir o arquivo original clonado pelo "augment.py" disponível nesse repositório, ou;
- Copiar o bloco de código entre as linhas 778-848 do "augment.py" disponível nesse repositório para o arquivo original clonado, sobreescrevendo o bloco entre as linhas 863-891.

**3.** Para gerar um modelo para cada espécie de afídeo, é preciso atualizar o caminho onde está o arquivo de configuração YAML. Na linha de comando, substituir a atribuição data="PASTA-DO-DATASET/ARQUIVO.yaml" pelo caminho correspondente.

**4.** Após realizar esses ajustes, basta rodar a linha de comando a partir da pasta principal "ultralytics/":

```bash
yolo task=detect mode=train model=yolov8m.pt imgsz=120 data="PASTA-DO-DATASET/ARQUIVO.yaml" epochs=150 batch=100 workers=20 device=0 val=True keras=True patience=10 augment=true
```

**5.** Ao final da execução, a YOLO gera um sumário com várias informações de saída, entre elas: 
- Tempo de processamento e as métricas precision, recall, mAP50 e mAP50-95.

Também gera gráficos PNG: 
- Matriz de confusão - versões padrão e normalizada;
- Curvas precision-recall, precision-confidence, curva recall-confidence e curva F1-confidence. 

Além disso, salva os modelos (best e last) em formato PT. 

Considerar somente o "best.pt".

**6.** Para calcular a métrica de accuracy, considerar os dados obtidos para a matriz de confus˜ao.

**7.** Para calcular a métrica de F1-Score, considerar as medidas obtidas de Precision e Recall.
