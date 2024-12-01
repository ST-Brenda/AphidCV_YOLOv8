# A comparative study between deep learning approaches for aphid classification

This study presents a performance comparison between two convolutional neural networks in the task of detecting aphids in digital images: AphidCV, customized for counting, classifying, and measuring aphids, and YOLOv8, state-of-the-art in real-time object detection. Our work considered 48,000 images for training for six different insect species (8,000 images divided into four classes), in addition to data augmentation techniques. For comparative purposes, we considered evaluation metrics available to both architectures (Accuracy, Precision, Recall, and F1-Score) and additional metrics (ROC Curve and PR AUC for AphidCV; mAP@50 and mAP@50-95 for YOLOv8). The results revealed an average F1-Score = 0.891 for the AphidCV architecture, version 3.0, and an average F1-Score = 0.882 for the YOLOv8, medium version, demonstrating the effectiveness of both architectures for training aphid detection models. Overall, AphidCV performed slightly better for the majority of metrics and species in the study, serving its design purpose very well. YOLOv8 proved to be faster to converge the models, with the potential to apply in research considering many aphid species.

To reproduce the results obtained in our paper, considering training and validation of the AphidCV models, please follow the instructions in [AphidCV 3.0](#aphidcv-30) section.

And to reproduce the results considering training and validation of the YOLO models, please follow the instructions in [YOLOv8m](#yolov8m) section.

For more info, please contact us or read our paper published in the [IEEE Latin America Transactions](https://latamt.ieeer9.org/index.php/transactions). 

## Authors:
1. Brenda Slongo Taca
2. Douglas Lau
3. Rafael Rieder

## Shipment ID
9209

***
## AphidCV 3.0
Version configured for comparison with YOLOv8m
***

### Requirements:

- Python 3.10.12
- TensorFlow 2.8.0
- Albumentations 1.4.21
- OpenCV 4.10

***

### Settings already defined in source code:

**Models' training and validation:**

- Learning rate: 0.001 (default value - Adam optimizer)
- Image size: (120, 120)
- Epochs: 150
- Batchsize: 100
- Patience: 10

**Metrics:**

- accuracy
- precision
- recall
- roc
- prc

**Augmentations:**

- RandomRotate90(p=0.5)
- Blur(p=0.25)
- RandomBrightnessContrast(p=0.25)
- Sharpen(p=0.25)
- Emboss(p=0.25)
- Opening(p=0.25)
- Closing(p=0.25)
- CLAHE(p=0.25)
- Affine(shear=([-45, 45]), scale=(0.5, 1.5), p=0.2)
- Flip(p=0.5)

***

### How to Run:

**1.** Please, consider the "PAPER_AphidCV_Albu_Color2kBal_BS100.py" file, once already has the above settings defined in source code.

**2.** To generate a model for each aphid species, you need to update the path where the images are located. In the source code, there is a "CHANGE HERE BEFORE EACH TRAINING" markup in the places where you need to change the path for each subset of the dataset, as well as the acronyms to define the folder and the names of the output documentation. For this training, consider the "Datasets_Config_AphidCV.zip" dataset, structured in subdirectories according to the AphidCV/TensorFlow specs.

**3.** After proceed these adjustments, simply run the Python script.

**4.** At the end of each run, the processing time is displayed, and the following file outputs are generated: PNG graph of the model architecture, PNG graphs of the learning curves (loss, accuracy, precision, recall, roc, prc), history in CSV format, and model in H5 format.

**5.** To calculate the F1-Score, please consider the obtained precision and recall measures.


***

## YOLOv8m
Version configured for comparison with AphidCV 3.0

***

### Requirements:

- Python 3.10.12
- Ultralytics 8.1.45
- Albumentations 1.4.21
- OpenCV 4.10

***

### Settings defined on the command line:

- Image size: (120, 120)
- Epochs: 150
- Batchsize: 100
- Patience: 10

***

### Settings already defined in source code:

**Models' training and validation:**
- Learning rate: 0.01 (default lr value in YOLO)

**Metrics:**
- confusion matrix
- precision (P)
- recall (R)
- mAP50
- mAP50-95
  
***

### Changes to be made to the original code:

**Augmentations (add in the "augment.py")**
- RandomRotate90(p=0.5)
- Blur(p=0.25)
- RandomBrightnessContrast(p=0.25)
- Sharpen(p=0.25)
- Emboss(p=0.25)
- Opening(p=0.25)
- Closing(p=0.25)
- CLAHE(p=0.25)
- Affine(shear=([-45, 45]), scale=(0.5, 1.5), p=0.2)
- Flip(p=0.5)
  
***

### How to Run:

**1.** Please, clone the YOLO locally:
```bash
git clone https://github.com/ultralytics/ultralytics.git -b v8.1.45
```
**2.** Add the required augmentations for the comparative process to the "ultralytics/ultralytics/data/augment.py" file: class Albumentations, variable T (# Transforms). To do this, you can:

- Replace the original cloned file with the "augment.py" available in this repository, or;
- Copy the code block between lines 778-848 of the "augment.py" available in this repository to the original cloned file, overwriting the block between lines 863-891.

**3.** To generate a model for each aphid species, you need to update the path where the YAML configuration file is located. On the command line, replace the data="DATASET-FOLDER/FILE.yaml" assignment with the corresponding path. For this training, consider the "Datasets_Config_YOLO.zip" dataset, structured in subdirectories according to the YOLO specs.

**4.** After proceed these adjustments, copy the file "yolov8m.pt" to the main "ultralytics/" folder. Then, simply run the command line from the main "ultralytics/" folder:

```bash
yolo task=detect mode=train model=yolov8m.pt imgsz=120 data="Brevicoryne_brassicae.yaml" epochs=150 batch=100 workers=20 device=0 val=True keras=True patience=10 augment=True
```

**5.** At the end of each run, YOLO generates a summary with several output information, including: processing time and the precision, recall, mAP50 and mAP50-95 metrics. It also generates PNG graphs: confusion matrix - standard and normalized versions, and precision-recall, precision-confidence, recall-confidence curves and F1-confidence curves.

In addition, it saves the models (best and last) in PT format. For the study, consider only the "best.pt" files of each species.

**6.** To calculate the Accuracy, please consider the data obtained for the confusion matrix.

**7.** To calculate the F1-Score, please consider the obtained precision and recall measures.
