###############################################################################
# AphidCV Augmentantion Version "Color-Input Dataset-2kBalanced Batchsize-100"
# Channels Last is default! NHWC
# (N: batch_size, C: nro_channels, H: input_img_height, W: input_img_width)
###############################################################################
# New models' support for AphidCV 3.0 (and comparison with YOLOv8m)
###############################################################################

# Please, check if these libs are installed. If not, proceed:
# !pip install tensorflow[and-cuda]==2.8
# !pip install albumentations
# !pip install ImageDataAugmentor.zip # instalation file without version-specific dependencies for albumentations and opencv-python
# !pip install pydot
# !pip install pydotplus
# !pip install pydot_ng
# !pip install graphviz  # AND => sudo apt-get install graphviz
# !pip install plot_model

import numpy as np
import tensorflow as tf
import keras.backend as K
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import concatenate
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint

from matplotlib import pyplot as plt
import pandas as pd
from PIL import ImageFont
import visualkeras
from keras.utils.vis_utils import plot_model
import time

###############################################################################
# Augmentation Segment ########################################################
###############################################################################

import albumentations as A
from ImageDataAugmentor.image_data_augmentor import *
import cv2
from albumentations.core.transforms_interface import ImageOnlyTransform

# Additional filter & preprocessing function implementations
def get_random_kernel():
    structure = np.random.choice([cv2.MORPH_RECT, cv2.MORPH_ELLIPSE, cv2.MORPH_CROSS])
    kernel = cv2.getStructuringElement(structure, tuple(np.random.randint(1, 6, 2)))
    return kernel

def closing(img):
    img = cv2.dilate(img, get_random_kernel(), iterations=1)
    img = cv2.erode(img, get_random_kernel(), iterations=1)
    return img

def opening(img):
    img = cv2.erode(img, get_random_kernel(), iterations=1)
    img = cv2.dilate(img, get_random_kernel(), iterations=1)
    return img    

class Closing(ImageOnlyTransform):
    def apply(self, img, **params):
        return closing(img)

class Opening(ImageOnlyTransform):
    def apply(self, img, **params):
        return opening(img)
        
def normalize_transf(image):
    return cv2.normalize(image, np.zeros((image.shape[1],image.shape[0])), 0, 255, cv2.NORM_MINMAX)

# Set data augmentation using Albumentations
AUGMENTATION_transform = A.Compose([
    A.RandomRotate90(p=0.5),
    A.Blur(p=0.25),
    A.RandomBrightnessContrast(p=0.25),
    A.Sharpen(p=0.25),
    A.Emboss(p=0.25),
    Opening(p=0.25),
    Closing(p=0.25),
    A.CLAHE(p=0.25),
    A.Affine(shear=([-45, 45]), scale=(0.5, 1.5), p=0.2),
    A.Flip(p=0.5),
])
        
###############################################################################
# Initial setup vars ##########################################################
###############################################################################

# Verify GPU is active
# Important: TF 2.8.0 executes only using Cuda 11.2-11.8 AND Cudnn 8.1-8.7
# (Python support: 3.7-3.10)
# More details: https://www.tensorflow.org/install/source?hl=pt-br#gpu
gpus = tf.config.list_physical_devices('GPU')
for gpu in gpus:
    print("*** Name:", gpu.name, "  Type:", gpu.device_type)

AphidCV_image_size = (120, 120)
AphidCV_input_shape = (120, 120, 3)
AphidCV_batch_size = 100
AphidCV_epochs = 150

import datetime
inicio = datetime.datetime.now()

###############################################################################
# AphidCV CNN Architecture ####################################################
###############################################################################

input_img = Input(shape = AphidCV_input_shape)
camada_1 = Conv2D(96, (7, 7), activation='relu', data_format='channels_last')(input_img)
camada_1_b = BatchNormalization()(camada_1)
camada_2 = Conv2D(96, (7, 7), activation='relu', data_format='channels_last')(camada_1_b)
camada_2_b = BatchNormalization()(camada_2)
camada_3 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(camada_2_b)
camada_4 = Dropout(0.2)(camada_3)
camada_5 = Conv2D(64, (5,5), activation='relu', data_format='channels_last')(camada_4)
camada_5_b = BatchNormalization()(camada_5)
camada_6 = Conv2D(64, (5,5), activation='relu', data_format='channels_last')(camada_5_b)
camada_6_b = BatchNormalization()(camada_6)
camada_7 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(camada_6_b)
camada_8 = Dropout(0.2)(camada_7)

tower_1 = Conv2D(64, (1,1), padding='same', activation='relu', data_format='channels_last')(camada_8)
tower_1 = Conv2D(64, (3,3), padding='same', activation='relu', data_format='channels_last')(tower_1)
tower_2 = Conv2D(64, (1,1), padding='same', activation='relu', data_format='channels_last')(camada_8)
tower_2 = Conv2D(64, (5,5), padding='same', activation='relu', data_format='channels_last')(tower_2)
tower_3 = MaxPooling2D((3,3), strides=(1,1), padding='same', data_format='channels_last')(camada_8)
tower_3 = Conv2D(64, (1,1), padding='same', activation='relu', data_format='channels_last')(tower_3)

camada_9 = concatenate([tower_1, tower_2, tower_3], axis = 1)
camada_9_b = BatchNormalization()(camada_9)
camada_9_ad_1 = MaxPooling2D(pool_size=(2, 2), data_format='channels_last')(camada_9_b)

tower_4 = Conv2D(32, (1,1), padding='same', activation='relu', data_format='channels_last')(camada_9_ad_1)
tower_4 = Conv2D(32, (3,3), padding='same', activation='relu', data_format='channels_last')(tower_4)
tower_5 = Conv2D(32, (1,1), padding='same', activation='relu', data_format='channels_last')(camada_9_ad_1)
tower_5 = Conv2D(32, (5,5), padding='same', activation='relu', data_format='channels_last')(tower_5)
tower_6 = MaxPooling2D((3,3), strides=(1,1), padding='same', data_format='channels_last')(camada_9_ad_1)
tower_6 = Conv2D(32, (1,1), padding='same', activation='relu', data_format='channels_last')(tower_6)

camada_9_ad_3 = concatenate([tower_4, tower_5, tower_6], axis = 1)

camada_9_ad_4 = Conv2D(64, (3,3), activation='relu', data_format='channels_last')(camada_9_ad_3)
camada_9_ad_4b = BatchNormalization()(camada_9_ad_4)
camada_9_ad_4d = Dropout(0.2)(camada_9_ad_4b)
camada_9_ad_5 = Conv2D(64, (1,1), activation='relu', data_format='channels_last')(camada_9_ad_4d)
camada_9_ad_5b = BatchNormalization()(camada_9_ad_5)
camada_9_ad_5d = Dropout(0.2)(camada_9_ad_5b)

tower_7 = Conv2D(32, (1,1), padding='same', activation='relu', data_format='channels_last')(camada_9_ad_5d)
tower_7 = Conv2D(32, (3,3), padding='same', activation='relu', data_format='channels_last')(tower_7)
tower_8 = Conv2D(32, (1,1), padding='same', activation='relu', data_format='channels_last')(camada_9_ad_5d)
tower_8 = Conv2D(32, (5,5), padding='same', activation='relu', data_format='channels_last')(tower_8)
tower_9 = MaxPooling2D((3,3), strides=(1,1), padding='same', data_format='channels_last')(camada_9_ad_5d)
tower_9 = Conv2D(32, (1,1), padding='same', activation='relu', data_format='channels_last')(tower_9)

camada_9_ad_6 = concatenate([tower_7, tower_8, tower_9], axis = 1)

camada_9_ad_7 = Conv2D(16, (3,3), activation='relu', data_format='channels_last')(camada_9_ad_6)
camada_9_ad_7b = BatchNormalization()(camada_9_ad_7)
camada_9_ad_7d = Dropout(0.2)(camada_9_ad_7b)
camada_9_ad_8 = Conv2D(16, (1,1), activation='relu', data_format='channels_last')(camada_9_ad_7d)
camada_9_ad_8b = BatchNormalization()(camada_9_ad_8)
camada_9_ad_8d = Dropout(0.2)(camada_9_ad_8b)

tower_10 = Conv2D(16, (1,1), padding='same', activation='relu', data_format='channels_last')(camada_9_ad_8b)
tower_10 = Conv2D(16, (3,3), padding='same', activation='relu', data_format='channels_last')(tower_10)
tower_11 = Conv2D(16, (1,1), padding='same', activation='relu', data_format='channels_last')(camada_9_ad_8b)
tower_11 = Conv2D(16, (5,5), padding='same', activation='relu', data_format='channels_last')(tower_11)
tower_12 = MaxPooling2D((3,3), strides=(1,1), padding='same', data_format='channels_last')(camada_9_ad_8b)
tower_12 = Conv2D(16, (1,1), padding='same', activation='relu', data_format='channels_last')(tower_12)

camada_9_ad_9 = concatenate([tower_10, tower_11, tower_12], axis = 1)

camada_10 = Dropout(0.3)(camada_9_ad_9)

output = Flatten()(camada_10)

camada_12 = Dense(256, activation='relu')(output)

camada_15 = Dense(64, activation='relu')(camada_12)

out = Dense(4, activation='softmax')(camada_15)

model = Model(inputs = input_img, outputs = out)

METRICS = [
    tf.keras.metrics.Precision(name='precision'),
    tf.keras.metrics.Recall(name='recall'),
    tf.keras.metrics.AUC(name='roc', curve='ROC'), # roc curve
    tf.keras.metrics.AUC(name='prc', curve='PR')   # precision-recall curve
]

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', METRICS])

###############################################################################
# Summary and model layer/arch graphs  ########################################
###############################################################################

model.summary()

#font = ImageFont.load_default()
#visualkeras.layered_view(model, legend=True, font=font, to_file='./PAPER/AphidCV_CNN_layer.png')

plot_model(model, to_file='./PAPER/AphidCV_CNN_model.png', show_shapes=True, show_layer_names=False, rankdir='TB', expand_nested=False, dpi=300)  # CHANGE HERE BEFORE EACH TRAINING

# To generate a colored plot model, use Netron!
# Upload a model and save as image in: https://github.com/lutzroeder/Netron

###############################################################################
# Data generator using ImageDataAugmentor #####################################
###############################################################################

# Training data (with augmentation, with normalization)
train_datagen = ImageDataAugmentor(
    augment=AUGMENTATION_transform,
    preprocess_input=normalize_transf,
    rescale=1./255,
    data_format='channels_last')

# Validation data (NO augmentation, with normalization)
val_datagen = ImageDataAugmentor(
    preprocess_input=normalize_transf,
    rescale=1./255,
    data_format='channels_last')

###############################################################################
# Load datasets applying augmentations ########################################
###############################################################################
# Aphid species:
# - Rhopalosiphum_padi
# - Schizaphis_graminum
# - Metopolophium_dirhodum
# - Sitobion_avenae
# - Myzus_persicae
# - Brevicoryne_brassicae
# 
# Dataset strucuture:
# - data: 0 1 2 3 (Winged, Wingless, Falses, Nymphs)
# - validation: 0 1 2 3  (Winged, Wingless, Falses, Nymphs)

X_train = train_datagen.flow_from_directory(
        './Rhopalosiphum_padi/2000/data',  # CHANGE HERE BEFORE EACH TRAINING
        target_size=AphidCV_image_size,
        batch_size=AphidCV_batch_size,
        class_mode='categorical',
        shuffle=True)

y_train = val_datagen.flow_from_directory(
         './Rhopalosiphum_padi/2000/validation',  # CHANGE HERE BEFORE EACH TRAINING
        target_size=AphidCV_image_size,
        batch_size=AphidCV_batch_size,
        class_mode='categorical',
        shuffle=True)

###############################################################################
# Compile, fit and save model #################################################
###############################################################################
# Acronyms for output models:
# AphidCV_Rp = Rhopalosiphum padi
# AphidCV_Sg = Schizaphis graminum
# AphidCV_Md = Metopolophium dirhodum
# AphidCV_Sa = Sitobion avenae
# AphidCV_Mp = Myzus persicae
# AphidCV_Bb = Brevicoryne brassicae

basename = "./PAPER/AphidCV_Rp"  # CHANGE HERE BEFORE EACH TRAINING

#timestr = time.strftime("-%d%b-%H%M")
modelfilepath =    basename + "-weights"   + "-{epoch:03d}-{val_accuracy:.3f}-{val_precision:.3f}-{val_recall:.3f}-{val_roc:.3f}-{val_prc:.3f}.h5"
historyfilepath =  basename + "-history"   + ".csv"
plotACCfilepath =  basename + "-accuracy"  + ".png"
plotLOSfilepath =  basename + "-loss"      + ".png"
plotPREfilepath =  basename + "-precision" + ".png"
plotRECfilepath =  basename + "-recall"    + ".png"
plotROCfilepath =  basename + "-roc"       + ".png"
plotPRCfilepath =  basename + "-prc"       + ".png"

checkpoint = ModelCheckpoint(modelfilepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

early_stopping = EarlyStopping(
    monitor="accuracy",
    min_delta=0,
    patience=10,
    verbose=1,
    mode="auto",
    baseline=None,
    restore_best_weights=False,
)
callbacks_list = [checkpoint, early_stopping]

model.fit(
    X_train,
    steps_per_epoch=len(X_train),   # 2k Test, 80% = 1600
    epochs=AphidCV_epochs,
    validation_data=y_train,
    validation_steps=len(y_train),  # 2k Test, 20% =  400
    callbacks=callbacks_list)

###############################################################################
# Save training/validation history ############################################
###############################################################################

plt.plot(model.history.history["accuracy"], label="Training accuracy")
plt.plot(model.history.history["val_accuracy"], label="Validation accuracy")
plt.legend()
#plt.show()
plt.savefig(plotACCfilepath)
plt.clf()

plt.plot(model.history.history["loss"], label="Training loss")
plt.plot(model.history.history["val_loss"], label="Validation loss")
plt.legend()
#plt.show()
plt.savefig(plotLOSfilepath)
plt.clf()

plt.plot(model.history.history["precision"], label="Training precision")
plt.plot(model.history.history["val_precision"], label="Validation precision")
plt.legend()
#plt.show()
plt.savefig(plotPREfilepath)
plt.clf()

plt.plot(model.history.history["recall"], label="Training recall")
plt.plot(model.history.history["val_recall"], label="Validation recall")
plt.legend()
#plt.show()
plt.savefig(plotRECfilepath)
plt.clf()

plt.plot(model.history.history["roc"], label="Training roc")
plt.plot(model.history.history["val_roc"], label="Validation roc")
plt.legend()
#plt.show()
plt.savefig(plotROCfilepath)
plt.clf()

plt.plot(model.history.history["prc"], label="Training prc")
plt.plot(model.history.history["val_prc"], label="Validation prc")
plt.legend()
#plt.show()
plt.savefig(plotPRCfilepath)
plt.clf()

hist_df = pd.DataFrame(model.history.history)
hist_df.to_csv(historyfilepath)

print("Rhopalosiphum_padi done!")  # CHANGE HERE BEFORE EACH TRAINING
###############################################################################
fim = datetime.datetime.now()
diff = fim - inicio
print("Time-processing: ", diff)
f = open("Time-processing.txt", "a")
f.write("AphidCV_Rp = " + str(diff) + "\n")  # CHANGE HERE BEFORE EACH TRAINING
f.close()

