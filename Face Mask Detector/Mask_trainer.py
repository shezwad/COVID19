
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import os


ini_learn_rate = 1e-4 # Initial learning rate
Epochs = 20           # Number of epochs
bs = 32               # Batch size

directory = r"C:\Users\SINGER\Desktop\FUN PYTHON\Face Mask Detector\dataset"
categorys = ["with_mask", "without_mask"]

data = []
labels = []

for category in categorys:
    path = os.path.join(directory, category)
    for img in os.listdir(path):
        img_path = os.path.join(path, img)
        image = load_img(img_path, target_size = (224,224))
        image = img_to_array(image)
        image = preprocess_input(image)

        data.append(image)
        labels.append(category)


LB = LabelBinarizer()
labels = LB.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype = "float32")
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size = 0.20, stratify = labels, random_state = 40)

augment = ImageDataGenerator(
    rotation_range = 20,
    zoom_range = 0.15,
    width_shift_range = 0.2,
    height_shift_range = 0.15,
    horizontal_flip = True,
    fill_mode = "nearest")

baseModel = MobileNetV2(weights = "imagenet", include_top = False, input_tensor = Input(shape = (224,224,3)))

headModel = baseModel.output
headModel = AveragePooling2D(pool_size = (7,7))(headModel)
headModel = Flatten(name = "flatten")(headModel)
headModel = Dense(128, activation = "relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation = "softmax")(headModel)

model = Model(inputs=baseModel.input, outputs=headModel)

for layer in baseModel.layers:
    layer.trainable = False

print("[INFO compiling model...")
opt = Adam(lr = ini_learn_rate, decay = ini_learn_rate/Epochs)
model.compile(loss = "binary_crossentropy", optimizer = opt, metrics = ["accuracy"])

print("[INFO] traning head...")
H = model.fit(
    augment.flow(trainX, trainY , batch_size = bs),
    steps_per_epoch = len(trainX) // bs,
    validation_data = (testX, testY),
    validation_steps = len(testX) // bs,
    epochs = Epochs)

print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size = bs)

predIdxs = np.argmax(predIdxs, axis = 1)


print(classification_report(testY.argmax(axis = 1), predIdxs, traget_names = LB.classes_))

print("[INFO] saving mask detector model...")
model.save("mask_detection.model", save_format = "h5")

N = Epochs
plt.styles.use("ggplot")
plt.figure()
plt.plot(np.arrange(0, N), H.history["loss"], Label = "train_loss")
plt.plot(np.arrange(0, N), H.history["val_loss"], Label = "val_loss")
plt.plot(np.arrange(0, N), H.history["accuracy"], Label = "train_acc")
plt.plot(np.arrange(0, N), H.history["val_accuracy"], Label = "val_acc")
plt.title("training Loss and Accuracy")
plt.xlabel("Epochs #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc = "lower left")
plt.savefig("plot.png")

print('Done')

