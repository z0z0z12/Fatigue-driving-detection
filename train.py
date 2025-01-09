from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import BatchNormalization, Concatenate, Conv2D, AveragePooling2D, Flatten, Dense, Input, Dropout, Add, MaxPooling2D
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
from PIL import Image
import os
from sklearn.metrics import f1_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import random

# def set_seed(seed):
#     tf.random.set_seed(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#
#
# set_seed(40)

tf.random.set_seed(30)
dic = {"drowsy" : 0, "notdrowsy" : 1}

def dispose1(image_path):
    image = Image.open(image_path)
    gray_image = image.convert('L')
    resized_image = gray_image.resize((64, 64))
    return np.array(resized_image)


def dispose(folder_path):
    image_tensors = []
    labels = []
    for imgg in os.listdir(folder_path):
        image = os.path.join(folder_path, imgg)
        for imgs in os.listdir(image):
            file_path = os.path.join(image, imgs)
            tensor = dispose1(file_path)
            image_tensors.append(tensor)
            label = os.path.basename(imgg)
            labels.append(dic[label])
    image_tensors = np.array(image_tensors)
    image_tensors = np.expand_dims(image_tensors, -1)
    label_indices = np.array(labels)
    return image_tensors, label_indices

dir = "data1"
train_dir = "train_data"
test_dir = "test_data"
validation_dir = "validation_data"

print("数据处理中")
train_path = os.path.join(dir, train_dir)
test_path = os.path.join(dir, test_dir)
validation_path = os.path.join(dir, validation_dir)
x_train, y_train = dispose(train_path)
x_test, y_test = dispose(test_path)
x_validation, y_validation = dispose(validation_path)

x_train = x_train / 255.0
x_test = x_test / 255.0
x_validation = x_validation / 255

y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)
y_validation = tf.keras.utils.to_categorical(y_validation)

print("数据处理完成")
# 空间注意力机制模块
def spatial_attention(input_feature):

    avg_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.mean(x, axis=3, keepdims=True))(input_feature)

    max_pool = tf.keras.layers.Lambda(lambda x: tf.keras.backend.max(x, axis=3, keepdims=True))(input_feature)

    concat = tf.keras.layers.Concatenate(axis=3)([avg_pool, max_pool])

    spatial_attention_weight = tf.keras.layers.Conv2D(1, kernel_size=(7, 7), padding='same', activation='sigmoid')(concat)

    spatial_attention_feature = tf.keras.layers.Multiply()([input_feature, spatial_attention_weight])

    return spatial_attention_feature
def model_gaijin():
    inp = Input(shape = (64, 64, 1))

    conv1_1 = Conv2D(16, 3, 1, activation = 'relu', padding = 'same')(inp)

    conv2_1 = Conv2D(16, 3, 1, padding = 'same')(inp)
    bn2_1 = BatchNormalization()(conv2_1)
    conv2_1 = tf.keras.layers.Activation('sigmoid')(bn2_1)
    pool2_1 = AveragePooling2D(2)(conv2_1)

    conv3_1 = Conv2D(16, 3, 1, activation = 'relu', padding = 'same')(inp)
    ma1 = tf.multiply(conv1_1, conv2_1)

    bn1_1 = BatchNormalization()(ma1)
    pool1_1 = AveragePooling2D(2)(bn1_1)
    bn3_1 = BatchNormalization()(conv3_1)
    pool3_1 = AveragePooling2D(2)(bn3_1)

    conv1_2 = Conv2D(32, 3, 1, activation = 'relu', padding = 'same')(pool1_1)

    conv2_2 = Conv2D(32, 3, 1, padding='same')(pool2_1)
    bn2_2 = BatchNormalization()(conv2_2)
    conv2_2 = tf.keras.layers.Activation('sigmoid')(bn2_2)

    conv3_2 = Conv2D(32, 3, 1, activation='relu', padding='same')(pool3_1)

    ma2 = tf.multiply(conv1_2, conv2_2)

    bn1_2 = BatchNormalization()(ma2)
    pool1_2 = AveragePooling2D(2)(bn1_2)
    bn3_2 = BatchNormalization()(conv3_2)
    pool3_2 = AveragePooling2D(2)(bn3_2)

    add1 = Add()([pool1_2, pool3_2])
    c = spatial_attention(add1)

    conv = Conv2D(32, 3, 1, activation = 'relu', padding = 'same')(c)
    bn = BatchNormalization()(conv)
    pool_11 = AveragePooling2D(2)(bn)

    # c = spatial_attention(pool_11)
    #
    # convv = Conv2D(16, 3, 1, activation = 'relu', padding = 'same')(c)
    # bnn = BatchNormalization()(convv)
    # pooll = AveragePooling2D(2)(bnn)

    f = Flatten()(pool_11)

    dense1 = Dense(120, activation = 'relu')(f)
    dense2 = Dense(60, activation = 'relu')(dense1)
    out = Dense(2, activation = 'softmax')(dense2)

    model = Model(inp, out)
    learning_rate = 0.00001
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

model = model_gaijin()
print(model.summary())
train_history = model.fit(x=x_train, y=y_train, epochs=60, batch_size=32, validation_data=(x_validation, y_validation))
score = model.evaluate(x=x_test, y=y_test, batch_size=32)
print("\n%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
model.save('model_v6.0.h5')

def print_history(history):
    # plt.subplot(121)
    plt.plot(history.history['accuracy'], marker = "o", color = "r")
    plt.plot(history.history['val_accuracy'], marker = "o", color = "g")
    plt.title('accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train_acc', 'Val_acc'])
    plt.show()
    # plt.subplot(122)
    plt.plot(history.history['loss'], marker = "o", color = "r")
    plt.plot(history.history['val_loss'], marker = "o", color = "g")
    plt.title('loss')
    plt.xlabel('Epoch')
    plt.legend(['Train_loss', 'Val_loss'])
    plt.show()
print(train_history.params)
print_history(train_history)

loaded_model = load_model('model_v6.0.h5')
test_loss, test_acc = loaded_model.evaluate(x=x_test, y=y_test, batch_size=32)
print(f"Test accuracy: {test_acc:.4f}")
print(f"Test loss: {test_loss:.4f}")

y_pred = loaded_model.predict(x_test)
y_pred = np.round(y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
print("f1_score：", f1)

recall = recall_score(y_test, y_pred, average='macro')
print("Recall：", recall)

# 以下是绘制混淆矩阵的代码
# 获取预测的类别标签（将one - hot编码转换为类别索引）
y_pred_labels = np.argmax(y_pred, axis=1)
y_test_labels = np.argmax(y_test, axis=1)

# 计算混淆矩阵
confusion_matrix = tf.math.confusion_matrix(labels=y_test_labels, predictions=y_pred_labels)

# 绘制混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion matrix')
plt.show()

