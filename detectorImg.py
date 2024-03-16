import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
import cv2
import os

# Carregar e pré-processar os dados
def load_data(data_dir, target_size=(100, 100)):
    X = []
    y = []

    # Iterar sobre os arquivos no diretório
    for filename in os.listdir(data_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Carregar a imagem usando OpenCV
            img = cv2.imread(os.path.join(data_dir, filename))
            # Redimensionar a imagem para o tamanho fixo
            img = cv2.resize(img, target_size)
            # Adicionar a imagem à lista X
            X.append(img)
            # Determinar a classe da imagem com base no nome do arquivo ou em algum outro critério
            if "amarela" in filename:
                y.append(1)  # Classe 1 para camisa amarela
            else:
                y.append(0)  # Classe 0 para camisa branca

    # Converter a lista X para um array numpy
    X = np.array(X)
    y = np.array(y)

    return X, y

# Dividir o conjunto de dados em treinamento e teste
data_dir = 'C:/Users/Felipe Nunes/Desktop/Projeto Impedimento/Imagens/Jogadores'
X, y = load_data(data_dir)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir a arquitetura do modelo CNN
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Saída binária (0 ou 1)
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Treinar o modelo
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

# Avaliar o modelo
test_loss, test_acc = model.evaluate(X_test, y_test)
print('Test accuracy:', test_acc)
