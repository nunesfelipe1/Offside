import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Função para carregar as imagens dos jogadores e suas classes correspondentes
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
            if "amarelo" in filename:
                y.append(1)  # Classe 1 para camisa amarela
            else:
                y.append(0)  # Classe 0 para camisa branca

    # Converter a lista X para um array numpy
    X = np.array(X)
    y = np.array(y)

    print(X)
    print(y)

    return X, y

# Função para detecção de jogadores em um frame
def detect_players(frame):
    print("detector de jogador")
    
    # Carregar os dados
    # data_dir = "C:/Users/Felipe Nunes/Desktop/Projeto Impedimento/Imagens/Jogadores/"
    # #data_dir = 'C:/Users/Felipe Nunes/Desktop/Projeto Impedimento/Imagens/Modelo/'
    # X, y = load_data(data_dir)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # Definir a arquitetura do modelo CNN
    # model = tf.keras.Sequential([
    #     tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(100, 100, 3)),
    #     tf.keras.layers.MaxPooling2D((2, 2)),
    #     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    #     tf.keras.layers.MaxPooling2D((2, 2)),
    #     tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(64, activation='relu'),
    #     tf.keras.layers.Dense(1, activation='sigmoid')  # Saída binária (0 ou 1)
    # ])

    # # # Compilar o modelo
    # model.compile(optimizer='adam',
    #             loss='binary_crossentropy',
    #             metrics=['accuracy'])
    
    # # # Treinar o modelo
    # model.fit(X_train, y_train, epochs=500, validation_data=(X_test, y_test))

    # model.save('C:/Users/Felipe Nunes/Desktop/Projeto Impedimento/Imagens/Modelo/ModeloAprendizado.h5')
    
    # Carregar o classificador Haar Cascade para detecção de jogadores
    model = tf.keras.models.load_model("C:/Users/Felipe Nunes/Desktop/Projeto Impedimento/Imagens/Modelo/ModeloAprendizado.h5")  # Adicione o caminho para o seu modelo   
   
    # Redimensionar o frame para o tamanho esperado pelo modelo
    resized_frame = cv2.resize(frame, (100, 100))
    input_image = resized_frame / 255.0
    # Expandir as dimensões do array de entrada
    input_image = np.expand_dims(input_image, axis=0)

    # Fazer previsão usando o modelo

    player_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')

    # Converter o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detectar jogadores no frame
    players = player_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Desenhar caixas delimitadoras em torno dos jogadores detectados
    for (x, y, w, h) in players:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame

    return frame
# Função para detecção de bolas em um frame
def detect_balls(frame):
    print("detector da bola")
    # Definir a faixa de cor da bola de futebol (neste exemplo, usei uma faixa de cor laranja)
    lower_orange = (0, 100, 200)
    upper_orange = (50, 255, 255)

    # Converter o frame de BGR para o espaço de cor HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Criar um mask para detectar a cor da bola de futebol
    mask = cv2.inRange(hsv_frame, lower_orange, upper_orange)

    # Aplicar operações morfológicas para eliminar ruídos na máscara
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # Encontrar contornos na máscara
    contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Inicializar lista para armazenar as caixas delimitadoras das bolas
    ball_boxes = []

    # Iterar sobre os contornos encontrados
    for contour in contours:
        # Calcular o retângulo delimitador para cada contorno
        (x, y, w, h) = cv2.boundingRect(contour)

        # Adicionar a caixa delimitadora à lista de caixas delimitadoras das bolas
        ball_boxes.append((x, y, x + w, y + h))

        # Desenhar a caixa delimitadora ao redor da bola no frame original
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    return frame, ball_boxes

# Caminho para o vídeo
video_path = "C:/Users/Felipe Nunes/Desktop/Projeto Impedimento/Video/Exemplo.mp4"

# Inicializa o objeto de captura de vídeo
cap = cv2.VideoCapture(video_path)

# Loop de leitura de frames
while(cap.isOpened()):
    ret, frame = cap.read()

    # Se o frame for lido corretamente
    if ret:
        # Detecção de jogadores
        frame_with_players = detect_players(frame)
        
        # Detecção de bolas
        #frame_with_balls = detect_balls(frame_with_players)
        
        # Mostra o frame com as detecções
        cv2.imshow('Detected Players', frame_with_players)

        # Fecha a janela quando a tecla 'q' é pressionada
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

# Libera o objeto de captura de vídeo e fecha todas as janelas
cap.release()
cv2.destroyAllWindows()
