import cv2

# Carrega o classificador Haar Cascade para detecção de rosto
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Abre a webcam (0 = webcam padrão)
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("Erro ao abrir a webcam!")
    exit()

while True:
    # Captura frame a frame
    ret, frame = camera.read()
    if not ret:
        print("Falha ao capturar imagem.")
        break

    # Converte o frame para escala de cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detecta rostos no frame
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Desenha retangulos ao redor dos rostos detectados
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Mostra o resultado
    cv2.imshow("Detecção de Rostos - Webcam", frame)

    # Aperte 'q' para sair
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libera a câmera e fecha as janelas
camera.release()
cv2.destroyAllWindows()
