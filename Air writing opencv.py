"""
Air Writing (sem mediapipe) - desenhe no ar usando a webcam.
Detecta a pele pela cor e usa o ponto mais alto da mao como "caneta".

Controles:
  - Mostre a mao aberta dentro do retangulo verde na tela
  - Levante SO o dedo indicador para desenhar
  - Feche a mao (sem dedo levantado) para pausar
  - 'c' limpa a tela
  - 'q' sai
"""

import cv2
import numpy as np

# Tenta abrir a câmera padrão (0) sem forçar o backend CAP_DSHOW
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)


# Verificação inicial se a câmera respondeu
if not cap.isOpened():
    print("ERRO: Não foi possível acessar a webcam!")
    print("Verifique se outro programa (Zoom, Teams, Meet) não está usando a câmera,")
    print("ou se as permissões de privacidade do Windows permitem o acesso à câmera.")
    exit()

canvas = None
prev_point = None

# Regiao de interesse (ROI) onde a mao deve ficar
ROI = (300, 100, 620, 400)  # x1, y1, x2, y2

print("Câmera iniciada com sucesso. Pressione 'q' na janela do vídeo para sair.")

while True:
    ok, frame = cap.read()
    if not ok or frame is None:
        print("ERRO: A câmera abriu, mas falhou ao capturar o frame de vídeo.")
        break

    frame = cv2.flip(frame, 1)
    if canvas is None:
        canvas = np.zeros_like(frame)

    x1, y1, x2, y2 = ROI
    roi = frame[y1:y2, x1:x2]

    # Deteccao de pele em HSV
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 30, 60], dtype=np.uint8)
    upper = np.array([25, 150, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.GaussianBlur(mask, (5, 5), 0)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=2)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    drawing_now = False
    if contours:
        hand = max(contours, key=cv2.contourArea)
        if cv2.contourArea(hand) > 3000:
            hull = cv2.convexHull(hand)

            # Ponto mais alto do contorno = provavel ponta do dedo indicador
            top_point = tuple(hand[hand[:, :, 1].argmin()][0])
            fx, fy = top_point[0] + x1, top_point[1] + y1

            # Conta "dedos" via defeitos de convexidade
            hull_idx = cv2.convexHull(hand, returnPoints=False)
            fingers = 0
            if hull_idx is not None and len(hull_idx) > 3:
                defects = cv2.convexityDefects(hand, hull_idx)
                if defects is not None:
                    for defect in defects:
                        # s, e, f, d = defect[0]
                        d = defect[0]
                        if d > 12000:
                            fingers += 1

            drawing_now = fingers <= 1

            cv2.drawContours(roi, [hull], -1, (0, 255, 0), 2)
            cv2.circle(frame, (fx, fy), 8, (0, 0, 255), -1)

            if drawing_now:
                if prev_point is not None:
                    cv2.line(canvas, prev_point, (fx, fy), (0, 255, 0), 5)
                prev_point = (fx, fy)
            else:
                prev_point = None
    else:
        prev_point = None

    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    output = cv2.addWeighted(frame, 1, canvas, 1, 0)
    cv2.imshow("Air Writing", output)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas = np.zeros_like(frame)

cap.release()
cv2.destroyAllWindows()