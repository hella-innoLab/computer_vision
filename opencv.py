import numpy
import cv2

# Webcam starten (0 steht für die Standardkamera)
#1 steht z.B.: für die Frontkamera am Surface Laptop
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Fehler: Kamera konnte nicht geöffnet werden.")
    exit()

while True:
    # Frame von der Kamera lesen
    ret, frame = cap.read()
    
    if not ret:
        print("Fehler: Kein Bild erhalten.")
        break
    
    # Frame anzeigen
    cv2.imshow('Webcam-Stream', frame)
    
    # Mit 'q' beenden
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Ressourcen freigeben
cap.release()
cv2.destroyAllWindows()

