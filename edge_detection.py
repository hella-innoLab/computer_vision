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
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    edges = cv2.Canny(gray, 90, 200)

    ''' image: Das Eingabebild (Graustufen).

        threshold1: Der niedrigere Schwellwert für die Hysterese-Schwellwertverarbeitung.

        threshold2: Der höhere Schwellwert für die Hysterese-Schwellwertverarbeitung.

        apertureSize: Die Größe des Sobel-Kernels (standardmäßig 3).

        L2gradient: Ein Flag, das angibt, ob die genauere L2-Norm für die Gradientenberechnung verwendet werden soll (standardmäßig False, d. h. L1-Norm)
    '''
    
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)  # Konvertiere zu BGR für Farbe
    edges_colored[edges == 255] = [0, 0, 255]

    # Kombiniere Original und Kanten (Überlagerung)
    combined = cv2.addWeighted(frame, 0.5, edges_colored, 1, 0)
    #Alpha= Intensität erstes Bild
    #Beta= Intensität zweites Bild
    #Gamma=Helligkeit

    # Zeige das kombinierte Bild an
    cv2.imshow("Live Kanten", combined)
    
    # Mit 'q' beenden
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Ressourcen freigeben
cap.release()
cv2.destroyAllWindows()

