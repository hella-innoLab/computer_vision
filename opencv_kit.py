import cv2
import numpy as np

# Funktion für Graustufen
# Funktion für Graustufen
def to_grayscale(image):
    # Überprüfe, ob das Bild bereits Graustufen hat
    if len(image.shape) == 3:
        # Stelle sicher, dass das Bild den richtigen Datentyp hat
        image = np.uint8(image)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    return image

# Funktion für Gaussian Blur (Tiefpassfilter)
def gaussian_blur(image, ksize=(5, 5), sigmaX=0):
    return cv2.GaussianBlur(image, ksize, sigmaX)

# Funktion für Laplacian Filter (Kantenverstärkung)
def laplacian_filter(image):
    return cv2.Laplacian(image, cv2.CV_64F)

# Funktion für Sobel Filter (Kantenverstärkung)
def sobel_filter(image):
    if len(image.shape) == 3:  # Falls Bild Farbbild ist, erst in Graustufen umwandeln
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    return cv2.magnitude(grad_x, grad_y)

# Funktion für Schärfefilter (Unsharp Mask)
def sharpen(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 1.5)
    return cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

# Funktion für Hochpassfilter (Differenz von Original und Weichzeichnung)
def high_pass_filter(image):
    blurred = cv2.GaussianBlur(image, (21, 21), 0)
    return cv2.absdiff(image, blurred)

# Funktion für Tiefpassfilter (Weichzeichnung)
def low_pass_filter(image):
    return cv2.GaussianBlur(image, (21, 21), 0)

# Funktion für Thresholding (Schwellenwert)
def threshold_filter(image, threshold_value=127):
    _, thresholded = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    return thresholded

# Funktion für Canny Edge Detection (Kantenfindung)
# Funktion für Canny Edge Detection (Kantenfindung)
def canny_edge_detection(image, low_threshold=1, high_threshold=200):
    gray_image = to_grayscale(image)  # Bild in Graustufen umwandeln
    
    # Stelle sicher, dass das Bild den richtigen Datentyp hat (8-Bit)
    gray_image = np.uint8(gray_image)
    
    return cv2.Canny(gray_image, low_threshold, high_threshold)

# Funktion für Farbverstärkung (z.B. Sättigung erhöhen)
def color_enhance(image):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv_image[..., 1] = hsv_image[..., 1] * 1.5  # Erhöhe die Sättigung
    return cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)

# Funktion für ein simples Rauschen (random noise)
def add_noise(image):
    row, col, ch = image.shape
    mean = 0
    sigma = 25
    gauss = np.random.normal(mean, sigma, (row, col, ch))
    noisy = np.add(image, gauss)
    return np.clip(noisy, 0, 255).astype(np.uint8)

# Funktion für den Hochpassfilter durch Frequenzbereich
def frequency_high_pass_filter(image):
    # Wenn das Bild ein Farbbild ist, konvertiere es in Graustufen
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Stelle sicher, dass das Bild den richtigen Datentyp hat (32-Bit Fließkommazahl)
    image_float = np.float32(image)  # Achte darauf, dass der Typ korrekt ist

    # Fourier-Transformation und Filterung im Frequenzbereich
    dft = cv2.dft(image_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # Erstelle einen Lowpass-Filter und wende ihn auf das Bild an
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    dft_shift[crow-30:crow+30, ccol-30:ccol+30] = 0
    
    # Rücktransformation
    idft_shift = np.fft.ifftshift(dft_shift)
    idft = cv2.idft(idft_shift)
    img_back = cv2.magnitude(idft[:, :, 0], idft[:, :, 1])
    
    return np.uint8(img_back)

# Funktion, um mehrere Filter miteinander zu kombinieren
def apply_filters(image, filters):
    filtered_image = image
    for filter_func in filters:
        filtered_image = filter_func(filtered_image)
    return filtered_image


def detect_and_draw_rectangles(image):
    # Schritt 1: Bild in Graustufen umwandeln
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Schritt 2: Bild unscharf machen, um Rauschen zu entfernen
    blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0)
    
    # Schritt 3: Canny-Kantenerkennung anwenden
    edges = cv2.Canny(blurred_image, 50, 150)
    
    # Schritt 4: Konturen finden
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Schritt 5: Rechtecke erkennen und zeichnen
    for contour in contours:
        # Berechne die umschließende Rechteck-Kontur
        epsilon = 0.02 * cv2.arcLength(contour, True)  # Genauigkeit
        approx = cv2.approxPolyDP(contour, epsilon, True)  # Näherung an die Kontur
        
        # Wenn die Kontur 4 Ecken hat, dann ein Rechteck
        if len(approx) == 4:
            # Zeichne das Rechteck auf das Originalbild
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)  # Grün, Dicke 2
    
    return image



# Beispiel für die Verwendung der Filter
if __name__ == '__main__':
    cap = cv2.VideoCapture(2)  # Webcam

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Beispiel: Kombiniere mehrere Filter
        filters_to_apply = [
           
            detect_and_draw_rectangles
            
            
        ]
        
        # Wende die Filter an
        processed_frame = apply_filters(frame, filters_to_apply)
        
        # Zeige das Ergebnis an
        cv2.imshow('Gefiltertes Bild', processed_frame)
        cv2.imshow('orignal', frame)

        
        # Beende das Programm mit der 'q'-Taste
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
