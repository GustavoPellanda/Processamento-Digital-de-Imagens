# Fazer um máscara com o gabarito correto de referência
# Receber um gabarito preenchido
# Detectar as bordas do gabarito preenchido
# Fazer a transformação perspectiva para alinhar o gabarito preenchido com o gabarito correto
# Comparar o gabarito preenchido com o gabarito correto
# Marcar as respostas corretas e incorretas no gabarito preenchido
# Exibir a pontuação final
# ------------------------------------------------------------------------------
# testar imagens tortas, com sombra e gabaritos preenchidos de forma inválida (marcar com X, cor de caneta errada)
# homografia - para fazer a máscara binária do gabarito
# ------------------------------------------------------------------------------
# ideia: sobrepor a máscara binária do gabarito correto sobre o gabarito preenchido alinhado
# a quantidade de bolinhas brancas que sorarem serão as questões erradas
# as questões corretas serão as bolinhas pretas, que vão acabar desaparecendo na máscara preta
# ------------------------------------------------------------------------------

import cv2
import numpy as np

def detectar_retangulo(img):
    """
    Detecta o maior retângulo/quadrilátero na imagem.
    Retorna (rect_box, imagem_annotada) onde rect_box é np.array shape (4,2) ou None.
    """
    orig = img.copy()

    # pré-processamento: escala de cinza, blur, detecção de bordas e fechamento
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel, iterations=2)

    # encontrar contornos externos
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print("Nenhum contorno encontrado.")
        return None, orig

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    rect_box = None

    # procurar o maior contorno quadrilátero (4 vértices)
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            rect_box = approx.reshape(4, 2)
            break

    # se não encontrar um quadrilátero, usar o retângulo de área mínima do maior contorno
    if rect_box is None:
        largest = contours[0]
        box = cv2.boxPoints(cv2.minAreaRect(largest))
        rect_box = box.astype(int)

    rect_box = rect_box.astype(int)

    # desenhar polilinha do retângulo detectado
    cv2.polylines(orig, [rect_box], isClosed=True, color=(0, 255, 0), thickness=3)

    # calcular e mostrar bounding rect
    x, y, w, h = cv2.boundingRect(rect_box)
    print(f"Retângulo detectado: x={x}, y={y}, w={w}, h={h}")

    return rect_box, orig

def criar_mascara_preto(gray, rect_box):
    """
    Cria uma máscara binária mantendo apenas o preto dentro do polígono rect_box.
    Recebe a imagem em escala de cinza e o rect_box (ou None).
    Retorna mask_final (ou None se rect_box for None).
    """
    if rect_box is None:
        print("Nenhum retângulo para criar máscara.")
        return None
    
    # máscara do polígono (retângulo detectado)
    mask_poly = np.zeros(gray.shape, dtype=np.uint8)
    cv2.fillPoly(mask_poly, [rect_box.astype(np.int32)], 255)

    # aplicar máscara à imagem em escala de cinza para isolar a região do gabarito
    masked_gray = cv2.bitwise_and(gray, gray, mask=mask_poly)

    # separar preto e branco; manter apenas o preto (inverter para que preto vire branco no binário)
    # usar Otsu para escolher limiar automaticamente
    _, mask_black = cv2.threshold(masked_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # preencher as bordas do polígono com preto
    mask_black = cv2.bitwise_and(mask_black, mask_poly)

    # remover pequeno ruído e linhas finas com erosão
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask_clean = cv2.morphologyEx(mask_black, cv2.MORPH_OPEN, kernel, iterations=1)

    # erosão adicional para remover linhas muito finas
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    mask_clean = cv2.erode(mask_clean, kernel_erode, iterations=6)

    # garantir que fora do polígono esteja em zero
    mask_final = cv2.bitwise_and(mask_clean, mask_poly)

    cv2.imshow("Mask Clean after Erosion", mask_final)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
         
    return mask_final

#def sobrepor_mascara():

    #TERMINAR            

    #return sobreposta

# abrir imagem e usar as funções
filename = "gabarito_preenchido.bmp"
img = cv2.imread(filename)
if img is None:
    print(f"Erro: não foi possível abrir '{filename}'")
else:
    rect_box, img_annotada = detectar_retangulo(img)

    # exibir e salvar resultado do retângulo detectado
    cv2.imshow("Retangulo Detectado", img_annotada)
    cv2.imwrite("gabarito_detectado.png", img_annotada)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # criar máscara binária dentro do retângulo detectado e manter apenas o preto
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask_final = criar_mascara_preto(gray, rect_box)
    if mask_final is not None:
        cv2.imwrite("gabarito_mascara_preto.png", mask_final)
        cv2.imshow("Mascara Preto", mask_final)
        cv2.waitKey(0)
        cv2.destroyAllWindows()