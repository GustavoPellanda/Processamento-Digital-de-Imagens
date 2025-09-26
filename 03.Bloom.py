#===============================================================================
# Trabalho 3 - Bloom
#-------------------------------------------------------------------------------
# Processamento Digital de Imagens
# Universidade Tecnológica Federal do Paraná
# Professor: Bogdan T. Nassu
#-------------------------------------------------------------------------------
# Resoulução: Gustavo Finau Pellanda - 2090740 - 11/09/2025
#===============================================================================

import cv2
import numpy as np
import time

INPUT_IMAGE = 'Wind Waker GC.bmp'

def bloom_gaussiano(imagem):
    limiar_luminancia = 0.6
    
    imagem_hls = cv2.cvtColor(imagem, cv2.COLOR_BGR2HLS)
    luminancia = imagem_hls[:, :, 1]
    
    mascara_luminancia = luminancia > limiar_luminancia
    mascara_brilho = imagem * mascara_luminancia[:, :, np.newaxis]

    # cv2.imshow('mascara', mascara_brilho)
    
    mascara_sigma1 = cv2.GaussianBlur(mascara_brilho, (0, 0), 10)
    mascara_sigma2 = cv2.GaussianBlur(mascara_brilho, (0, 0), 20)
    mascara_sigma3 = cv2.GaussianBlur(mascara_brilho, (0, 0), 30)
    mascara_sigma4 = cv2.GaussianBlur(mascara_brilho, (0, 0), 40)
    mascara_sigma5 = cv2.GaussianBlur(mascara_brilho, (0, 0), 50)

    # cv2.imshow('blur', mascara_sigma5)
    
    soma_mascaras = mascara_sigma1 + mascara_sigma2 + mascara_sigma3 + mascara_sigma4 + mascara_sigma5
    soma_mascaras = np.clip(soma_mascaras, 0, 1)

    # cv2.imshow('soma', soma_mascaras)
    
    resultado = cv2.addWeighted(imagem, 0.7, soma_mascaras, 0.3, 0)
    
    return resultado

def bloom_box(imagem):
    limiar_luminancia = 0.6
    
    imagem_hls = cv2.cvtColor(imagem, cv2.COLOR_BGR2HLS)
    luminancia = imagem_hls[:, :, 1]
    
    mascara_luminancia = luminancia > limiar_luminancia
    mascara_brilho = imagem * mascara_luminancia[:, :, np.newaxis]

    # cv2.imshow('mascara', mascara_brilho)
    
    mascara_kernel1 = mascara_brilho.copy()
    for _ in range(5):
        mascara_kernel1 = cv2.blur(mascara_kernel1, (7, 7))
    
    mascara_kernel2 = mascara_brilho.copy()
    for _ in range(5):
        mascara_kernel2 = cv2.blur(mascara_kernel2, (21, 21))
    
    mascara_kernel3 = mascara_brilho.copy()
    for _ in range(5):
        mascara_kernel3 = cv2.blur(mascara_kernel3, (42, 42))
    
    mascara_kernel4 = mascara_brilho.copy()
    for _ in range(5):
        mascara_kernel4 = cv2.blur(mascara_kernel4, (63, 63))
    
    mascara_kernel5 = mascara_brilho.copy()
    for _ in range(5):
        mascara_kernel5 = cv2.blur(mascara_kernel5, (84, 84))

    # cv2.imshow('blur', mascara_kernel5)
    
    soma_mascaras = mascara_kernel1 + mascara_kernel2 + mascara_kernel3 + mascara_kernel4 + mascara_kernel5
    soma_mascaras = np.clip(soma_mascaras, 0, 1)

    # cv2.imshow('soma', soma_mascaras)
    
    resultado = cv2.addWeighted(imagem, 0.7, soma_mascaras, 0.3, 0)
    
    return resultado

if __name__ == "__main__":
    imagem = cv2.imread(INPUT_IMAGE).astype(np.float32) / 255.0
    
    inicio = time.time()
    resultado_gaussiano = bloom_gaussiano(imagem)
    fim = time.time()
    print("Tempo de execução Gaussiano: ", fim - inicio)
    
    inicio = time.time()
    resultado_box = bloom_box(imagem)
    fim = time.time()
    print("Tempo de execução Box: ", fim - inicio)
    
    cv2.imshow('Original', imagem)
    cv2.imshow('Resultado Gaussiano', resultado_gaussiano)
    cv2.imshow('Resultado Box', resultado_box)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
