#===============================================================================
# Exemplo: segmentação de uma imagem em escala de cinza.
#-------------------------------------------------------------------------------
# Autor: Bogdan T. Nassu
# Universidade Tecnológica Federal do Paraná
#-------------------------------------------------------------------------------
# Resoulução: Gustavo Finau Pellanda - 2090740 - 19/08/2025
#===============================================================================

import sys
import timeit
import numpy as np
import cv2

#===============================================================================

INPUT_IMAGE =  'documento-3mp.bmp'
NEGATIVO = False
THRESHOLD = 0.8
ALTURA_MIN = 3
LARGURA_MIN = 3
N_PIXELS_MIN = 50

#===============================================================================

def binariza (img, threshold):
    ''' Binarização simples por limiarização.

Parâmetros: img: imagem de entrada. Se tiver mais que 1 canal, binariza cada
              canal independentemente.
            threshold: limiar.
            
Valor de retorno: versão binarizada da img_in.'''

    img_binarizada = np.where(img >= threshold, 0.0, 1.0)

    ''' np where cria uma nova matriz (não altera a imagem inicial), 
    sendo que os pixels maiores que o limiar serão 1 (branco) e os menores 0 (preto)'''
    
    return img_binarizada

#-------------------------------------------------------------------------------

def rotula (img, largura_min, altura_min, n_pixels_min):
    '''Rotulagem usando flood fill. Marca os objetos da imagem com os valores
[0.1,0.2,etc].

Parâmetros: img: imagem de entrada E saída.
            largura_min: descarta componentes com largura menor que esta.
            altura_min: descarta componentes com altura menor que esta.
            n_pixels_min: descarta componentes com menos pixels que isso.

Valor de retorno: uma lista, onde cada item é um vetor associativo (dictionary)
com os seguintes campos:

'label': rótulo do componente.
'n_pixels': número de pixels do componente.
'T', 'L', 'B', 'R': coordenadas do retângulo envolvente de um componente conexo,
respectivamente: topo, esquerda, baixo e direita.'''

    componentes_conexos = [] # lista dos "blobs" encontrados
    rotulo_indice = 0.1 # qual dos blobs está sendo rotulado
    
    altura, largura = img.shape[:2] # shape retorna as dimensões da imagem
    # Percorre cada pixel da imagem:
    for y in range(altura):
        for x in range(largura):
            if img[y, x] == 1.0: # se o pixel for branco
                
                # Inicializa o novo blob encontrado:
                componente = {
                    'label': rotulo_indice,
                    'n_pixels': 0, # será incrementado pela função flood fill
                    'T': y, 'L': x, 'B': y, 'R': x # coordenadas do retângulo
                }
                
                flood_fill(img, x, y, componente)
                
                # Proteção de ruído (descarta componentes muito pequenos):
                largura_componente = componente['R'] - componente['L'] + 1
                altura_componente = componente['B'] - componente['T'] + 1
                if (componente['n_pixels'] >= n_pixels_min and
                    largura_componente >= largura_min and
                    altura_componente >= altura_min):
                    
                    componentes_conexos.append(componente) # Componente aprovado!
                    
                rotulo_indice += 0.1 # Aumenta o rótulo para o próximo componente

    return componentes_conexos

#-------------------------------------------------------------------------------

def flood_fill (img, x, y, componente):
    ''' Quando um pixel branco for encontrado, a função flood fill irá:
    1 -> rotula-lo com o índice correspondente ao componente atual
    2 -> incrementar o número de pixels do componente
    3 -> verificar qual das dimensões do componente o novo pixel encontrado deve incrementar
    4 -> chama a função flood fill para os 4 vizinhos do pixel atual
    Qunado mais nenhum pixel for branco ou dentro dos limites,
    encerra-se a a pilha de chamadas recursivas.'''

    altura, largura = img.shape[:2]
    # Early return se o pixel estiver fora dos limites ou se não for um pixel branco (1.0):
    if (y < 0 or y >= altura or x < 0 or x >= largura 
        or img[y, x] != 1.0):
        return
    
    # Aos pixels que dever ser rotulados:
    img[y, x] = componente['label'] # Atribui número do componente ao pixel

    # Atualiza as dinmesões do componente:
    componente['n_pixels'] += 1
    componente['T'] = min(componente['T'], y) # Menor coordenada Y
    componente['L'] = min(componente['L'], x) # Menor coordenada X
    componente['B'] = max(componente['B'], y) # Maior coordenada Y
    componente['R'] = max(componente['R'], x) # Maior coordenada X

    # Chamadas Recursivas para os vizinhos:
    flood_fill(img, x, y + 1, componente) # Abaixo
    flood_fill(img, x + 1, y, componente) # Direita
    flood_fill(img, x, y - 1, componente) # Acima
    flood_fill(img, x - 1, y, componente) # Esquerda

#===============================================================================

def main ():

    # Abre a imagem em escala de cinza.
    img = cv2.imread (INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.reshape ((img.shape [0], img.shape [1], 1))
    img = img.astype (np.float32) / 255

    # Mantém uma cópia colorida para desenhar a saída.
    img_out = cv2.cvtColor (img, cv2.COLOR_GRAY2BGR)

    # Segmenta a imagem.
    if NEGATIVO:
        img = 1 - img
    img = binariza (img, THRESHOLD)
    cv2.imshow ('01 - binarizada', img)
    cv2.imwrite ('01 - binarizada.png', img*255)

    start_time = timeit.default_timer ()
    componentes = rotula (img, LARGURA_MIN, ALTURA_MIN, N_PIXELS_MIN)
    n_componentes = len (componentes)
    print ('Tempo: %f' % (timeit.default_timer () - start_time))
    print ('%d componentes detectados.' % n_componentes)

    # Mostra os objetos encontrados.
    for c in componentes:
        cv2.rectangle (img_out, (c ['L'], c ['T']), (c ['R'], c ['B']), (0,0,1))

    cv2.imshow ('02 - out', img_out)
    cv2.imwrite ('02 - out.png', img_out*255)
    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
    main ()

#===============================================================================
