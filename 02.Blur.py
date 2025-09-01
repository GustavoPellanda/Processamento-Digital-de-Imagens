#===============================================================================
# Trabalho 2 - Blur
#-------------------------------------------------------------------------------
# Processamento Digital de Imagens
# Universidade Tecnológica Federal do Paraná
# Professor: Bogdan T. Nassu
#-------------------------------------------------------------------------------
# Resoulução: Gustavo Finau Pellanda - 2090740 - 03/09/2025
#===============================================================================

import numpy as np
import cv2
import time

INPUT_IMAGE = 'b01.bmp'
WINDOW_SIZE = (15, 15)  # (largura, altura) - deve ser ímpar

class FiltroIngenuo:
    def __init__(self, imagem):
        self.imagem = imagem.copy()
        self.altura, self.largura = imagem.shape[:2]
        self.canais = 1 if len(imagem.shape) == 2 else imagem.shape[2]
    
    def blur(self, tamanho_janela):
        """
        Implementação ingênua do filtro da média.
        Para cada pixel, percorre todos os pixels na vizinhança definida pela janela
        e calcula a média aritmética dos valores.
        """
        w, h = tamanho_janela
        w_half = w // 2  # Metade da largura da janela
        h_half = h // 2  # Metade da altura da janela
        total_pixels = w * h  # Número total de pixels na janela
        
        # Inicializar imagem de resultado com o mesmo formato da original
        if self.canais == 1:
            resultado = np.zeros_like(self.imagem, dtype=np.float32)
        else:
            resultado = np.zeros_like(self.imagem, dtype=np.float32)
        
        # Percorrer cada pixel da imagem (ignorando bordas)
        for y in range(h_half, self.altura - h_half):
            for x in range(w_half, self.largura - w_half):
                if self.canais == 1:
                    # Imagem em tons de cinza
                    soma = 0.0
                    # Percorrer vizinhança do pixel (janela w x h)
                    for j in range(-h_half, h_half + 1):
                        for i in range(-w_half, w_half + 1):
                            soma += self.imagem[y + j, x + i]
                    # Calcular média e armazenar resultado
                    resultado[y, x] = soma / total_pixels
                else:
                    # Imagem colorida - processar cada canal separadamente
                    for canal in range(self.canais):
                        soma = 0.0
                        # Percorrer vizinhança do pixel (janela w x h)
                        for j in range(-h_half, h_half + 1):
                            for i in range(-w_half, w_half + 1):
                                soma += self.imagem[y + j, x + i, canal]
                        # Calcular média e armazenar resultado
                        resultado[y, x, canal] = soma / total_pixels
        
        return resultado.astype(np.uint8)

class FiltroSeparavel:
    def __init__(self, imagem):
        self.imagem = imagem.copy()
        self.altura, self.largura = imagem.shape[:2]
        self.canais = 1 if len(imagem.shape) == 2 else imagem.shape[2]
    
    def blur(self, tamanho_janela):
        """
        Implementação do filtro da média usando a propriedade de separabilidade.
        Aplica primeiro um filtro horizontal e depois um filtro vertical.
        """
        w, h = tamanho_janela
        w_half = w // 2  # Metade da largura da janela
        h_half = h // 2  # Metade da altura da janela
        
        # Passo 1: Aplicar filtro horizontal (média ao longo das linhas)
        temp = np.zeros_like(self.imagem, dtype=np.float32)
        
        for y in range(self.altura):
            for x in range(w_half, self.largura - w_half):
                if self.canais == 1:
                    # Imagem em tons de cinza
                    soma = 0.0
                    # Calcular média ao longo da linha horizontal
                    for i in range(-w_half, w_half + 1):
                        soma += self.imagem[y, x + i]
                    temp[y, x] = soma / w
                else:
                    # Imagem colorida - processar cada canal separadamente
                    for canal in range(self.canais):
                        soma = 0.0
                        # Calcular média ao longo da linha horizontal
                        for i in range(-w_half, w_half + 1):
                            soma += self.imagem[y, x + i, canal]
                        temp[y, x, canal] = soma / w
        
        # Passo 2: Aplicar filtro vertical (média ao longo das colunas)
        resultado = np.zeros_like(self.imagem, dtype=np.float32)
        
        for y in range(h_half, self.altura - h_half):
            for x in range(w_half, self.largura - w_half):
                if self.canais == 1:
                    # Imagem em tons de cinza
                    soma = 0.0
                    # Calcular média ao longo da coluna vertical
                    for j in range(-h_half, h_half + 1):
                        soma += temp[y + j, x]
                    resultado[y, x] = soma / h
                else:
                    # Imagem colorida - processar cada canal separadamente
                    for canal in range(self.canais):
                        soma = 0.0
                        # Calcular média ao longo da coluna vertical
                        for j in range(-h_half, h_half + 1):
                            soma += temp[y + j, x, canal]
                        resultado[y, x, canal] = soma / h
        
        return resultado.astype(np.uint8)

class FiltroIntegral:
    def __init__(self, imagem):
        self.imagem = imagem.copy()
        self.altura, self.largura = imagem.shape[:2]
        self.canais = 1 if len(imagem.shape) == 2 else imagem.shape[2]
        
        # Pré-calcular imagens integrais para cada canal
        # A imagem integral I(x,y) contém a soma de todos os pixels acima e à esquerda de (x,y)
        if self.canais == 1:
            self.integral = self.calcular_integral(self.imagem)
        else:
            self.integral = []
            for canal in range(self.canais):
                self.integral.append(self.calcular_integral(self.imagem[:, :, canal]))
    
    def calcular_integral(self, canal_imagem):
        """
        Constrói uma imagem integral para um canal da imagem.
        Cada posição [y, x] na imagem integral armazena a soma acumulada
        de todos os pixels desde (0,0) até (y-1, x-1) na imagem original.
        """
        # Criar matriz integral com uma linha e coluna extras (preenchidas com zeros)
        integral = np.zeros((self.altura + 1, self.largura + 1), dtype=np.float32)
        
        # Preencher a imagem integral usando a fórmula recursiva:
        # I(x,y) = f(x,y) + I(x-1,y) + I(x,y-1) - I(x-1,y-1)
        for y in range(1, self.altura + 1):
            for x in range(1, self.largura + 1):
                integral[y, x] = (canal_imagem[y-1, x-1] + 
                                 integral[y-1, x] + 
                                 integral[y, x-1] - 
                                 integral[y-1, x-1])
        return integral
    
    def blur(self, tamanho_janela):
        """
        Calcula a média da janela utilizando quatro acessos à imagem integral: 
        pega a soma total até o canto inferior direito, subtrai a soma acima da janela e a soma antes da janela, 
        depois soma de volta a área que foi subtraída duas vezes. 
        Por fim, divide o resultado pelo número de pixels na janela para obter o valor médio.
        """
        w, h = tamanho_janela
        w_half = w // 2  # Metade da largura da janela
        h_half = h // 2  # Metade da altura da janela
        
        resultado = np.zeros_like(self.imagem, dtype=np.float32)
        
        # Processar cada pixel da imagem
        for y in range(self.altura):
            for x in range(self.largura):
                # Definir coordenadas da janela considerando bordas
                # Garantir que as coordenadas não saiam dos limites da imagem
                y1 = max(0, y - h_half)                # Topo da janela
                y2 = min(self.altura - 1, y + h_half)  # Base da janela
                x1 = max(0, x - w_half)                # Esquerda da janela
                x2 = min(self.largura - 1, x + w_half) # Direita da janela
                
                # Calcular número real de pixels na janela (importante para bordas)
                pixels_reais = (y2 - y1 + 1) * (x2 - x1 + 1)
                
                if self.canais == 1:
                    # Usar a fórmula da imagem integral para calcular a soma da região:
                    # Soma = I(y2+1, x2+1) - I(y1, x2+1) - I(y2+1, x1) + I(y1, x1)
                    soma = (self.integral[y2+1, x2+1] - 
                           self.integral[y1, x2+1] - 
                           self.integral[y2+1, x1] + 
                           self.integral[y1, x1])
                    resultado[y, x] = soma / pixels_reais
                else:
                    # Processar cada canal de cor separadamente
                    for canal in range(self.canais):
                        soma = (self.integral[canal][y2+1, x2+1] - 
                               self.integral[canal][y1, x2+1] - 
                               self.integral[canal][y2+1, x1] + 
                               self.integral[canal][y1, x1])
                        resultado[y, x, canal] = soma / pixels_reais
        
        return resultado.astype(np.uint8)

def main():
    # Carregar imagem
    try:
        imagem = cv2.imread(INPUT_IMAGE)
        if imagem is None:
            raise FileNotFoundError(f"Imagem {INPUT_IMAGE} não encontrada")
    except Exception as e:
        print(f"Erro ao carregar imagem: {e}")
        return
    
    print(f"Imagem carregada: {imagem.shape}")
    print(f"Tamanho da janela: {WINDOW_SIZE}")
    
    # OpenCV blur (para comparação)
    inicio = time.time()
    blur_cv2 = cv2.blur(imagem, WINDOW_SIZE)
    tempo_cv2 = time.time() - inicio
    print(f"OpenCV blur: {tempo_cv2:.4f} segundos")
    
    # Algoritmo ingênuo
    ingenuo = FiltroIngenuo(imagem)
    inicio = time.time()
    blur_ingenuo = ingenuo.blur(WINDOW_SIZE)
    tempo_ingenuo = time.time() - inicio
    print(f"Algoritmo ingênuo: {tempo_ingenuo:.4f} segundos")
    
    # Algoritmo separável
    separavel = FiltroSeparavel(imagem)
    inicio = time.time()
    blur_separavel = separavel.blur(WINDOW_SIZE)
    tempo_separavel = time.time() - inicio
    print(f"Algoritmo separável: {tempo_separavel:.4f} segundos")
    
    # Algoritmo com imagem integral
    integral = FiltroIntegral(imagem)
    inicio = time.time()
    blur_integral = integral.blur(WINDOW_SIZE)
    tempo_integral = time.time() - inicio
    print(f"Algoritmo integral: {tempo_integral:.4f} segundos")
    
    # Comparar resultados (ignorando bordas)
    w_half = WINDOW_SIZE[0] // 2
    h_half = WINDOW_SIZE[1] // 2
    
    # Recortar regiões centrais para comparação (onde todos os algoritmos têm valores válidos)
    regiao_comparacao = (slice(h_half, -h_half), slice(w_half, -w_half))
    if len(imagem.shape) == 3:
        regiao_comparacao = regiao_comparacao + (slice(None),)
    
    cv2_crop = blur_cv2[regiao_comparacao]
    ingenuo_crop = blur_ingenuo[regiao_comparacao]
    separavel_crop = blur_separavel[regiao_comparacao]
    integral_crop = blur_integral[regiao_comparacao]
    
    # Calcular diferenças máximas
    diff_ingenuo = np.max(np.abs(cv2_crop.astype(np.int16) - ingenuo_crop.astype(np.int16)))
    diff_separavel = np.max(np.abs(cv2_crop.astype(np.int16) - separavel_crop.astype(np.int16)))
    diff_integral = np.max(np.abs(cv2_crop.astype(np.int16) - integral_crop.astype(np.int16)))
    
    print(f"\nDiferença máxima em relação ao OpenCV:")
    print(f"Ingênuo: {diff_ingenuo}")
    print(f"Seprável: {diff_separavel}")
    print(f"Integral: {diff_integral}")
    
    # Exibir resultados
    cv2.imshow('Original', imagem)
    cv2.imshow('OpenCV Blur', blur_cv2)
    cv2.imshow('Ingenuo Blur', blur_ingenuo)
    cv2.imshow('Separavel Blur', blur_separavel)
    cv2.imshow('Integral Blur', blur_integral)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

"""
Comparação dos Resultados:

Subtrai o valor de cada pixel processado pelo OpenCV pelo pixel de mesma posição processado por 
cada um dos algoritmos e exibe a maior diferença absoluta encontrada entre todos os pixels da imagem. 
Diferenças de 1 ou 2 são consideradas normais (resultado de arredondamentos numéricos). 
Diferenças acima de 3 podem indicar erros significativos na implementação dos algoritmos.

"""
