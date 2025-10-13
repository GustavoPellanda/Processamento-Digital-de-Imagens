#===============================================================================
# Trabalho 4 - Contagem de Arroz
#-------------------------------------------------------------------------------
# Processamento Digital de Imagens
# Universidade Tecnológica Federal do Paraná
# Professor: Bogdan T. Nassu
#-------------------------------------------------------------------------------
# Resoulução: Gustavo Finau Pellanda - 2090740 - 13/10/2025
#===============================================================================

import sys
import timeit
import numpy as np
import cv2
from statsmodels import robust  # necessário para o cálculo do MAD

IMAGEM_ENTRADA = '150.bmp' # 60 82 114 150 205
MOSTRAR_ETAPAS = False # Marcar como true para ver as imagens intermediárias do pipeline

class ContaArroz:
    """Classe que executa o pipeline de contagem de grãos de arroz."""

    def __init__(self, caminho_imagem, mostrar_etapas=False):
        self.caminho_imagem = caminho_imagem
        self.mostrar_etapas = mostrar_etapas
        self.imagem_original = None     # Armazena a imagem original (cinza normalizada)
        self.imagem_atual = None        # Armazena a imagem que será modificada nas etapas
        self.componentes = []           # Lista de componentes detectados
        self.resultado_final = None     # Imagem final com bounding boxes e contagem
        self.tempo_processamento = 0    # Tempo gasto na rotulagem e processamento
        self.contagem_final = 0         # Quantidade final estimada de grãos
        
    def _mostrar(self, titulo, imagem):
        """Exibe uma imagem se a flag de visualização estiver ativada."""
        if self.mostrar_etapas:
            cv2.imshow(titulo, imagem)

    def carregar_imagem(self):
        """Carrega a imagem original (colorida e em cinza) e normaliza a versão em cinza."""
        # Carrega imagem colorida
        img_colorida = cv2.imread(self.caminho_imagem, cv2.IMREAD_COLOR)
        if img_colorida is None:
            print('Erro ao abrir a imagem.')
            sys.exit()

        self._mostrar('01 - Imagem Original Colorida', img_colorida)

        # Converte para escala de cinza e normaliza
        self.imagem_original = cv2.cvtColor(img_colorida, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
        self.imagem_atual = self.imagem_original.copy()

        print("Imagem carregada com sucesso.")
        self._mostrar('02 - Imagem Original (Cinza Normalizada)', self.imagem_original)

    def preprocessar(self):
        """Aplica filtro gaussiano para suavizar ruídos."""
        self.imagem_atual = cv2.GaussianBlur(self.imagem_atual, (5, 5), 0)
        print("Etapa 1: Pré-processamento concluído.")
        self._mostrar('03 - Imagem Pré-processada', self.imagem_atual)

    def binarizar(self):
        """Realiza a binarização adaptativa com base na média local."""
        img_media = cv2.blur(self.imagem_atual, (99, 99))
        img_limiar = img_media + 0.2 # Define o limiar adaptativo somando o valor fixo
        # Aplica a binarização: 1.0 (branco) para grãos, 0.0 (preto) para fundo:
        self.imagem_atual = np.where(self.imagem_atual > img_limiar, 1.0, 0.0)
        print("Etapa 2: Binarização adaptativa concluída.")
        self._mostrar('04 - Imagem Binarizada', self.imagem_atual)

    def remover_ruido(self):
        """Aplica filtro de mediana para remover ruído na imagem binarizada."""
        img_uint8 = (self.imagem_atual * 255).astype(np.uint8)
        img_filtrada = cv2.medianBlur(img_uint8, 3)
        self.imagem_atual = img_filtrada.astype(np.float32) / 255.0 # Converte de volta para float32
        print("Etapa 3: Remoção de ruído concluída.")
        self._mostrar('05 - Ruído Removido', self.imagem_atual)

    def _flood_fill_iterativo(self, img, y_inicial, x_inicial, componente, mask_nao_visitados):
        """Implementação iterativa do flood fill para rotulagem."""
        altura, largura = img.shape
        pilha = [(y_inicial, x_inicial)]

        while pilha:
            y, x = pilha.pop()
            # Verifica se o pixel está dentro dos limites e é ativo não visitado
            if (y < 0 or y >= altura or x < 0 or x >= largura or 
                not mask_nao_visitados[y, x]):
                continue
            
            # Marca como visitado e rotula
            mask_nao_visitados[y, x] = False
            img[y, x] = componente['label']

            # Atualiza as dimensões do componente:
            componente['n_pixels'] += 1
            componente['T'] = min(componente['T'], y) # Menor coordenada Y
            componente['L'] = min(componente['L'], x) # Menor coordenada X
            componente['B'] = max(componente['B'], y) # Maior coordenada Y
            componente['R'] = max(componente['R'], x) # Maior coordenada X

            pilha.extend([(y-1, x), (y+1, x), (y, x-1), (y, x+1)])

    def rotular(self):
        """Executa a rotulagem de componentes conexos."""
        print("Etapa 4: Iniciando rotulagem...")
        inicio = timeit.default_timer()

        img = self.imagem_atual.copy()
        componentes = [] # lista dos "blobs" encontrados
        rotulo_indice = 2.0  # Começa em 2.0 para evitar conflito com 1.0
        altura, largura = img.shape

        # Cria uma máscara para pixels ativos (evita problemas com comparação de float)
        mask_nao_visitados = (img == 1.0)

        # Percorre cada pixel da imagem:
        for y in range(altura):
            for x in range(largura):
                # Verifica se é um pixel ativo e não visitado
                if mask_nao_visitados[y, x]:
                    # Inicializa o novo blob encontrado:
                    componente = {
                        'label': rotulo_indice,
                        'n_pixels': 0, # será incrementado pela função flood fill
                        'T': y, 'L': x, 'B': y, 'R': x  # coordenadas do retângulo
                    }
                    self._flood_fill_iterativo(img, y, x, componente, mask_nao_visitados)
                    
                    # Proteção de ruído (descarta componentes muito pequenos):
                    largura_comp = componente['R'] - componente['L'] + 1
                    altura_comp = componente['B'] - componente['T'] + 1
                    if (componente['n_pixels'] >= 50 and
                        largura_comp >= 3 and
                        altura_comp >= 3):
                        componentes.append(componente)

                    rotulo_indice += 1.0

        fim = timeit.default_timer()
        self.componentes = componentes
        self.tempo_processamento = fim - inicio
        print(f"Rotulagem concluída.\n{len(componentes)} componentes detectados.")
        print(f"Tempo de processamento: {self.tempo_processamento:.2f}s")

    def estimar(self):
        """
        Estima a quantidade de grãos com base na área dos blobs detectados.
        Utiliza mediana e MAD (Median Absolute Deviation).
        """
        if not self.componentes:
            print("Nenhum componente detectado.")
            return

        # Lista com a área (número de pixels) de cada blob
        areas_blob = np.array([c['n_pixels'] for c in self.componentes])

        if len(areas_blob) == 0:
            print("Erro: Nenhuma área encontrada para análise.")
            self.contagem_final = 0
            return

        # Estatísticas para identificar blobs de grãos individuais:
        mediana_areas = np.median(areas_blob)  # mediana das áreas
        mad_areas = robust.mad(areas_blob)     # desvio absoluto mediano

        # Limite para considerar um blob como "grão único"
        # Qualquer blob maior que isso pode conter múltiplos grãos
        fator_limite = 2.0
        limite_grao_unico = mediana_areas + fator_limite * mad_areas

        # Seleciona apenas os blobs que estão abaixo do limite, considerados "grãos sozinhos"
        candidatos_grao_unico = areas_blob[areas_blob <= limite_grao_unico]

        # Calcula a área de referência do grão
        # Se não houver candidatos, usamos a mediana geral
        if len(candidatos_grao_unico) == 0:
            print("Aviso: Nenhuma área considerada 'grão único'. Usando a mediana geral.")
            area_ref_grao = mediana_areas
        else:
            # Mediana das áreas de grãos individuais é area de referência para contagem de blobs grandes
            area_ref_grao = np.median(candidatos_grao_unico)

        # Contagem de grãos por componente:
        total_graos_estimado = 0
        for i, c in enumerate(self.componentes):
            if area_ref_grao > 0:
                # Estima quantos grãos estão naquele blob
                # Divide a área do blob pela área de um grão individual e arredonda
                num_graos_no_blob = np.round(c['n_pixels'] / area_ref_grao)
                # Garante que cada blob conte pelo menos 1 grão
                c['n_graos'] = int(max(1, num_graos_no_blob))
            else:
                c['n_graos'] = 1  # Caso a área de referência seja zero
            total_graos_estimado += c['n_graos']

        # Armazena o total final estimado de grãos:
        self.contagem_final = int(total_graos_estimado)
        print(f"Etapa 6: Estimativa concluída. \n   Grãos estimados: {self.contagem_final}\n")

    def exibir_resultado(self):
        """Exibe imagem final com bounding boxes e informações de grãos e pixels."""
        img_saida = cv2.cvtColor(self.imagem_original, cv2.COLOR_GRAY2BGR)
        
        for c in self.componentes:
            cv2.rectangle(img_saida, (c['L'], c['T']), (c['R'], c['B']), (0, 0, 1), 1)
            texto = f"G:{c['n_graos']} P:{c['n_pixels']}"  # G = grãos, P = pixels
            cv2.putText(img_saida, texto, (c['L'], c['T'] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 1))
        
        cv2.imshow('Resultado Final', img_saida)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':

    contador = ContaArroz(IMAGEM_ENTRADA, mostrar_etapas=MOSTRAR_ETAPAS)
    contador.carregar_imagem()
    contador.preprocessar()
    contador.binarizar()
    contador.remover_ruido()
    contador.rotular()
    contador.estimar()
    contador.exibir_resultado()
