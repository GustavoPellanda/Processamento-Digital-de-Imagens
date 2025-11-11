#===============================================================================
# Trabalho 5 - Chroma Key
#-------------------------------------------------------------------------------
# Processamento Digital de Imagens
# Universidade Tecnológica Federal do Paraná
# Professor: Bogdan T. Nassu
#-------------------------------------------------------------------------------
# Resoulução: Gustavo Finau Pellanda - 2090740 - 10/11/2025
#===============================================================================

import os
import glob
import cv2
import numpy as np
import shutil

PASTA_ENTRADA = 'img'
IMAGEM_FUNDO = 'fundo.bmp'
PASTA_SAIDA_BRANCO = 'resultados_fundo_branco'
PASTA_SAIDA_FUNDO = 'resultados_com_fundo'

# ╔═════════════════════════════════════════════════════════════════════╗
# ║ Ao executar, o script processa todas as imagens .bmp na pasta 'img'.║
# ║ Se existir um arquivo 'fundo.bmp' no mesmo diretório do código,     ║
# ║ as imagens recortadas serão coladas sobre esse fundo.               ║
# ╚═════════════════════════════════════════════════════════════════════╝

class ProcessadorChromaKey:
    """
    Pipeline para remoção de fundo verde.
    Sequência do pipeline:
    > medir verdice
    > suavizar verdice
    > remover verde em HLS
    > correção de spill
    > gerar alpha
    > encontrar bounding box
    > recortar
    > aplicar fundos
    """

    def calcular_verdice(self, imagem):
        """
        Mede o quanto cada pixel é verde comparado aos outros canais.
        Define quais pixels são considerados fundo (alta verdice) e quais 
        são foreground (baixa verdice) em um intervalo normalizado [0, 1].
        """
        img = imagem.astype(np.float32)
        B, G, R = cv2.split(img)

        # Diferença entre G e o maior dos canais concorrentes
        # Se G for muito maior que R e B, o pixel é considerado verde
        diff_verde = G - np.maximum(R, B)

        # Limites do fundo verde
        verde_fraco = 30 
        verde_forte = 80

        # Normalização linear
        verdice = (diff_verde - verde_fraco) / (verde_forte - verde_fraco)

        verdice = np.clip(verdice, 0, 1)
        return cv2.GaussianBlur(verdice, (5, 5), 0)  # Suaviza o mapa de intensidade de verde
    
    def remover_verde(self, imagem):
        """
        Reduz saturação nas regiões detectadas como verdes.
        A operação é feita em HLS (melhor para controlar saturação).
        """
        img_rgb = cv2.cvtColor(imagem, cv2.COLOR_BGR2RGB)
        hls = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HLS)

        h, l, s = cv2.split(hls)

        # Faixa típica de matiz do verde
        h_min, h_max = 35, 90
        s_min = 60
        fator = 0.1  # desaturação forte

        # Máscara pura baseada em cor real
        green_mask = (h >= h_min) & (h <= h_max) & (s > s_min)

        # Aplicação da desaturação
        s = s.astype(np.float32)
        s[green_mask] = s[green_mask] * fator
        s = np.clip(s, 0, 255).astype(np.uint8)

        hls_final = cv2.merge([h, l, s])
        img_rgb_final = cv2.cvtColor(hls_final, cv2.COLOR_HLS2RGB)
        return cv2.cvtColor(img_rgb_final, cv2.COLOR_RGB2BGR)
    
    def corrigir_spill_bgr(self, imagem_corrigida, verdice):
        """
        Correção leve baseada na verdice para reforço de bordas.
        Ajusta os canais diretamente (reduz G, reforça R e B),
        removendo resíduos de verde que ficam mesmo após a desaturação, 
        deixando bordas menos artificiais.
        """
        img = imagem_corrigida.astype(np.float32) / 255.0
        B, G, R = cv2.split(img)

        # Redução suave do verde
        G_corr = G * (1 - 0.25 * verdice)

        # Reforço moderado dos canais R e B
        media = (R + G + B) / 3.0
        R_corr = R + (media - R) * verdice * 0.15
        B_corr = B + (media - B) * verdice * 0.15

        img_corr = cv2.merge([
            np.clip(B_corr, 0, 1),
            np.clip(G_corr, 0, 1),
            np.clip(R_corr, 0, 1)
        ])
        return (img_corr * 255).astype(np.uint8)

    def gerar_alpha(self, verdice):
        """
        Gera o alpha a partir do valor da verdice.
        Quanto menos verde for o pixel, menos transparente ele será na imagem recortada.
        """
        alpha = 1 - verdice                 # alpha contínuo baseado no nível de verde
        alpha = np.clip(alpha, 0, 1)        # garantia de faixa correta
        alpha = cv2.GaussianBlur(alpha, (3, 3), 0)  # refinamento das bordas
        return alpha
    
    def encontrar_bbox(self, alpha, margem=10):
        """
        Gera uma bounding box que engloba toda a região considerada como foreground.

        A função converte o alpha contínuo em uma máscara binária, identifica todos
        os contornos presentes e calcula o retângulo mínimo que cobre a soma dessas
        regiões. Isso garante que objetos fragmentados, partes finas ou múltiplos
        elementos sejam tratados como um único bloco contínuo.
        """
        mask_bin = (alpha > 0.5).astype(np.uint8) * 255
        contornos, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contornos:
            return None

        xs, ys, xe, ye = [], [], [], []

        # Coleta das caixas individuais
        for c in contornos:
            x, y, w, h = cv2.boundingRect(c)
            xs.append(x)
            ys.append(y)
            xe.append(x + w)
            ye.append(y + h)

        # Ajuste com margem
        x_min = max(0, min(xs) - margem)
        y_min = max(0, min(ys) - margem)
        x_max = min(mask_bin.shape[1], max(xe) + margem)
        y_max = min(mask_bin.shape[0], max(ye) + margem)

        return x_min, y_min, x_max - x_min, y_max - y_min

    def aplicar_fundo(self, imagem_corrigida, alpha, fundo):
        """
        Alpha blending com fundo personalizado.
        Ajusta tamanho do fundo automaticamente.
        """
        # Redimensiona o fundo se o tamanho não corresponder ao da imagem
        if fundo.shape[:2] != imagem_corrigida.shape[:2]:
            fundo = cv2.resize(fundo, (imagem_corrigida.shape[1], imagem_corrigida.shape[0]))

        # Expande o alpha de 1 canal para 3 canais para permitir blending por pixel RGB
        alpha_3d = np.stack([alpha] * 3, axis=2)

        # Combinação dos pixels:
        comp = imagem_corrigida.astype(np.float32) * alpha_3d \
             + fundo.astype(np.float32) * (1 - alpha_3d)

        return comp.astype(np.uint8)

    def aplicar_fundo_branco(self, imagem_corrigida, alpha):
        """
        Composição com fundo branco.
        """
        fundo_branco = np.ones_like(imagem_corrigida) * 255
        return self.aplicar_fundo(imagem_corrigida, alpha, fundo_branco)
    
if __name__ == '__main__':
    # Preparar pastas de saída
    for pasta in [PASTA_SAIDA_BRANCO, PASTA_SAIDA_FUNDO]:
        if os.path.exists(pasta):
            shutil.rmtree(pasta)
        os.makedirs(pasta)
        print(f"Pasta '{pasta}' criada/limpa.")

    # Carregar fundo
    fundo_img = cv2.imread(IMAGEM_FUNDO)
    fundo_disponivel = fundo_img is not None

    # Procurar imagens na pasta de entrada
    imagens = glob.glob(os.path.join(PASTA_ENTRADA, '*.bmp'))
    if not imagens:
        print("Nenhuma imagem encontrada na pasta de entrada.")
        exit()

    processador = ProcessadorChromaKey()

    # Loop principal
    for caminho in sorted(imagens):
        nome = os.path.basename(caminho)
        print(f"\nProcessando: {nome}")

        imagem = cv2.imread(caminho)
        if imagem is None:
            print(f"  Erro ao carregar {caminho}")
            continue

        try:
            # PIPELINE:
            verdice = processador.calcular_verdice(imagem)
            imagem_corrigida = processador.remover_verde(imagem)
            imagem_corrigida = processador.corrigir_spill_bgr(imagem_corrigida, verdice)
            alpha_ref = processador.gerar_alpha(verdice)

            # Bounding box considerando todos os objetos
            bbox = processador.encontrar_bbox(alpha_ref)
            if bbox is None:
                print("  Nenhum foreground detectado.")
                continue

            x, y, w, h = bbox
            img_crop = imagem_corrigida[y:y + h, x:x + w]
            alpha_crop = alpha_ref[y:y + h, x:x + w]

            # Fundo branco
            resultado_branco = processador.aplicar_fundo_branco(img_crop, alpha_crop)
            cv2.imwrite(os.path.join(PASTA_SAIDA_BRANCO, nome), resultado_branco)

            # Fundo personalizado (se disponível)
            if fundo_disponivel:
                resultado_fundo = processador.aplicar_fundo(img_crop, alpha_crop, fundo_img)
                cv2.imwrite(os.path.join(PASTA_SAIDA_FUNDO, nome), resultado_fundo)

            print("  ✓ Imagem processada e salva com sucesso.")

        except Exception as e:
            print(f"  ✗ Erro ao processar {nome}: {e}")

    print("\n=== PROCESSAMENTO CONCLUÍDO ===")
