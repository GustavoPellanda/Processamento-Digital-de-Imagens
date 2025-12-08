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
import os
import glob

# CONFIGURAÇÕES DO PIPELINE

PASTA_IMAGENS = "fotos"          # Pasta onde estão as provas dos alunos
ARQUIVO_MESTRE = "Gabarito_modelo.bmp" # O gabarito gabarito correto
WIDTH_WORK = 450
HEIGHT_WORK = 650

# DEBUG 
# Mude para True para ver a imagem na tela quando ocorrer um erro de detecção
VISUALIZAR_ERROS = True 

# ------------------------------------------------------------------
# FUNÇÕES
def detectar_e_alinhar(img):
    orig = img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    
    # Dilatação para fechar bordas quebradas
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    dilated = cv2.dilate(edged, kernel, iterations=2)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    rect_box = None
    
    # Tenta achar o quadrilátero
    for c in contours:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            rect_box = approx.reshape(4, 2)
            break
            
    # Se não achar quadrilátero perfeito, tenta o bounding box como fallback
    if rect_box is None:
        if len(contours) > 0:
            rect = cv2.minAreaRect(contours[0])
            box = cv2.boxPoints(rect)
            rect_box = np.int0(box)
        else:
            return None # Falha total

    # Ordenar pontos (Top-Left, Top-Right, Bottom-Right, Bottom-Left)
    rect_ord = np.zeros((4, 2), dtype="float32")
    s = rect_box.sum(axis=1)
    rect_ord[0] = rect_box[np.argmin(s)]
    rect_ord[2] = rect_box[np.argmax(s)]
    diff = np.diff(rect_box, axis=1)
    rect_ord[1] = rect_box[np.argmin(diff)]
    rect_ord[3] = rect_box[np.argmax(diff)]

    # Transformação de Perspectiva
    dst = np.array([
        [0, 0], [WIDTH_WORK-1, 0], [WIDTH_WORK-1, HEIGHT_WORK-1], [0, HEIGHT_WORK-1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect_ord, dst)
    return cv2.warpPerspective(orig, M, (WIDTH_WORK, HEIGHT_WORK))

def processar_mascara_limpa(img):
    """Limpa a imagem para deixar apenas as marcações de caneta."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    
    h, w = thresh.shape
    margem_esq = 90 
    margem_bordas = 10
    
    # Zonas mortas
    thresh[:, 0:margem_esq] = 0
    thresh[0:margem_bordas, :] = 0
    thresh[h-margem_bordas:h, :] = 0
    thresh[:, w-margem_bordas:w] = 0

    # Limpeza de ruído
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    return clean

def gerar_zonas_de_acerto_perfeitas(mask_mestre_raw):
    """Gera os círculos 'alvo' baseados no gabarito mestre."""
    zonas_perfeitas = np.zeros_like(mask_mestre_raw)
    contours, _ = cv2.findContours(mask_mestre_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    RAIO_ALVO = 14 
    
    for cnt in contours:
        if cv2.contourArea(cnt) < 50: continue
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(zonas_perfeitas, (cX, cY), RAIO_ALVO, 255, -1)
    return zonas_perfeitas

def calcular_nota(mask_mestre_zones, mask_aluno):
    """Calcula a nota baseada na colisão de centróides."""
    h, w = mask_aluno.shape
    step_y = h // 10
    
    contours_aluno, _ = cv2.findContours(mask_aluno, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    resultados = {i: [] for i in range(10)}
    AREA_MINIMA_ALUNO = 80 
    
    for cnt in contours_aluno:
        if cv2.contourArea(cnt) < AREA_MINIMA_ALUNO: continue
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else: continue
            
        idx_questao = int(cY // step_y)
        if idx_questao >= 10: idx_questao = 9
        
        # Teste de colisão
        if mask_mestre_zones[cY, cX] == 255:
            resultados[idx_questao].append('C')
        else:
            resultados[idx_questao].append('E')

    nota = 0
    anuladas = 0
    
    for i in range(10):
        resps = resultados[i]
        if 'C' in resps and 'E' not in resps and len(resps) == 1:
            nota += 1
        elif ('C' in resps and 'E' in resps) or len(resps) > 1:
            anuladas += 1
            
    return nota, anuladas


# --------------------------------------------------------------------------
# PIPELINE PRINCIPAL

def main():
    print(f"Processando imagens da pasta ""/""fotos")
    
    # 1. Carregar e Preparar o Gabarito Mestre
    if not os.path.exists(ARQUIVO_MESTRE):
        print(f"ERRO: Gabarito Mestre '{ARQUIVO_MESTRE}' não encontrado.")
        return

    img_mestre = cv2.imread(ARQUIVO_MESTRE)
    warp_m = detectar_e_alinhar(img_mestre)
    
    if warp_m is None:
        print("ERRO: Não foi possível ler o Gabarito Mestre.")
        if VISUALIZAR_ERROS:
            cv2.imshow("Erro Mestre", img_mestre)
            cv2.waitKey(0)
        return

    mask_m_raw = processar_mascara_limpa(warp_m)
    mask_master_zones = gerar_zonas_de_acerto_perfeitas(mask_m_raw)
    print(">>> Gabarito Mestre Processado com Sucesso.")

    # 2. Ler pasta de imagens
    # Pega extensões comuns
    lista_arquivos = []
    for ext in ['*.png', '*.jpg', '*.jpeg', '*.bmp']:
        lista_arquivos.extend(glob.glob(os.path.join(PASTA_IMAGENS, ext)))
    
    print(f">>> Encontrados {len(lista_arquivos)} arquivos para corrigir.\n")
    print(f"{'ARQUIVO':<40} | {'NOTA':<10} | {'STATUS'}")
    print("-" * 70)

    # 3. Loop de Processamento
    for arquivo_path in lista_arquivos:
        nome_arquivo = os.path.basename(arquivo_path)
        
        try:
            # A. Ler Imagem
            img_aluno = cv2.imread(arquivo_path)
            if img_aluno is None:
                print(f"{nome_arquivo:<40} | -          | ERRO: Imagem Corrompida")
                continue

            # B. Detectar e Alinhar
            warp_a = detectar_e_alinhar(img_aluno)
            
            # Condicional para debug
            if warp_a is None:
                print(f"{nome_arquivo:<40} | -          | FALHA: Gabarito não detectado")
                
                if VISUALIZAR_ERROS:
                    print("   -> Pressione qualquer tecla na janela para continuar...")
                    # Desenha um texto na imagem para ajudar no debug
                    debug_img = cv2.resize(img_aluno, (600, 800)) # Reduz para caber na tela se for grande
                    cv2.putText(debug_img, "FALHA NA DETECCAO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow("DEBUG - Falha Deteccao", debug_img)
                    cv2.waitKey(0)
                    cv2.destroyAllWindows()
                continue
            # ---------------------------------------

            # C. Processar Máscara
            mask_a_raw = processar_mascara_limpa(warp_a)

            # D. Calcular Nota
            nota, anuladas = calcular_nota(mask_master_zones, mask_a_raw)
            
            msg_anulada = f"({anuladas} anuladas)" if anuladas > 0 else ""
            print(f"{nome_arquivo:<40} | {nota:>2}/10      | Sucesso {msg_anulada}")

        except Exception as e:
            print(f"{nome_arquivo:<40} | -          | ERRO CRÍTICO: {e}")
            if VISUALIZAR_ERROS:
                 print(f"   -> Erro: {e}")
                 cv2.waitKey(0)

    print("Processamento Finalizado.")

if __name__ == "__main__":
    main()