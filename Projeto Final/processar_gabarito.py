import cv2
import numpy as np
import os

# --- Parâmetros de Calibração ---
# Kernel para fechar (juntar) riscos de caneta em manchas sólidas (Maior que o kernel de abertura)
KERNEL_FECHAR_SIZE = 7
# Kernel para abrir (remover) ruído fino, como as letras A, B, C, D, E (Menor que a marcação)
KERNEL_ABRIR_SIZE = 5
# Limiar para considerar a bolha como "marcada" (calibrar conforme o tamanho da bolha e resolução)
LIMIAR_PIXEL_MARCADO = 320
# Block Size e C para o Adaptive Threshold (ajustar para focar bem nas marcações)
ADAPTIVE_BLOCK_SIZE = 25
ADAPTIVE_C = 15


def ordenar_pontos(pts):
    """Ordena os 4 pontos: TL, TR, BR, BL (superior esquerdo, superior direito, inferior direito, inferior esquerdo)."""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Superior Esquerdo (menor soma x+y)
    rect[2] = pts[np.argmax(s)]  # Inferior Direito (maior soma x+y)
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Superior Direito (menor diferença y-x)
    rect[3] = pts[np.argmax(diff)]  # Inferior Esquerdo (maior diferença y-x)
    return rect


def encontrar_marcadores_cantos(imagem):
    """Detecta 4 marcadores quadrados usando contornos e aproximação."""
    process_img = imagem.copy()
    gray = cv2.cvtColor(process_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Inverte para que os marcadores pretos fiquem brancos (para cv2.findContours)
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)

    # Encontra contornos externos
    cnts, _ = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    marcadores = []

    for c in cnts:
        area = cv2.contourArea(c)
        # Filtra por área mínima para ignorar ruído
        if area > 500:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            # Apenas contornos com 4 vértices (quadrados/retângulos)
            if len(approx) == 4:
                marcadores.append(c)

    # Pega os 4 maiores marcadores encontrados
    marcadores = sorted(marcadores, key=cv2.contourArea, reverse=True)[:4]

    if len(marcadores) != 4:
        return None

    pontos = []
    # Calcula o centro (momento) de cada marcador
    for m in marcadores:
        M = cv2.moments(m)
        if M["m00"] != 0:
            pontos.append([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])

    return np.array(pontos, dtype="float32")


# --- LÓGICA DO GABARITO MESTRE ---
def criar_gabarito_mestre(path_preenchido, path_branco, pasta_saida):
    """Cria a máscara mestra pela diferença entre o gabarito preenchido e o em branco."""
    img_preenchida = cv2.imread(path_preenchido)
    img_branca = cv2.imread(path_branco)

    if img_preenchida is None or img_branca is None:
        print("Erro: Não foi possível carregar um dos gabaritos mestres.")
        return None, None

    gray_preenchida = cv2.cvtColor(img_preenchida, cv2.COLOR_BGR2GRAY)
    gray_branca = cv2.cvtColor(img_branca, cv2.COLOR_BGR2GRAY)

    # Calcula a diferença absoluta entre as duas imagens
    diferenca = cv2.absdiff(gray_branca, gray_preenchida)

    # Suaviza a diferença para remover ruído de impressão antes de limiarizar
    diferenca_blurred = cv2.GaussianBlur(diferenca, (3, 3), 0)

    # Limiariza para gerar a máscara das bolhas (regiões de interesse)
    _, mask_mestre = cv2.threshold(diferenca_blurred, 50, 255, cv2.THRESH_BINARY)

    # Limpeza final da máscara mestre (abertura para remover pequenos ruídos da diferença)
    kernel_limpeza = np.ones((3, 3), np.uint8)
    mask_mestre = cv2.morphologyEx(mask_mestre, cv2.MORPH_OPEN, kernel_limpeza)

    # Dilatação para garantir que a máscara cubra toda a bolha
    kernel_dilatacao = np.ones((5, 5), np.uint8)
    mask_mestre = cv2.dilate(mask_mestre, kernel_dilatacao, iterations=2)

    cv2.imwrite(f"{pasta_saida}00_Ref_Final_Mask.png", mask_mestre)

    return img_preenchida, mask_mestre


def processar_fotos():
    pasta_fotos = "./fotos/"
    pasta_saida = "./saida/"

    gabarito_mestre_preenchido = "Gabarito_mestre.bmp"
    gabarito_mestre_branco = "Gabarito_em_branco.bmp"

    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)

    # GERAÇÃO DA MÁSCARA MESTRE
    print("Gerando máscara mestre...")
    img_ref, mask_mestre = criar_gabarito_mestre(
        gabarito_mestre_preenchido, gabarito_mestre_branco, pasta_saida
    )

    if img_ref is None or mask_mestre is None:
        return

    (h_ref, w_ref) = img_ref.shape[:2]

    # Detecta e ordena os pontos de referência (para alinhamento)
    pts_ref = encontrar_marcadores_cantos(img_ref)
    if pts_ref is None:
        print(
            "Erro: Não foi possível detectar os marcadores do gabarito de referência."
        )
        return
    pts_ref = ordenar_pontos(pts_ref)

    if not os.path.exists(pasta_fotos):
        return

    arquivos = [
        f
        for f in os.listdir(pasta_fotos)
        if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp"))
    ]

    for arquivo in arquivos:
        print(f"\n--- Processando: {arquivo} ---")
        path_img = os.path.join(pasta_fotos, arquivo)

        img_foto_raw = cv2.imread(path_img)
        if img_foto_raw is None:
            continue

        try:
            img_foto = img_foto_raw.copy()

            # Homografia e Alinhamento
            pts_foto = encontrar_marcadores_cantos(img_foto)
            if pts_foto is None:
                print(f"Pular {arquivo}: Cantos não detectados.")
                continue

            # DEBUG 01
            debug_cantos = img_foto.copy()
            for p in pts_foto:
                cv2.circle(debug_cantos, (int(p[0]), int(p[1])), 20, (0, 0, 255), -1)
            cv2.imwrite(f"{pasta_saida}01_Cantos_{arquivo}", debug_cantos)

            # Mapeamento e Alinhamento
            pts_foto = ordenar_pontos(pts_foto)
            M = cv2.getPerspectiveTransform(pts_foto, pts_ref)
            img_alinhada = cv2.warpPerspective(img_foto, M, (w_ref, h_ref))

            # DEBUG 02
            cv2.imwrite(f"{pasta_saida}02_Alinhada_{arquivo}", img_alinhada)

            # Binarização
            gray_alinhada = cv2.cvtColor(img_alinhada, cv2.COLOR_BGR2GRAY)

            # APLICAÇÃO DE FILTRO ANTES DO ADAPTIVE THRESHOLD (melhora a estabilidade)
            gray_blur = cv2.medianBlur(gray_alinhada, 3)

            # Adaptive Threshold (Invertido: marcação preta vira branca)
            thresh_diff = cv2.adaptiveThreshold(
                gray_blur,
                255,  # Usa a imagem suavizada
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                ADAPTIVE_BLOCK_SIZE,
                ADAPTIVE_C,
            )
            cv2.imwrite(f"{pasta_saida}05_Thresh_Adaptive_{arquivo}", thresh_diff)

            # 1. Fechamento (MORPH_CLOSE): Junta pequenos riscos em manchas sólidas
            kernel_fechar = np.ones((KERNEL_FECHAR_SIZE, KERNEL_FECHAR_SIZE), np.uint8)
            thresh_limpo = cv2.morphologyEx(thresh_diff, cv2.MORPH_CLOSE, kernel_fechar)

            # 2. Abertura (MORPH_OPEN): Remove o ruído fino (letras A, B, C...)
            # ESTA É A ETAPA QUE LIMPA AS LETRAS
            kernel_abrir = np.ones((KERNEL_ABRIR_SIZE, KERNEL_ABRIR_SIZE), np.uint8)
            thresh_limpo = cv2.morphologyEx(thresh_limpo, cv2.MORPH_OPEN, kernel_abrir)

            # DEBUG 06: Imagem limpa para verificação (deve ter apenas as marcações)
            cv2.imwrite(f"{pasta_saida}06_Thresh_Clean_{arquivo}", thresh_limpo)

            # Verificação Final e Contagem
            # O gabarito (mask_mestre) diz ONDE olhar. A imagem limpa (thresh_limpo) diz SE tem tinta.
            verificacao_visual = cv2.bitwise_and(
                thresh_limpo, thresh_limpo, mask=mask_mestre
            )
            cv2.imwrite(f"{pasta_saida}07_Isoladas_{arquivo}", verificacao_visual)

            # Encontra as zonas de contorno (as bolhas) na máscara MESTRE
            contornos_zonas, _ = cv2.findContours(
                mask_mestre.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )

            # Ordena as zonas pela coordenada Y (de cima para baixo)
            contornos_zonas = sorted(
                contornos_zonas, key=lambda c: cv2.boundingRect(c)[1]
            )

            total_marcadas = 0
            img_resultado = img_alinhada.copy()

            for i, cnt in enumerate(contornos_zonas):
                # Cria uma máscara para a bolha atual (Máscara única)
                mask_unica = np.zeros(mask_mestre.shape, dtype="uint8")
                cv2.drawContours(mask_unica, [cnt], -1, 255, -1)

                # Faz a intersecção (AND) da Máscara do Aluno (thresh_limpo) com a Máscara única da bolha
                pixels_tinta = cv2.countNonZero(
                    cv2.bitwise_and(thresh_limpo, thresh_limpo, mask=mask_unica)
                )

                x, y, w, h = cv2.boundingRect(cnt)

                if pixels_tinta > LIMIAR_PIXEL_MARCADO:
                    total_marcadas += 1
                    # Desenha retângulo verde (Acerto)
                    cv2.rectangle(img_resultado, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(
                        img_resultado,
                        str(pixels_tinta),
                        (x + 5, y + 15),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (0, 255, 0),
                        1,
                    )
                else:
                    # Desenha retângulo vermelho (Não marcado ou marcação fraca)
                    cv2.rectangle(img_resultado, (x, y), (x + w, y + h), (0, 0, 255), 1)
                    if pixels_tinta > 0:
                        cv2.putText(
                            img_resultado,
                            str(pixels_tinta),
                            (x + 5, y + 15),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.4,
                            (0, 0, 255),
                            1,
                        )

            print(f"--> Marcas corretas detectadas: {total_marcadas}")
            cv2.imwrite(f"{pasta_saida}08_FINAL_{arquivo}", img_resultado)

        except Exception as e:
            print(f"Erro crítico em {arquivo}: {e}")
            import traceback

            traceback.print_exc()


if __name__ == "__main__":
    processar_fotos()
