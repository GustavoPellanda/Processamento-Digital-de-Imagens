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

PASTA_IMAGENS = "fotos"  # Pasta onde estão as provas dos alunos
ARQUIVO_MESTRE = "Gabarito_modelo.bmp"  # O gabarito gabarito correto
WIDTH_WORK = 450
HEIGHT_WORK = 650

# DEBUG
# Mude para True para ver a imagem na tela quando ocorrer um erro de detecção
VISUALIZAR_ERROS = True


# ------------------------------------------------------------------
# FUNÇÕES
def ordenar_pontos(pontos):
    """
    ordena os 4 pontos nas direcoes:
    cima para esquerda, cima para direita, baixo para direita, baixo para esquerda]
    """
    pontos = np.array(pontos, dtype="float32")
    soma = pontos.sum(axis=1)
    dfrenca = np.diff(pontos, axis=1)

    cimaesq = pontos[np.argmin(soma)]
    baixodir = pontos[np.argmax(soma)]
    cimadir = pontos[np.argmin(dfrenca)]
    baixoesq = pontos[np.argmax(dfrenca)]

    lista_pontos = np.array([cimaesq, cimadir, baixodir, baixoesq], dtype="float32")
    return lista_pontos


def warp_quadrilateral(imgoriginal, pontos, tamanho_final=(WIDTH_WORK, HEIGHT_WORK)):
    """
    recebe a imagem original e 4 pontos do quadrilátero,
    faz a homografia e redimensiona
    """
    retangulo = ordenar_pontos(pontos)

    (cimaesq, cimadir, baixodir, baixoesq) = retangulo

    # calcula larguras e alturas
    larguraA = np.linalg.norm(baixodir - baixoesq)
    larguraB = np.linalg.norm(cimadir - cimaesq)
    altA = np.linalg.norm(cimadir - baixodir)
    altB = np.linalg.norm(cimaesq - baixoesq)

    larguramax = int(max(larguraA, larguraB))
    altmax = int(max(altA, altB))

    # faz o mapeamento pro retângulo final
    maparetangulo = np.array(
        [[0, 0], [larguramax - 1, 0], [larguramax - 1, altmax - 1], [0, altmax - 1]],
        dtype="float32",
    )

    # corrige a perspectiva
    matrizpersp = cv2.getPerspectiveTransform(retangulo, maparetangulo)
    warp = cv2.warpPerspective(imgoriginal, matrizpersp, (larguramax, altmax))

    # redimensiona
    warp_final = cv2.resize(warp, tamanho_final, interpolation=cv2.INTER_AREA)
    return warp_final


def detectar_alinhar(img):
    """
    Detecta o gabarito na imagem, encontra o retangulo principal
    e aplica homografia.
    """

    imgoriginal = img.copy()
    alt0, larg0 = imgoriginal.shape[:2]

    cinza = cv2.cvtColor(imgoriginal, cv2.COLOR_BGR2GRAY)
    cinza = cv2.GaussianBlur(cinza, (7, 7), 1.5)

    bordas = cv2.Canny(cinza, 40, 160)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    bordas = cv2.dilate(bordas, kernel, iterations=2)
    bordas = cv2.erode(bordas, kernel, iterations=1)

    cv2.imshow("Canny", bordas)
    cv2.waitKey(0)

    # contornos
    contornos, _ = cv2.findContours(bordas, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contornos:
        print("nenhum contorno encontrado")
        return None

    contornos = sorted(contornos, key=cv2.contourArea, reverse=True)

    retan = None

    # tenta achar o retangulo
    for cnt in contornos:
        area = cv2.contourArea(cnt)
        if area < (larg0 * alt0) * 0.1:
            continue

        perimetro = cv2.arcLength(cnt, True)

        for toleranciaerro in [0.01, 0.015, 0.02, 0.03]:
            approx = cv2.approxPolyDP(cnt, toleranciaerro * perimetro, True)
            if len(approx) == 4:
                retan = approx.reshape(4, 2)
                break

        if retan is not None:
            break

    if retan is None:
        print("nao foi possivel achar nenhum retangulo")
        return None

    imagem = imgoriginal.copy()
    retan_int = retan.astype(int)
    cv2.polylines(imagem, [retan_int], True, (0, 0, 255), 3)
    cv2.imshow("retangulo", imagem)
    cv2.waitKey(0)

    # homografia e redimensiona
    warp = warp_quadrilateral(imgoriginal, retan, (WIDTH_WORK, HEIGHT_WORK))
    cv2.imshow("Warp", warp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return warp


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
    thresh[h - margem_bordas : h, :] = 0
    thresh[:, w - margem_bordas : w] = 0

    # Limpeza de ruído
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    return clean


def gerar_zonas_de_acerto_perfeitas(mask_mestre_raw):
    """Gera os círculos 'alvo' baseados no gabarito mestre."""
    zonas_perfeitas = np.zeros_like(mask_mestre_raw)
    contours, _ = cv2.findContours(
        mask_mestre_raw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    RAIO_ALVO = 14

    for cnt in contours:
        if cv2.contourArea(cnt) < 50:
            continue
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

    contours_aluno, _ = cv2.findContours(
        mask_aluno, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    resultados = {i: [] for i in range(10)}
    AREA_MINIMA_ALUNO = 80

    for cnt in contours_aluno:
        if cv2.contourArea(cnt) < AREA_MINIMA_ALUNO:
            continue
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
        else:
            continue

        idx_questao = int(cY // step_y)
        if idx_questao >= 10:
            idx_questao = 9

        # Teste de colisão
        if mask_mestre_zones[cY, cX] == 255:
            resultados[idx_questao].append("C")
        else:
            resultados[idx_questao].append("E")

    nota = 0
    anuladas = 0

    for i in range(10):
        resps = resultados[i]
        if "C" in resps and "E" not in resps and len(resps) == 1:
            nota += 1
        elif ("C" in resps and "E" in resps) or len(resps) > 1:
            anuladas += 1

    return nota, anuladas


# --------------------------------------------------------------------------
# PIPELINE PRINCIPAL


def main():
    print(f"Processando imagens da pasta " "/" "fotos")

    # 1. Carregar e Preparar o Gabarito Mestre
    if not os.path.exists(ARQUIVO_MESTRE):
        print(f"ERRO: Gabarito Mestre '{ARQUIVO_MESTRE}' não encontrado.")
        return

    img_mestre = cv2.imread(ARQUIVO_MESTRE)
    warp_m = detectar_alinhar(img_mestre)

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
    for ext in ["*.png", "*.jpg", "*.jpeg", "*.bmp"]:
        lista_arquivos.extend(glob.glob(os.path.join(PASTA_IMAGENS, ext)))

    print(f">>> Encontrados {len(lista_arquivos)} arquivos para corrigir.\n")
    print(f"{'ARQUIVO':<40} | {'NOTA':<10} | {'STATUS'}")
    print("-" * 70)

    # 3. Loop de Processamento
    for arquivo_path in lista_arquivos:
        nome_arquivo = os.path.basename(arquivo_path)

        print(f"\n===== DEBUG: {nome_arquivo} =====")

        try:
            img_aluno = cv2.imread(arquivo_path)
            if img_aluno is None:
                print(f"{nome_arquivo:<40} | -          | ERRO: Imagem Corrompida")
                continue

            warp_a = detectar_alinhar(img_aluno)

            if warp_a is None:
                print(
                    f"{nome_arquivo:<40} | -          | FALHA: Gabarito não detectado"
                )
                continue

            cv2.imshow(f"{nome_arquivo} - alinhamento", warp_a)
            cv2.waitKey(0)

            # C. Processar Máscara
            mask_a_raw = processar_mascara_limpa(warp_a)

            # 🟦 Mostrar máscara do aluno
            cv2.imshow(f"{nome_arquivo} - mascara", mask_a_raw)
            cv2.waitKey(0)

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
