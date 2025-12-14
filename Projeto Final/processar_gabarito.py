import cv2
import numpy as np
import os

def ordenar_pontos(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def encontrar_marcadores_cantos(imagem):
    process_img = imagem.copy() 
    gray = cv2.cvtColor(process_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0) 
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    
    cnts, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    marcadores = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 500: 
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                marcadores.append(c)

    marcadores = sorted(marcadores, key=cv2.contourArea, reverse=True)[:4]
    if len(marcadores) != 4: return None 
    pontos = []
    for m in marcadores:
        M = cv2.moments(m)
        if M["m00"] != 0:
            pontos.append([int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])])
            
    return np.array(pontos, dtype="float32")

# --- LÓGICA DO GABARITO MESTRE ---
def criar_gabarito_mestre(path_preenchido, path_branco, pasta_saida):
    img_preenchida = cv2.imread(path_preenchido)
    img_branca = cv2.imread(path_branco)
    
    if img_preenchida is None or img_branca is None:
        print("Erro: Não foi possível carregar um dos gabaritos mestres.")
        return None, None
    
    gray_preenchida = cv2.cvtColor(img_preenchida, cv2.COLOR_BGR2GRAY)
    gray_branca = cv2.cvtColor(img_branca, cv2.COLOR_BGR2GRAY)
    
    diferenca = cv2.absdiff(gray_branca, gray_preenchida)
    
    _, mask_mestre = cv2.threshold(diferenca, 50, 255, cv2.THRESH_BINARY)
    
    # Limpeza da máscara mestre
    kernel_limpeza = np.ones((3, 3), np.uint8)
    mask_mestre = cv2.morphologyEx(mask_mestre, cv2.MORPH_OPEN, kernel_limpeza)
    kernel_dilatacao = np.ones((3, 3), np.uint8)
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
    img_ref, mask_mestre = criar_gabarito_mestre(gabarito_mestre_preenchido, gabarito_mestre_branco, pasta_saida)
    
    if img_ref is None: return

    (h_ref, w_ref) = img_ref.shape[:2]
    
    pts_ref = encontrar_marcadores_cantos(img_ref)
    if pts_ref is None: return
    pts_ref = ordenar_pontos(pts_ref)

    if not os.path.exists(pasta_fotos): return

    arquivos = [f for f in os.listdir(pasta_fotos) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    for arquivo in arquivos:
        print(f"\n--- Processando: {arquivo} ---")
        path_img = os.path.join(pasta_fotos, arquivo)
        
        img_foto_raw = cv2.imread(path_img)
        if img_foto_raw is None: continue

        try:
            img_foto = img_foto_raw.copy()

            # Homografia e Alinhamento
            pts_foto = encontrar_marcadores_cantos(img_foto)
            if pts_foto is None:
                print(f"Pular {arquivo}: Cantos não detectados.")
                continue
            
            # DEBUG 01
            debug_cantos = img_foto.copy()
            for p in pts_foto: cv2.circle(debug_cantos, (int(p[0]), int(p[1])), 20, (0, 0, 255), -1)
            cv2.imwrite(f"{pasta_saida}01_Cantos_{arquivo}", debug_cantos)

            pts_foto = ordenar_pontos(pts_foto)
            M = cv2.getPerspectiveTransform(pts_foto, pts_ref)
            img_alinhada = cv2.warpPerspective(img_foto, M, (w_ref, h_ref))
            
            # DEBUG 02
            cv2.imwrite(f"{pasta_saida}02_Alinhada_{arquivo}", img_alinhada)
            
            # Binarização
            gray_alinhada = cv2.cvtColor(img_alinhada, cv2.COLOR_BGR2GRAY)
            
            # DEBUG 03
            cv2.imwrite(f"{pasta_saida}03_Gray_{arquivo}", gray_alinhada)
            
            # Adaptive Threshold
            # Block size 25 pega bem as bolinhas.
            thresh_diff = cv2.adaptiveThreshold(
                gray_alinhada, 255, 
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                cv2.THRESH_BINARY_INV, 25, 15
            )
            cv2.imwrite(f"{pasta_saida}05_Thresh_Adaptive_{arquivo}", thresh_diff)
            
            
            # Transforma "riscos de caneta" em "mancha sólida"
            kernel_fechar = np.ones((7, 7), np.uint8) # Aumentei um pouco para garantir fechamento
            thresh_limpo = cv2.morphologyEx(thresh_diff, cv2.MORPH_CLOSE, kernel_fechar)
            
            # O kernel deve ser MAIOR que a grossura da fonte impressa.
            # Se a fonte tem 3px de grossura, usamos 5x5 para apagá-la.
            kernel_abrir = np.ones((5, 5), np.uint8) 
            thresh_limpo = cv2.morphologyEx(thresh_limpo, cv2.MORPH_OPEN, kernel_abrir)
            
            # DEBUG 06: Verifique se as letras sumiram aqui!
            cv2.imwrite(f"{pasta_saida}06_Thresh_Clean_{arquivo}", thresh_limpo)
            
            # Verificação Final e Contagem
            # A máscara mestre diz ONDE olhar. A imagem limpa diz SE tem tinta.
            verificacao_visual = cv2.bitwise_and(thresh_limpo, thresh_limpo, mask=mask_mestre)
            cv2.imwrite(f"{pasta_saida}07_Isoladas_{arquivo}", verificacao_visual)
            
            contornos_zonas, _ = cv2.findContours(mask_mestre.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contornos_zonas = sorted(contornos_zonas, key=lambda c: cv2.boundingRect(c)[1])

            total_marcadas = 0
            img_resultado = img_alinhada.copy()

            for cnt in contornos_zonas:
                mask_unica = np.zeros(mask_mestre.shape, dtype="uint8")
                cv2.drawContours(mask_unica, [cnt], -1, 255, -1)
                
                # Conta pixels brancos na imagem JÁ LIMPA
                pixels_tinta = cv2.countNonZero(cv2.bitwise_and(thresh_limpo, thresh_limpo, mask=mask_unica))
                
                x, y, w, h = cv2.boundingRect(cnt)
                
                # CALIBRAÇÃO DO LIMIAR
                # Com a limpeza Kernel 5x5
                # Uma bolinha pintada deve dar > 350.
                limiar_pixel = 350 
                
                if pixels_tinta > limiar_pixel: 
                    total_marcadas += 1
                    cv2.rectangle(img_resultado, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    # Escreve qtd de pixels em VERDE (Acerto)
                    cv2.putText(img_resultado, str(pixels_tinta), (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)
                else:
                    cv2.rectangle(img_resultado, (x, y), (x+w, y+h), (0, 0, 255), 1)
                    # Escreve qtd de pixels em VERMELHO
                    if pixels_tinta > 0:
                        cv2.putText(img_resultado, str(pixels_tinta), (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,255), 1)

            print(f"--> Marcas corretas detectadas: {total_marcadas}")
            cv2.imwrite(f"{pasta_saida}08_FINAL_{arquivo}", img_resultado)

        except Exception as e:
            print(f"Erro crítico em {arquivo}: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    processar_fotos()