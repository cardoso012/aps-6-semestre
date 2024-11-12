import os
import cv2
import inquirer
import numpy as np
from matplotlib import pyplot as plt

choices = []

for index, file in enumerate(os.listdir("exemplos/")):
    choices.append(file.replace(".BMP", ""))

questions = [
    inquirer.List(
        'filename',
        message="Escolha uma das digitais abaixo. Dica: utilize as setas para navegar e ENTER para selecionar uma das opções.",
        choices=choices,
    ),
]
choice = inquirer.prompt(questions)

best_score = 0
filename = None 
founded_image = None
keypoints_1, keypoints_2, match_points = None, None, None

# Carrega as imagens de impressões digitais
example = cv2.imread(f"exemplos/{choice['filename']}.BMP", cv2.IMREAD_GRAYSCALE)
print(f"Exemplo escolhido: {choice['filename']}")
print("Buscando correspondência...")
print("Para acompanhar o processo abra a janela que o CV2 instânciou")

for index, file in enumerate(os.listdir("banco_de_imagens/")):
    fingerprint_image = cv2.imread(f"banco_de_imagens/{file}", cv2.IMREAD_GRAYSCALE)

    # Verifica se as imagens foram carregadas corretamente
    if example is None or fingerprint_image is None:
        print("Erro ao carregar as imagens.")
        exit()

    # Cria o detector SIFT
    sift = cv2.SIFT_create()

    # Detecta os keypoints e computa os descritores
    kp1, des1 = sift.detectAndCompute(example, None)
    kp2, des2 = sift.detectAndCompute(fingerprint_image, None)

    # Verifica se foram encontrados keypoints em ambas as imagens
    if des1 is None or des2 is None:
        print("Não foi possível encontrar keypoints em uma das imagens.")
        exit()

    # Cria o objeto BFMatcher (Brute Force Matcher)
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # Realiza a correspondência dos descritores
    matches = cv2.FlannBasedMatcher({'algorithm': 1, 'tress': 10}, {}).knnMatch(des1, des2, k=2)
    mp = []

    for p, q in matches:
        if p.distance < 0.1 * q.distance:
            mp.append(p)

    keypoints = min(len(kp1), len(kp2))
    score = len(mp) / keypoints * 1000

    result = cv2.drawMatches(example, kp1, fingerprint_image, kp2, mp, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    result = cv2.resize(result, None, fx=4, fy=4)
    cv2.imshow("Busca por correspondência", result)
    cv2.waitKey(1)

    if score > best_score:
        best_score = score
        filename = file 
        founded_image = fingerprint_image
        keypoints_1, keypoints_2, match_points = kp1, kp2, mp

print(f"Melhor pontuação: {filename}")
print(f"Pontuação: {best_score}")

result = cv2.drawMatches(example, keypoints_1, founded_image,  keypoints_2, match_points, None)
result = cv2.resize(result, None, fx=4, fy=4)
cv2.imshow("Resultado", result)
cv2.waitKey(0)
cv2.destroyAllWindows()