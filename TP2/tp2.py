import cv2
import os
import matplotlib.pyplot as plt


dossier_images = "images"
dossier_resultats = "resultats"

if not os.path.exists(dossier_resultats):
    os.makedirs(dossier_resultats)


noms_images = os.listdir(dossier_images)

for nom in noms_images:
    chemin = os.path.join(dossier_images, nom)
    image = cv2.imread(chemin, cv2.IMREAD_GRAYSCALE)

    if image is None:
        print(f"Erreur de lecture : {nom}")
        continue

    print(f"Image chargée avec succès : {nom}")

    # Affichage
    plt.imshow(image, cmap='gray')
    plt.title(nom)
    plt.axis('off')
    plt.show()

    print("\n--- APPLICATION DES FILTRES DE RÉDUCTION DU BRUIT ---\n")

for nom in noms_images:
    chemin = os.path.join(dossier_images, nom)
    image = cv2.imread(chemin, cv2.IMREAD_GRAYSCALE)

    if image is None:
        continue

    # --- Filtre moyen ---
    filtre_moyen = cv2.blur(image, (5, 5))

    # --- Filtre gaussien ---
    filtre_gaussien = cv2.GaussianBlur(image, (5, 5), 0)

    # --- Filtre médian ---
    filtre_median = cv2.medianBlur(image, 5)

    # --- Filtre min ---
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    filtre_min = cv2.erode(image, kernel)

    # --- Filtre max ---
    filtre_max = cv2.dilate(image, kernel)

    # --- Sauvegarde des résultats ---
    cv2.imwrite(os.path.join(dossier_resultats, f"moyen_{nom}"), filtre_moyen)
    cv2.imwrite(os.path.join(dossier_resultats, f"gaussien_{nom}"), filtre_gaussien)
    cv2.imwrite(os.path.join(dossier_resultats, f"median_{nom}"), filtre_median)
    cv2.imwrite(os.path.join(dossier_resultats, f"min_{nom}"), filtre_min)
    cv2.imwrite(os.path.join(dossier_resultats, f"max_{nom}"), filtre_max)

print("✅ Tous les filtres ont été appliqués et sauvegardés dans le dossier 'resultats'.")


print("\n--- DÉTECTION DES CONTOURS ---\n")

for nom in noms_images:
    chemin = os.path.join(dossier_images, nom)
    image = cv2.imread(chemin, cv2.IMREAD_GRAYSCALE)

    if image is None:
        continue

    # --- 1) Méthode par la norme du gradient (Sobel) ---
    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)

    norme_gradient = cv2.magnitude(grad_x, grad_y)
    norme_gradient = cv2.convertScaleAbs(norme_gradient)

    # Application du seuillage
    _, contour_gradient_50 = cv2.threshold(norme_gradient, 50, 255, cv2.THRESH_BINARY)
    _, contour_gradient_100 = cv2.threshold(norme_gradient, 100, 255, cv2.THRESH_BINARY)

    # Sauvegarde
    cv2.imwrite(os.path.join(dossier_resultats, f"gradient_50_{nom}"), contour_gradient_50)
    cv2.imwrite(os.path.join(dossier_resultats, f"gradient_100_{nom}"), contour_gradient_100)

    # --- 2) Méthode de Canny ---
    canny_50_150 = cv2.Canny(image, 50, 150)
    canny_100_200 = cv2.Canny(image, 100, 200)

    # Sauvegarde
    cv2.imwrite(os.path.join(dossier_resultats, f"canny_50_150_{nom}"), canny_50_150)
    cv2.imwrite(os.path.join(dossier_resultats, f"canny_100_200_{nom}"), canny_100_200)

print("✅ Détection des contours terminée et sauvegardée dans le dossier 'resultats'.")

