import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# --- Création du dossier des résultats ---
os.makedirs("resultats", exist_ok=True)

# --- Fonctions ---

def afficher_histogramme(image, titre, nom_fichier):
    couleurs = ('b', 'g', 'r')
    plt.figure()
    for i, couleur in enumerate(couleurs):
        hist = cv2.calcHist([image], [i], None, [256], [0, 256])
        plt.plot(hist, color=couleur)

    plt.title(titre)
    plt.xlabel("Niveaux d'intensité")
    plt.ylabel("Nombre de pixels")
    plt.xlim([0, 256])
    plt.savefig(nom_fichier)
    plt.close()

def appliquer_egalisation(image):
    image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(image_ycrcb)
    y_egalise = cv2.equalizeHist(y)
    image_ycrcb_egalisee = cv2.merge((y_egalise, cr, cb))
    return cv2.cvtColor(image_ycrcb_egalisee, cv2.COLOR_YCrCb2BGR)

def transformation_lineaire(image, alpha=1.3, beta=20):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)

def correction_gamma(image, gamma=0.6):
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                      for i in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)

# --- Lecture des images ---
dossier_images = "images"
images = os.listdir(dossier_images)

for nom_image in images:
    if nom_image.lower().endswith((".jpg", ".png", ".jpeg")):
        print(f"Traitement de {nom_image} ...")

        chemin = os.path.join(dossier_images, nom_image)
        image = cv2.imread(chemin)

        dossier = f"resultats/{nom_image.split('.')[0]}"
        os.makedirs(dossier, exist_ok=True)

        # Original
        cv2.imwrite(f"{dossier}/originale.png", image)
        afficher_histogramme(image, "Histogramme original",
                             f"{dossier}/histogramme_original.png")

        # Egalisation
        image_eq = appliquer_egalisation(image)
        cv2.imwrite(f"{dossier}/egalisation.png", image_eq)
        afficher_histogramme(image_eq, "Histogramme égalisation",
                             f"{dossier}/histogramme_egalisation.png")

        # Transformation linéaire
        image_lin = transformation_lineaire(image)
        cv2.imwrite(f"{dossier}/lineaire.png", image_lin)
        afficher_histogramme(image_lin, "Histogramme linéaire",
                             f"{dossier}/histogramme_lineaire.png")

        # Correction gamma
        image_g = correction_gamma(image)
        cv2.imwrite(f"{dossier}/gamma.png", image_g)
        afficher_histogramme(image_g, "Histogramme gamma",
                             f"{dossier}/histogramme_gamma.png")

        print(f"{nom_image} terminé ✅\n")

print("✅ Tous les traitements du TP1 sont terminés.")
