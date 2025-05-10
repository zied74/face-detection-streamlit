import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Titre de l'application
st.title("🔍 Détection de visages - Algorithme de Viola-Jones")

# Instructions d'utilisation
st.markdown("""
## ℹ️ Instructions :
1. Téléchargez une image contenant un ou plusieurs visages.
2. Choisissez la couleur du rectangle autour des visages détectés.
3. Ajustez les paramètres `scaleFactor` et `minNeighbors` pour optimiser la détection.
4. Cliquez sur "Détecter les visages" pour afficher l'image traitée.
5. Cliquez sur "💾 Sauvegarder l'image" pour enregistrer l'image avec les visages détectés.
""")

# Chargement du classificateur de visages
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Upload d'image
uploaded_file = st.file_uploader("📤 Téléchargez une image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Lecture de l'image
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)

    # Choix de la couleur
    color = st.color_picker("🎨 Choisissez la couleur du rectangle", "#00FF00")
    color_rgb = tuple(int(color[i:i+2], 16) for i in (1, 3, 5))  # hex vers RGB

    # Ajustement des paramètres
    scaleFactor = st.slider("🔧 scaleFactor", min_value=1.05, max_value=1.5, value=1.1, step=0.05)
    minNeighbors = st.slider("🔧 minNeighbors", min_value=1, max_value=10, value=5)

    # Bouton de détection
    if st.button("🚀 Détecter les visages"):
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        st.success(f"{len(faces)} visage(s) détecté(s).")

        # Dessiner les rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(image_np, (x, y), (x + w, y + h), color_rgb, 2)

        # Affichage de l'image
        st.image(image_np, caption="Image avec visages détectés", channels="RGB")

        # Sauvegarde de l'image
        result_img = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        img_filename = "image_visages_detectes.jpg"
        cv2.imwrite(img_filename, result_img)

        with open(img_filename, "rb") as file:
            st.download_button(
                label="💾 Sauvegarder l'image",
                data=file,
                file_name=img_filename,
                mime="image/jpeg"
            )
