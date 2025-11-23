# app_poubelle.py
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import time

# === Configuration de la page ===
st.set_page_config(page_title="Système de Détection de Poubelle", page_icon="🗑️", layout="wide")

# === Titre principal ===
st.markdown("<h1 style='text-align: center; color: #1E90FF;'>🗑️ Système de Détection de Poubelle</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: black;'>Téléversez une image ou utilisez une URL pour prédire si la poubelle est vide ou pleine.</p>", unsafe_allow_html=True)
st.write("---")

# === Charger le modèle ===
model = load_model("poubelle_vide_pleine.h5") 

# === Fonction de prédiction avec confiance et temps ===
def predict_poubelle(img, model):
    start_time = time.time()
    img_resized = img.resize((224,224))
    img_array = image.img_to_array(img_resized)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)[0][0]
    end_time = time.time()
    temps = end_time - start_time
    if pred < 0.5:
        label = "Vide"
        confidence = (1 - pred) * 100
    else:
        label = "Pleine"
        confidence = pred * 100
    return label, confidence, temps

# === Colonnes principales avec séparateur vertical ===
col_deco, col_sep, col_systeme = st.columns([1, 0.05, 1])

# === Colonne gauche : Décoration avec deux images côte à côte ===
with col_deco:
    st.markdown("<h3 style='text-align: center; color: #1E90FF;'>🌿 Pourquoi un système de détection de poubelle ?</h3>", unsafe_allow_html=True)
    st.markdown(
        """
        <p style='color: black; text-align: justify;'>
        Un système de détection de poubelle permet de :
        </p>
        <ul style='color: black;'>
        <li>Améliorer la gestion des déchets en optimisant les collectes. ♻️</li>
        <li>Réduire les débordements et maintenir la propreté. 🧹</li>
        <li>Économiser du temps et des ressources pour les villes et entreprises. ⏱️</li>
        <li>Contribuer à la protection de l’environnement. 🌍</li>
        </ul>
        """, unsafe_allow_html=True
    )
    try:
        deco_col1, deco_col2 = st.columns([1, 1])
        with deco_col1:
            img1 = Image.open("poubelle.jpg")
            st.image(img1, width=200)
        with deco_col2:
            img2 = Image.open("plein.jpg")
            st.image(img2, width=200)
    except:
        st.info("Ajoutez les images 'poubelle.jpg' et 'plein.jpg' dans le dossier pour la décoration.")

# === Séparateur vertical décoré ===
with col_sep:
    st.markdown(
        "<div style='border-left:3px solid #1E90FF; height: 650px; margin-left: 10px;'></div>",
        unsafe_allow_html=True
    )

# === Colonne droite : Système de prédiction avec image et infos côte à côte ===
with col_systeme:
    st.markdown("<h3 style='text-align: center; color: #1E90FF;'>🖼️ Prédiction de l'image</h3>", unsafe_allow_html=True)
    option = st.radio("Choisissez comment téléverser l'image :", ("Depuis le PC", "Depuis URL"))

    uploaded_file = None
    if option == "Depuis le PC":
        uploaded_file = st.file_uploader("Parcourir les fichiers", type=["jpg","jpeg","png"])
    elif option == "Depuis URL":
        url = st.text_input("Entrez l'URL de l'image :")
        if url:
            try:
                response = requests.get(url)
                uploaded_file = Image.open(BytesIO(response.content))
            except:
                st.error("Impossible de charger l’image depuis l’URL.")

    # === Affichage image + prédiction ===
    if uploaded_file:
        if isinstance(uploaded_file, Image.Image):
            img = uploaded_file
        else:
            img = Image.open(uploaded_file)
        img_col, info_col = st.columns([1, 0.5])
        with img_col:
            st.image(img, caption="Image sélectionnée", width=300)
        with info_col:
            label, confidence, temps = predict_poubelle(img, model)
            st.markdown(f"<p style='color: black;'>Prédiction : <strong>{label}</strong> ✅</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: black;'>Confiance : <strong>{confidence:.2f}%</strong> 💯</p>", unsafe_allow_html=True)
            st.markdown(f"<p style='color: black;'>Temps sExecution: <strong>{temps:.3f} secondes</strong> ⏱️</p>", unsafe_allow_html=True)

    # === Bouton pour télécharger le modèle seulement ===
    st.write("---")
    st.download_button("Télécharger le modèle", data=open("poubelle_vide_pleine.h5","rb"), file_name="poubelle_vide_pleine.h5")
