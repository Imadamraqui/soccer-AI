# soccer-AI ⚽

Ce projet est une application de pointe conçue pour l'analyse automatisée de matchs de football à l'aide de l'intelligence artificielle. Il permet de transformer des vidéos de caméras tactiques en données exploitables via une interface interactive Streamlit.

## 🌟 Contexte
L'application permet d'uploader des vidéos de matchs, de détecter les joueurs, le ballon, l'arbitre et les gardiens de but. Grâce à une phase de calibration des couleurs basée sur l'espace colorimétrique LAB, l'utilisateur peut assigner précisément les joueurs à leurs équipes respectives. Le système génère ensuite une vidéo annotée avec des statistiques détaillées :
- **Possession de balle** et répartition spatiale.
- **Vitesse du ballon** (moyenne, max) et distance parcourue.
- **Détection des tirs** et mouvements offensifs/défensifs.

---

## 🏗️ Fichiers d'Entraînement (`train_...`)
Les modèles utilisés par l'application ont été entraînés spécifiquement pour le contexte du football :
- **`train_pitch_keypoint_detector.ipynb`** : Notebook dédié à l'entraînement du modèle de détection des points clés du terrain, essentiel pour la transformation homographique et la projection des données sur un plan 2D.
- **`train_player_detector.ipynb`** : Notebook pour l'entraînement du modèle de détection des joueurs, optimisé pour distinguer les acteurs du jeu dans diverses conditions de prise de vue.

---

## 🔬 Fichier Vamelio
- **`vaamelio (3).ipynb`** : Ce fichier représente une étape majeure d'amélioration (Amélioration/Vamelio) du pipeline de détection. Il contient des expérimentations avancées sur l'intégration de la bibliothèque `roboflow/sports`, le tracking des joueurs et l'affinage des algorithmes de détection sur des extraits de matchs réels (comme Liverpool vs Real Madrid). C'est le laboratoire de recherche pour les futures versions de l'outil.

---

