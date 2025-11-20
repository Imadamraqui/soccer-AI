import os
import tempfile

import streamlit as st
import cv2
import numpy as np
import supervision as sv
from inference import get_model
from streamlit_image_coordinates import streamlit_image_coordinates

from pipeline import extract_jersey_lab, run_pipeline


# =========================================================
# 1) Chargement modèle Roboflow (players / ball / referee / GK)
# =========================================================

@st.cache_resource
def load_player_model():
    return get_model(
        "football-players-detection-3zvbc/11",       # ton workspace Roboflow
        api_key=st.secrets["ROBOFLOW_API_KEY"]       # à mettre dans .streamlit/secrets.toml
    )


# =========================================================
# 2) Extraction frame & crops (version C6 adaptée)
# =========================================================

def get_frame_at_index(video_path: str, frame_index: int):
    """
    Récupère la frame numero `frame_index` via la même méthode que le notebook :
    sv.get_video_frames_generator => cohérence couleurs avec C13.
    """
    frame_gen = sv.get_video_frames_generator(video_path, stride=1)
    for i, frame in enumerate(frame_gen):
        if i == frame_index:
            return frame
    return None


def extract_crops_from_frame(frame: np.ndarray, model):
    """
    Version C6 : applique le modèle sur une frame et retourne (detections, crops).
    """
    # Détections Roboflow
    result = model.infer(frame, confidence=0.3)[0]
    det = sv.Detections.from_inference(result).with_nms(0.5, True)

    # Crops RGB
    crops = [sv.crop_image(frame, xyxy) for xyxy in det.xyxy]
    return det, crops


# =========================================================
# 3) Helpers couleurs (LAB -> HEX) + refs
# =========================================================

def lab_to_hex(lab_vec: np.ndarray) -> str:
    """
    Convertit un vecteur LAB (OpenCV) en couleur HEX.
    On crée un pixel 1x1 en LAB puis conversion vers RGB.
    """
    lab_pixel = np.uint8([[lab_vec]])   # shape (1,1,3)
    rgb_pixel = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2RGB)
    r, g, b = map(int, rgb_pixel[0, 0])
    return f"#{r:02x}{g:02x}{b:02x}"


def compute_mean_lab_for_indices(crops, indices, mask_grass=True):
    """
    Reproduit la logique de C9 : moyenne de extract_jersey_lab sur les indices donnés.
    Retourne un vecteur LAB moyen ou None.
    """
    vecs = []
    for i in indices:
        if 0 <= i < len(crops):
            lab = extract_jersey_lab(crops[i], mask_grass=mask_grass)
            if lab is not None:
                vecs.append(lab)

    if len(vecs) == 0:
        return None

    return np.mean(np.stack(vecs, axis=0), axis=0)


# =========================================================
# 4) Interface Streamlit principale
# =========================================================

def main():
    st.set_page_config(
        page_title="AI Soccer – Team Color Assignment & Analysis",
        layout="wide"
    )

    st.title("⚽ AI Soccer – Color Calibration & Video Analysis")

    # -------------------------
    #  Upload vidéo
    # -------------------------
    file = st.file_uploader("📥 Upload match video (tactical camera)", type=["mp4", "mov", "avi", "m4v", "asf"])
    if not file:
        st.info("Veuillez uploader une vidéo pour commencer.")
        st.stop()

    # Sauvegarde dans un fichier temporaire persistant
    if "video_path" not in st.session_state or st.session_state.get("video_name") != file.name:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp.write(file.read())
        tmp.flush()
        st.session_state.video_path = tmp.name
        st.session_state.video_name = file.name

        # Reset des états liés à cette vidéo
        st.session_state.crops = None
        st.session_state.det_frame_index = None
        st.session_state.selected_index = None
        st.session_state.assignments = {
            "team_A": [],
            "team_B": [],
            "referee": [],
            "team_A_GK": [],
            "team_B_GK": [],
        }
        st.session_state.color_hex = {
            "team_A": None,
            "team_B": None,
            "referee": None,
            "team_A_GK": None,
            "team_B_GK": None,
        }

    video_path = st.session_state.video_path

    # Affichage vidéo originale
    st.subheader("🎥 Vidéo originale")
    st.video(video_path)

    # -------------------------
    #  Chargement modèle
    # -------------------------
    model = load_player_model()

    # -------------------------
    #  Sélection frame pour les crops
    # -------------------------
    st.markdown("---")
    st.subheader("🎞 Sélection de la frame pour calibrer les couleurs")

    vi = sv.VideoInfo.from_video_path(video_path)
    total_frames = int(vi.total_frames) if vi.total_frames is not None else 300

    frame_index = st.slider(
        "Frame pour extraire les joueurs",
        min_value=0,
        max_value=max(0, total_frames - 1),
        value=3,
        step=1
    )

    # Bouton pour extraire les crops
    if st.button("🔍 Extraire les crops de cette frame"):
        frame = get_frame_at_index(video_path, frame_index)
        if frame is None:
            st.error("Impossible de lire cette frame.")
        else:
            with st.spinner("Détection des joueurs et extraction des crops..."):
                det, crops = extract_crops_from_frame(frame, model)

            st.session_state.crops = crops
            st.session_state.det_frame_index = frame_index
            st.session_state.selected_index = None

            st.success(f"✅ {len(crops)} crops extraits sur la frame {frame_index}.")

    crops = st.session_state.get("crops", None)

    if crops is None or len(crops) == 0:
        st.info("Aucun crop encore extrait. Choisissez une frame et cliquez sur « Extraire les crops ».")
        st.stop()

    # Afficher la frame utilisée (optionnel)
    st.caption(f"Frame utilisée pour les crops : {st.session_state.det_frame_index}")

    # -------------------------
    #  Grille de crops cliquables
    # -------------------------
    st.markdown("### 🧩 Sélectionner un joueur en cliquant sur un crop")

    crop_w, crop_h = 60, 80
    crops_resized = [cv2.resize(c, (crop_w, crop_h)) for c in crops]

    # Construction de la grille (8 par ligne)
    per_row = 8
    rows = (len(crops_resized) + per_row - 1) // per_row
    grid_rows = []

    for r in range(rows):
        row_imgs = crops_resized[r*per_row:(r+1)*per_row]
        # padding si dernière ligne incomplète
        while len(row_imgs) < per_row:
            row_imgs.append(np.ones((crop_h, crop_w, 3), dtype=np.uint8) * 255)
        row_concat = cv2.hconcat(row_imgs)
        grid_rows.append(row_concat)

    grid_img = cv2.vconcat(grid_rows)
    # grid_img est déjà en RGB si source l'est; sinon, pas grave, c'est juste visuel
    click = streamlit_image_coordinates(grid_img, key="crop_grid")

    if "selected_index" not in st.session_state:
        st.session_state.selected_index = None

    selected_index = st.session_state.selected_index

    if click:
        r = click["y"] // crop_h
        c = click["x"] // crop_w
        idx = r * per_row + c
        if idx < len(crops):
            selected_index = idx
            st.session_state.selected_index = idx

    st.write("📌 Index crop sélectionné :", selected_index)

    # =====================================================
    #  Assignation des crops aux équipes / arbitres / GK
    # =====================================================

    if "assignments" not in st.session_state:
        st.session_state.assignments = {
            "team_A": [],
            "team_B": [],
            "referee": [],
            "team_A_GK": [],
            "team_B_GK": [],
        }

    if "color_hex" not in st.session_state:
        st.session_state.color_hex = {
            "team_A": None,
            "team_B": None,
            "referee": None,
            "team_A_GK": None,
            "team_B_GK": None,
        }

    assignments = st.session_state.assignments
    color_hex = st.session_state.color_hex

    def assign_to_group(group_key: str):
        idx = st.session_state.selected_index
        if idx is None:
            st.warning("Veuillez d'abord cliquer sur un crop.")
            return

        if idx not in assignments[group_key]:
            assignments[group_key].append(idx)

        # Couleur moyenne du maillot via extract_jersey_lab (C9)
        lab = extract_jersey_lab(crops[idx], mask_grass=(group_key not in ["team_A_GK", "team_B_GK"]))
        if lab is not None:
            color_hex[group_key] = lab_to_hex(lab)
        st.session_state.assignments = assignments
        st.session_state.color_hex = color_hex

    st.markdown("### 🎨 Assigner le crop sélectionné à un groupe")

    colA, colB, colR, colGA, colGB = st.columns(5)

    with colA:
        if st.button("Team A"):
            assign_to_group("team_A")
        if color_hex["team_A"]:
            st.markdown(
                f"<div style='width:35px;height:35px;background:{color_hex['team_A']};"
                f"border-radius:6px;border:1px solid #555;'></div>",
                unsafe_allow_html=True
            )

    with colB:
        if st.button("Team B"):
            assign_to_group("team_B")
        if color_hex["team_B"]:
            st.markdown(
                f"<div style='width:35px;height:35px;background:{color_hex['team_B']};"
                f"border-radius:6px;border:1px solid #555;'></div>",
                unsafe_allow_html=True
            )

    with colR:
        if st.button("Referee"):
            assign_to_group("referee")
        if color_hex["referee"]:
            st.markdown(
                f"<div style='width:35px;height:35px;background:{color_hex['referee']};"
                f"border-radius:6px;border:1px solid #555;'></div>",
                unsafe_allow_html=True
            )

    with colGA:
        if st.button("GK A"):
            assign_to_group("team_A_GK")
        if color_hex["team_A_GK"]:
            st.markdown(
                f"<div style='width:35px;height:35px;background:{color_hex['team_A_GK']};"
                f"border-radius:6px;border:1px solid #555;'></div>",
                unsafe_allow_html=True
            )

    with colGB:
        if st.button("GK B"):
            assign_to_group("team_B_GK")
        if color_hex["team_B_GK"]:
            st.markdown(
                f"<div style='width:35px;height:35px;background:{color_hex['team_B_GK']};"
                f"border-radius:6px;border:1px solid #555;'></div>",
                unsafe_allow_html=True
            )

    st.markdown("#### 📋 Résumé des indices assignés")
    st.json(assignments)

    # =====================================================
    #  Génération des références LAB + vidéo de sortie
    # =====================================================

    st.markdown("---")
    st.subheader("📽 Générer la vidéo annotée avec la pipeline C13")

    if st.button("🚀 Calculer les références & lancer la pipeline vidéo"):
        # Vérifier qu'au moins une équipe a des exemples
        if all(len(v) == 0 for v in assignments.values()):
            st.error("Aucun crop assigné. Assignez au moins quelques joueurs à Team A / Team B / Referee / GK.")
        else:
            with st.spinner("Calcul des références couleur (LAB) à partir des crops..."):
                team_refs = {
                    "team_A":    compute_mean_lab_for_indices(crops, assignments["team_A"], mask_grass=True),
                    "team_B":    compute_mean_lab_for_indices(crops, assignments["team_B"], mask_grass=True),
                    "referee":   compute_mean_lab_for_indices(crops, assignments["referee"], mask_grass=True),
                    "team_A_GK": compute_mean_lab_for_indices(crops, assignments["team_A_GK"], mask_grass=False),
                    "team_B_GK": compute_mean_lab_for_indices(crops, assignments["team_B_GK"], mask_grass=False),
                }

            st.write("✅ Références LAB calculées :")
            st.json({k: (v.tolist() if v is not None else None) for k, v in team_refs.items()})

            # Lancer la pipeline vidéo
            with st.spinner("Analyse de la vidéo & génération de la sortie annotée (C13)..."):
                out_tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                out_tmp.close()
                output_path = out_tmp.name

                output_path = run_pipeline(
                    video_path=video_path,
                    output_path=output_path,
                    model=model,
                    team_refs=team_refs
                )

            st.success("🎉 Vidéo annotée générée !")
            st.video(output_path)

            # Bouton de téléchargement
            with open(output_path, "rb") as f:
                st.download_button(
                    label="⬇️ Télécharger la vidéo annotée",
                    data=f,
                    file_name="video_tacticalview_annotated.mp4",
                    mime="video/mp4"
                )


if __name__ == "__main__":
    main()
