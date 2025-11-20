# ============================================================
#  pipeline.py — Version complète du notebook C6/C9/C13
#  Utilisable directement depuis Streamlit
# ============================================================

import cv2
import numpy as np
import supervision as sv
from collections import deque, Counter, defaultdict

TEAM_LABELS = {
    'team_A': 0,
    'team_B': 1,
    'referee': 2,
    'team_A_GK': 3,
    'team_B_GK': 4
}

ellipse_annotator = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(["#5F9EF0", "#C24154", "#FFFF00"]),
    thickness=2
)

label_annotator = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(["#5F9EF0", "#C24154", "#FFFF00"]),
    text_color=sv.Color.from_hex('#000000'),
    text_position=sv.Position.BOTTOM_CENTER
)

triangle_annotator = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FFD700'),
    base=25,
    height=21,
    outline_thickness=1
)

tracker = sv.ByteTrack()
tracker.reset()


# ============================================================
# 1) CONSTANTES
# ============================================================

BALL_ID       = 0
GOALKEEPER_ID = 1
PLAYER_ID     = 2
REFEREE_ID    = 3

TEAM_LABELS = {
    "team_A": 0,
    "team_B": 1,
    "referee": 2,
    "team_A_GK": 3,
    "team_B_GK": 4
}

USE_AB_ONLY = False

# Reset du tracker Ballon
_last_ball = {"xyxy": None, "c": None, "v": np.array([0.0,0.0]), "miss": 0}

# ByteTrack global
byte_tracker = sv.ByteTrack()


# ============================================================
# 2) FONCTIONS UTILITAIRES COULEURS (C9)
# ============================================================

def extract_jersey_lab(crop_rgb, torso_margin=(0.25, 0.70),
                       side_margin=0.25, mask_grass=True, min_pixels=120):
    """Notebook C9 — strictement identique."""
    if crop_rgb is None or crop_rgb.size == 0:
        return None

    h, w = crop_rgb.shape[:2]
    y1 = int(h * torso_margin[0])
    y2 = int(h * torso_margin[1])
    x1 = int(w * side_margin)
    x2 = int(w * (1 - side_margin))

    if x2 <= x1 or y2 <= y1:
        return None

    roi = crop_rgb[y1:y2, x1:x2].copy()
    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
    H, S, V = cv2.split(hsv)

    keep = np.ones(roi.shape[:2], bool)

    if mask_grass:
        grass = (H >= 35) & (H <= 90) & (S > 55) & (V > 40)
        keep &= ~grass

    low_sv = (S < 35) | (V < 25)
    keep &= ~low_sv

    if keep.sum() < min_pixels:
        keep = np.ones_like(keep)

    lab = cv2.cvtColor(roi, cv2.COLOR_RGB2LAB).reshape(-1,3)
    lab = lab[keep.reshape(-1)]

    if lab.size == 0:
        return None

    mean_lab = np.mean(lab, axis=0)
    return mean_lab if not USE_AB_ONLY else mean_lab[1:]


def assign_team_with_scores(crop_rgb, refs, mask_grass=True):
    """Retourne la classe et les distances (notebook identique)."""
    col = extract_jersey_lab(crop_rgb, mask_grass=mask_grass)
    if col is None:
        for k in ["referee", "team_A", "team_B", "team_A_GK", "team_B_GK"]:
            if refs.get(k) is not None:
                return TEAM_LABELS[k], {k: 0.0}
        return TEAM_LABELS["team_A"], {}

    dcol = col if not USE_AB_ONLY else col[1:]
    dists = {}

    for name, ref in refs.items():
        if ref is None:
            continue
        dref = ref if not USE_AB_ONLY else ref[1:]
        dists[name] = float(np.linalg.norm(dcol - dref))

    assigned = min(dists, key=dists.get)
    return TEAM_LABELS[assigned], dists


def assign_team(crop_rgb, refs, *, mask_grass=True):
    lab, _ = assign_team_with_scores(crop_rgb, refs, mask_grass=mask_grass)
    return lab


# ============================================================
# 3) PITCH DETECTION
# ============================================================

def build_pitch_mask_fast(frame_rgb, h_low=35, h_high=90,
                          s_min=40, v_min=40, morph_ks=7, scale=0.33):
    H, W = frame_rgb.shape[:2]

    small = cv2.resize(frame_rgb, (int(W*scale), int(H*scale)),
                       interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    Hc, Sc, Vc = cv2.split(hsv)

    grass = (Hc >= h_low) & (Hc <= h_high) & (Sc >= s_min) & (Vc >= v_min)
    mask = (grass * 255).astype(np.uint8)

    if morph_ks > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ks, morph_ks))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)

    num, labels = cv2.connectedComponents(mask)

    if num > 1:
        areas = [(labels == i).sum() for i in range(1, num)]
        biggest = 1 + int(np.argmax(areas))
        mask = (labels == biggest).astype(np.uint8) * 255

    return cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)


def keep_if_on_pitch(dets, pitch_mask_bool, patch_px=8, min_cover=0.25):
    if len(dets.xyxy) == 0:
        return dets

    H, W = pitch_mask_bool.shape
    keep = np.zeros(len(dets.xyxy), bool)

    for i, (x1,y1,x2,y2) in enumerate(dets.xyxy.astype(int)):
        cx = int((x1+x2)/2)
        by = int(y2)

        x1p = max(0, cx - patch_px)
        x2p = min(W, cx + patch_px + 1)
        y1p = max(0, by - patch_px)
        y2p = min(H, by + patch_px + 1)

        patch = pitch_mask_bool[y1p:y2p, x1p:x2p]
        cover = patch.mean()

        keep[i] = cover >= min_cover

    return dets[keep]


# ============================================================
# 4) BALL TRACKER ROBUSTE
# ============================================================

def _iou(a, b):
    if a is None or b is None:
        return 0.0
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1, iy1 = max(ax1,bx1), max(ay1,by1)
    ix2, iy2 = min(ax2,bx2), min(ay2,by2)

    iw, ih = max(0,ix2-ix1), max(0,iy2-iy1)
    inter  = iw*ih

    area_a = (ax2-ax1)*(ay2-ay1)
    area_b = (bx2-bx1)*(by2-by1)

    union = area_a + area_b - inter + 1e-6
    return inter/union


def _center(xyxy):
    x1,y1,x2,y2 = xyxy
    return np.array([(x1+x2)/2, (y1+y2)/2], float)


def _pick_best_ball(frame_rgb, cand):
    if len(cand.xyxy) == 0:
        return None

    bd = cand.with_nms(0.35, class_agnostic=True)
    if len(bd.xyxy) == 0:
        return None

    pred_c = _last_ball["c"] + _last_ball["v"] if _last_ball["c"] is not None else None

    best, best_score = -1, -1e9

    for i, xy in enumerate(bd.xyxy):
        c = _center(xy)

        # distance from predicted
        dist = np.linalg.norm(c - pred_c) if pred_c is not None else 0.0
        dist_term = -dist

        iou_score = _iou(_last_ball["xyxy"], xy)

        # couleur ballon
        crop = sv.crop_image(frame_rgb, xy)
        hsv = cv2.cvtColor(crop, cv2.COLOR_RGB2HSV)
        H,S,V = cv2.split(hsv)
        white  = (S<60)&(V>140)
        orange = (H>=5)&(H<=30)&(S>90)&(V>90)
        color_score = float((white|orange).mean())

        conf = float(bd.confidence[i]) if bd.confidence is not None else 0.0

        score = 1.5*conf + 0.7*iou_score + 0.3*color_score + 0.02*dist_term

        if score > best_score:
            best_score = score
            best = i

    if best < 0:
        return None

    return bd[best:best+1]


def track_ball_robust(ball_dets, frame_rgb):
    global _last_ball

    chosen = _pick_best_ball(frame_rgb, ball_dets)

    if chosen is not None and len(chosen.xyxy) > 0:
        xy = chosen.xyxy[0]
        c = _center(xy)

        if _last_ball["c"] is not None:
            v = c - _last_ball["c"]
            _last_ball["v"] = 0.6*v + 0.4*_last_ball["v"]

        _last_ball.update({"xyxy": xy, "c": c, "miss": 0})
        return chosen

    # fallback
    _last_ball["miss"] += 1
    return sv.Detections.empty()


# ============================================================
# 5) CLASSIFICATION JOUEURS / GK / ARBITRES
# ============================================================

# ---- joueurs ----
team_state = {}
VOTE_WINDOW = 5
LOCK_AFTER  = 3
AB_MARGIN_MIN = 10.0
STRONG_SWITCH = 6

def classify_team_stable(frame_rgb, players_dets, refs, frame_idx):
    if len(players_dets.xyxy) == 0:
        return players_dets

    out = []
    tids = getattr(players_dets, "tracker_id", None)

    ab_refs = {k:v for k,v in refs.items() if k in ["team_A","team_B"]}

    for i,xy in enumerate(players_dets.xyxy):
        tid = int(tids[i]) if tids is not None else i

        if tid not in team_state:
            team_state[tid] = {"votes": deque(maxlen=VOTE_WINDOW),
                               "team": None,
                               "opposition": 0}

        crop = sv.crop_image(frame_rgb, xy)
        lab, dist = assign_team_with_scores(crop, ab_refs)

        vote = int(lab)
        margin = abs(dist.get("team_A",0)-dist.get("team_B",0))

        st = team_state[tid]

        if st["team"] is not None and margin < AB_MARGIN_MIN:
            out.append(st["team"])
            continue

        st["votes"].append(vote)

        if st["team"] is None:
            most, n = Counter(st["votes"]).most_common(1)[0]
            if margin >= AB_MARGIN_MIN and n >= LOCK_AFTER:
                st["team"] = most
            out.append(st["team"] if st["team"] is not None else most)
            continue

        if vote != st["team"] and margin >= AB_MARGIN_MIN:
            st["opposition"] += 1
            if st["opposition"] >= STRONG_SWITCH:
                st["team"] = vote
                st["opposition"] = 0
        else:
            st["opposition"] = 0

        out.append(st["team"])

    players_dets.class_id = np.array(out, int)
    return players_dets


# ---- arbitres ----
def split_refs_by_color(others_dets, frame_rgb, refs, pitch_mask_bool):
    if len(others_dets.xyxy) == 0:
        return sv.Detections.empty(), others_dets

    ref_vec = refs.get("referee", None)
    ab_refs = {k:v for k,v in refs.items() if k in ["team_A","team_B"]}

    keep_ref = np.zeros(len(others_dets.xyxy), bool)

    tids = getattr(others_dets, "tracker_id", None)

    for i, xy in enumerate(others_dets.xyxy):
        crop = sv.crop_image(frame_rgb, xy)

        _, d_ref = assign_team_with_scores(crop, {"referee": ref_vec})
        d_r = d_ref.get("referee", 1e9)

        _, d_ab = assign_team_with_scores(crop, ab_refs)
        min_ab = min(d_ab.values()) if len(d_ab)>0 else 1e9

        is_ref = (d_r < 12.0) or ((min_ab - d_r) >= 8.0)

        keep_ref[i] = is_ref

    return others_dets[keep_ref], others_dets[~keep_ref]


def stabilize_refs(ref_dets):
    if len(ref_dets.xyxy) == 0:
        return ref_dets

    ref_dets.class_id = np.full(len(ref_dets.xyxy),
                                TEAM_LABELS["referee"], int)
    return ref_dets


# ---- GK ----

gk_state = {}

def classify_gk_locked(gk_dets, frame_rgb, refs):
    if len(gk_dets.xyxy) == 0:
        return gk_dets

    out = []
    tids = getattr(gk_dets, "tracker_id", None)

    gk_refs = {k:v for k,v in refs.items() if "GK" in k}

    for i,xy in enumerate(gk_dets.xyxy):
        tid = int(tids[i]) if tids is not None else i

        if tid in gk_state:
            out.append(gk_state[tid])
            continue

        crop = sv.crop_image(frame_rgb, xy)

        if len(gk_refs)>0:
            rid = assign_team(crop, gk_refs, mask_grass=False)
            final = (
                TEAM_LABELS["team_A"]
                if rid == TEAM_LABELS["team_A_GK"]
                else TEAM_LABELS["team_B"]
            )
        else:
            ab = {k:v for k,v in refs.items() if k in ["team_A","team_B"]}
            rid = assign_team(crop, ab) if len(ab)>0 else TEAM_LABELS["team_A"]
            final = TEAM_LABELS["team_A"] if rid == TEAM_LABELS["team_A"] else TEAM_LABELS["team_B"]

        gk_state[tid] = final
        out.append(final)

    gk_dets.class_id = np.array(out, int)
    return gk_dets


# ============================================================
# 6) PIPELINE VIDÉO COMPLET (C13)
# ============================================================

def run_pipeline(video_path, output_path, model, team_refs):
    """
    Exécute l’équivalent de C13.
    Utilisé depuis Streamlit.
    """

    vi = sv.VideoInfo.from_video_path(video_path)
    W,H,FPS = int(vi.width), int(vi.height), float(vi.fps or 25.0)

    writer = cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        FPS,
        (W,H)
    )

    frame_gen = sv.get_video_frames_generator(video_path)
    frame_idx = 0
    pitch_mask_bool = None

    for frame in frame_gen:

        if pitch_mask_bool is None or frame_idx % 10 == 0:
            pitch_mask_bool = build_pitch_mask_fast(frame)

        result = model.infer(frame, confidence=0.30)[0]
        detections = sv.Detections.from_inference(result)

        # --- ballon
        ball_raw = detections[detections.class_id == BALL_ID]
        ball_dets = track_ball_robust(ball_raw, frame)

        # --- autres objets
        others = detections[detections.class_id != BALL_ID]
        others = others.with_nms(0.50, class_agnostic=True)
        others = byte_tracker.update_with_detections(others)
        others = keep_if_on_pitch(others, pitch_mask_bool)

        # --- split
        refs_raw, non_refs = split_refs_by_color(others, frame, team_refs, pitch_mask_bool)

        gk_raw = non_refs[non_refs.class_id == GOALKEEPER_ID]
        ppl_raw = non_refs[non_refs.class_id == PLAYER_ID]

        # joueurs
        ppl = classify_team_stable(frame, ppl_raw, team_refs, frame_idx)

        # gardiens
        gk = classify_gk_locked(gk_raw, frame, team_refs)

        # arbitres
        refs = stabilize_refs(refs_raw)

        all_dets = sv.Detections.merge([ppl, gk, refs])

        labels = (
            [f"#{tid}" for tid in getattr(all_dets, "tracker_id", [])]
            if getattr(all_dets, "tracker_id", None) is not None
            else []
        )

        annotated = frame.copy()

        if len(all_dets.xyxy) > 0:
            annotated = ellipse_annotator.annotate(annotated, all_dets)
            if len(labels)==len(all_dets.xyxy):
                annotated = label_annotator.annotate(annotated, all_dets, labels=labels)

        if len(ball_dets.xyxy) > 0:
            lifted_xy = ball_dets.xyxy
            dets_triangle = sv.Detections(xyxy=lifted_xy,
                                          class_id=np.zeros(len(lifted_xy), int))
            annotated = triangle_annotator.annotate(annotated, dets_triangle)

        writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        frame_idx += 1

    writer.release()
    return output_path
