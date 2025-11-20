# ============================================================
#  pipeline.py — VERSION FINALE COMPLÈTE + FIXES
# ============================================================

import cv2
import numpy as np
import supervision as sv
from collections import deque, Counter, defaultdict

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

# ============================================================
# 2) ANNOTATEURS
# ============================================================

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

byte_tracker = sv.ByteTrack()


# ============================================================
# 3) COULEURS — extract_jersey_lab (C9)
# ============================================================

def extract_jersey_lab(crop_rgb, torso_margin=(0.25,0.70),
                       side_margin=0.25, mask_grass=True,
                       min_pixels=120):

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

    return np.mean(lab, axis=0)


def assign_team_with_scores(crop_rgb, refs, *, mask_grass=True):
    col = extract_jersey_lab(crop_rgb, mask_grass=mask_grass)

    if col is None:
        # fallback → premier ref valide
        for k in ["team_A","team_B","referee","team_A_GK","team_B_GK"]:
            if refs.get(k) is not None:
                return TEAM_LABELS[k], {k: 0.0}

        return TEAM_LABELS["team_A"], {}

    dists = {}
    for name, ref in refs.items():
        if ref is None:
            continue
        dists[name] = float(np.linalg.norm(col - ref))

    assigned = min(dists, key=dists.get)
    return TEAM_LABELS[assigned], dists


def assign_team(crop_rgb, refs, mask_grass=True):
    lab, _ = assign_team_with_scores(crop_rgb, refs, mask_grass=mask_grass)
    return lab


# ============================================================
# 4) TERRAIN — Pitch mask
# ============================================================

def build_pitch_mask_fast(frame_rgb, h_low=35, h_high=90,
                          s_min=40, v_min=40,
                          morph_ks=7, scale=0.33):

    H, W = frame_rgb.shape[:2]

    small = cv2.resize(frame_rgb, (int(W*scale), int(H*scale)),
                       interpolation=cv2.INTER_AREA)

    hsv = cv2.cvtColor(small, cv2.COLOR_RGB2HSV)
    Hc, Sc, Vc = cv2.split(hsv)

    grass = (Hc >= h_low) & (Hc <= h_high) & (Sc >= s_min) & (Vc >= v_min)
    mask = grass.astype(np.uint8) * 255

    if morph_ks > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph_ks,morph_ks))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)

    num, labels = cv2.connectedComponents(mask)

    if num > 1:
        areas = [(labels==i).sum() for i in range(1, num)]
        biggest = 1 + int(np.argmax(areas))
        mask = (labels == biggest).astype(np.uint8) * 255

    mask = cv2.resize(mask, (W,H), interpolation=cv2.INTER_NEAREST)
    return mask.astype(bool)


# ============================================================
# 4.1) Fonction manquante → AJOUTÉE 🔥
# ============================================================

def _is_on_pitch_xyxy(xyxy, pitch_mask_bool, patch_px=8, min_cover=0.25):
    x1,y1,x2,y2 = xyxy.astype(int)
    cx = int((x1+x2)/2)
    by = int(y2)

    H, W = pitch_mask_bool.shape

    x1p = max(0, cx - patch_px)
    x2p = min(W, cx + patch_px)
    y1p = max(0, by - patch_px)
    y2p = min(H, by + patch_px)

    patch = pitch_mask_bool[y1p:y2p, x1p:x2p]

    if patch.size == 0:
        return False

    return patch.mean() >= min_cover


# ============================================================
# 4.2) keep_if_on_pitch → REAJOUTÉ 🔥
# ============================================================

def keep_if_on_pitch(dets, pitch_mask_bool, patch_px=8, min_cover=0.25):

    if len(dets.xyxy) == 0:
        return dets

    keep = np.zeros(len(dets.xyxy), bool)

    for i, xy in enumerate(dets.xyxy):
        keep[i] = _is_on_pitch_xyxy(xy, pitch_mask_bool,
                                    patch_px=patch_px,
                                    min_cover=min_cover)

    return dets[keep]


# ============================================================
# 5) BALL TRACKER — FIXED (float)
# ============================================================

_last_ball = {
    "xyxy": None,
    "c": None,
    "v": np.array([0.0,0.0], dtype=float),
    "miss": 0
}

BALL_COAST = 20
EMA = 0.6

def track_ball_robust(ball_dets, frame_rgb):
    global _last_ball

    if len(ball_dets.xyxy) > 0:
        xy = ball_dets.xyxy[0].astype(float)
        c = np.array([(xy[0]+xy[2])/2, (xy[1]+xy[3])/2], float)

        if _last_ball["c"] is not None:
            v = c - _last_ball["c"]
            _last_ball["v"] = EMA*v + (1-EMA)*_last_ball["v"]

        _last_ball["xyxy"] = xy
        _last_ball["c"] = c
        _last_ball["miss"] = 0
        return ball_dets

    if _last_ball["xyxy"] is not None and _last_ball["miss"] < BALL_COAST:
        _last_ball["miss"] += 1
        _last_ball["v"] *= 0.75

        c = _last_ball["c"] + _last_ball["v"]
        x1,y1,x2,y2 = _last_ball["xyxy"]
        w = x2-x1 ; h = y2-y1

        xy = np.array([c[0]-w/2, c[1]-h/2, c[0]+w/2, c[1]+h/2])
        return sv.Detections(xyxy=xy[np.newaxis,:],
                             class_id=np.array([0], int))

    _last_ball = {
        "xyxy": None,
        "c": None,
        "v": np.array([0.0,0.0], float),
        "miss": 0
    }
    return sv.Detections.empty()


# ============================================================
# 6) CLASSIFICATION JOUEURS
# ============================================================

VOTE_WINDOW = 5
LOCK_AFTER  = 3
MARGIN_MIN  = 10.0
STRONG_SWITCH = 6

team_state = {}

def classify_team_stable(frame_rgb, players_dets, team_refs, frame_idx):

    if len(players_dets.xyxy) == 0:
        return players_dets

    ab_refs = {
        k:v for k,v in team_refs.items()
        if k in ["team_A","team_B"] and v is not None
    }

    out = []
    tids = getattr(players_dets,"tracker_id",None)

    for i,xy in enumerate(players_dets.xyxy):
        tid = int(tids[i]) if tids is not None else i

        if tid not in team_state:
            team_state[tid] = {
                "votes": deque(maxlen=VOTE_WINDOW),
                "team": None,
                "opp": 0
            }

        st = team_state[tid]
        crop = sv.crop_image(frame_rgb, xy)
        lab, dist = assign_team_with_scores(crop, ab_refs)

        margin = abs(dist.get("team_A",0)-dist.get("team_B",0))
        vote = int(lab)

        ## existing locked team? keep it
        if st["team"] is not None and margin < MARGIN_MIN:
            out.append(st["team"])
            continue

        # accumulate votes
        st["votes"].append(vote)

        if st["team"] is None:
            best, n = Counter(st["votes"]).most_common(1)[0]
            if margin >= MARGIN_MIN and n >= LOCK_AFTER:
                st["team"] = best
            out.append(st["team"] if st["team"] is not None else best)
            continue

        if vote != st["team"] and margin >= MARGIN_MIN:
            st["opp"] += 1
            if st["opp"] >= STRONG_SWITCH:
                st["team"] = vote
                st["opp"] = 0
        else:
            st["opp"] = 0

        out.append(st["team"])

    players_dets.class_id = np.array(out, int)
    return players_dets


# ============================================================
# 7) PIPELINE VIDÉO COMPLET
# ============================================================

def run_pipeline(video_path, output_path, model, team_refs):

    vi = sv.VideoInfo.from_video_path(video_path)
    W,H,FPS = int(vi.width), int(vi.height), float(vi.fps or 25)

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

        # detections
        result = model.infer(frame, confidence=0.30)[0]
        detections = sv.Detections.from_inference(result)

        # ball
        ball_raw  = detections[detections.class_id == BALL_ID]
        ball_dets = track_ball_robust(ball_raw, frame)

        # players/ref/gk
        others = detections[detections.class_id != BALL_ID]
        others = others.with_nms(0.50, class_agnostic=True)
        others = byte_tracker.update_with_detections(others)
        others = keep_if_on_pitch(others, pitch_mask_bool)

        refs_raw = others[others.class_id == REFEREE_ID]
        ppl_raw  = others[others.class_id == PLAYER_ID]
        gk_raw   = others[others.class_id == GOALKEEPER_ID]

        ppl = classify_team_stable(frame, ppl_raw, team_refs, frame_idx)
        gk  = gk_raw     # GK logic can be added
        refs = refs_raw

        all_dets = sv.Detections.merge([ppl, gk, refs])

        labels = (
            [f"#{tid}" for tid in getattr(all_dets,"tracker_id",[])]
            if getattr(all_dets,"tracker_id",None) is not None else []
        )

        annotated = frame.copy()

        if len(all_dets.xyxy) > 0:
            annotated = ellipse_annotator.annotate(annotated, all_dets)
            if len(labels)==len(all_dets.xyxy):
                annotated = label_annotator.annotate(annotated, all_dets, labels=labels)

        if len(ball_dets.xyxy) > 0:
            dets_triangle = sv.Detections(xyxy=ball_dets.xyxy,
                                          class_id=np.zeros(len(ball_dets.xyxy),int))
            annotated = triangle_annotator.annotate(annotated, dets_triangle)

        writer.write(cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))
        frame_idx += 1

    writer.release()
    return output_path
