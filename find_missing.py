"""Full missing-object pipeline: sample frames, detect, temporal-vote into stable
tracks, match parent vs. test video (Hungarian algorithm on IoU, or on color-
histogram similarity for a handheld/moving camera), pixel-diff cross-check
each reported miss, read any printed product text via OCR, and save
annotated images + results.csv.
"""
import argparse
import csv
import re
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from ultralytics import YOLO

try:
    import pytesseract
    _tess_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    if Path(_tess_path).exists():
        pytesseract.pytesseract.tesseract_cmd = _tess_path
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False


# Open-vocabulary prompt list for YOLO-World models. Deliberately a broad list
# of everyday desk/household objects rather than the specific items in any one
# test scene, so the tool stays general-purpose; COCO's 80 classes can't name
# common things like glasses or a pen at all.
DEFAULT_VOCAB = [
    "laptop", "keyboard", "mouse", "cell phone", "tablet", "headphones", "earbuds case",
    "charger", "power bank", "cable", "remote control", "calculator", "glasses", "watch",
    "wallet", "keys", "pen", "pencil", "marker", "notebook", "book", "paper", "sticky note",
    "scissors", "stapler", "tape", "bottle", "cup", "mug", "jar", "can", "box", "bag",
    "toy", "figurine", "lamp", "plant", "tissue box", "cosmetics jar", "spray bottle",
]


def load_vocab(spec: str | None) -> list[str]:
    """--vocab accepts a path to a text file (one class per line) or a
    comma-separated list; omitted, it falls back to DEFAULT_VOCAB."""
    if not spec:
        return DEFAULT_VOCAB
    path = Path(spec)
    if path.is_file():
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return [c.strip() for c in spec.split(",") if c.strip()]


def sample_indices(frame_count: int, n_frames: int) -> list[int]:
    if frame_count <= 0:
        return []
    n = min(n_frames, frame_count)
    if n == 1:
        return [0]
    return [round(i * (frame_count - 1) / (n - 1)) for i in range(n)]


def read_sampled_frames(video_path: Path, n_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for idx in sample_indices(frame_count, n_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if ok:
            frames.append(frame)
    cap.release()
    return frames


def box_area(box: tuple) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def dedupe_same_class(dets: list, contain_thr: float = 0.8) -> list:
    """YOLO sometimes fires twice on one physical object with the same class
    (e.g. a bottle body plus just its hanging strap both called "bottle").
    Plain IoU misses these because the smaller box barely overlaps the big
    one's area, but it sits almost entirely *inside* it. Flag that as a
    duplicate and keep only the highest-confidence box of the pair."""
    kept = []
    for class_name, box, score in sorted(dets, key=lambda d: -d[2]):
        duplicate = False
        for k_class, k_box, _k_score in kept:
            if k_class != class_name:
                continue
            ix1, iy1 = max(box[0], k_box[0]), max(box[1], k_box[1])
            ix2, iy2 = min(box[2], k_box[2]), min(box[3], k_box[3])
            inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
            smaller_area = min(box_area(box), box_area(k_box))
            if smaller_area > 0 and inter / smaller_area >= contain_thr:
                duplicate = True
                break
        if not duplicate:
            kept.append((class_name, box, score))
    return kept


def detect_all_frames(model: YOLO, frames: list[np.ndarray], conf: float, imgsz: int = 640,
                      agnostic_nms: bool = False) -> list[list]:
    """Returns, per frame, a list of (class_name, box_xyxy, det_conf)."""
    per_frame = []
    for frame in frames:
        result = model.predict(frame, conf=conf, imgsz=imgsz, agnostic_nms=agnostic_nms, verbose=False)[0]
        dets = []
        for box in result.boxes:
            class_name = model.names[int(box.cls[0])]
            xyxy = tuple(box.xyxy[0].tolist())
            score = float(box.conf[0])
            dets.append((class_name, xyxy, score))
        per_frame.append(dedupe_same_class(dets))
    return per_frame


def iou(box_a: tuple, box_b: tuple) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


class Track:
    """A persistent object across frames of one video: same class, overlapping box."""

    def __init__(self, class_name: str, box: tuple):
        self.class_name = class_name
        self.boxes = [box]

    def add(self, box: tuple) -> None:
        self.boxes.append(box)

    @property
    def count(self) -> int:
        return len(self.boxes)

    @property
    def box(self) -> tuple:
        # Median is more robust to one-off detector jitter than a mean would be.
        return tuple(np.median(np.array(self.boxes), axis=0).tolist())


def build_tracks(per_frame_detections: list[list], iou_thr: float) -> list[Track]:
    """Greedy temporal tracking: match each frame's detections to an existing
    track of the same class with the highest IoU, or start a new track."""
    tracks: list[Track] = []
    for dets in per_frame_detections:
        used_indices = set()
        for class_name, box, _score in dets:
            best_idx, best_iou = None, 0.0
            for i, track in enumerate(tracks):
                if i in used_indices or track.class_name != class_name:
                    continue
                score = iou(track.box, box)
                if score > best_iou:
                    best_idx, best_iou = i, score
            if best_idx is not None and best_iou >= iou_thr:
                tracks[best_idx].add(box)
                used_indices.add(best_idx)
            else:
                tracks.append(Track(class_name, box))
    return tracks


def stable_tracks(tracks: list, n_frames: int, min_frac: float) -> list:
    return [t for t in tracks if n_frames > 0 and t.count / n_frames >= min_frac]


def crop_histogram(frame: np.ndarray, box: tuple):
    """A lightweight appearance fingerprint: the color distribution inside a
    box, as an HSV histogram. Unlike a box's pixel position, this stays
    roughly stable when the camera moves between recordings."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
    if x2 <= x1 or y2 <= y1:
        return None
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()


def hist_similarity(h1, h2) -> float:
    if h1 is None or h2 is None:
        return 0.0
    dist = cv2.compareHist(h1.astype("float32"), h2.astype("float32"), cv2.HISTCMP_BHATTACHARYYA)
    return max(0.0, 1.0 - dist)


def read_label_text(frame: np.ndarray, box: tuple, max_len: int = 40) -> str:
    """OCR the crop for any printed product text (e.g. "LLOYDS", "120W").
    Returns "" if OCR isn't installed, the crop is empty, or nothing legible
    is found -- this is a best-effort enrichment, not something the pipeline
    depends on."""
    if not OCR_AVAILABLE:
        return ""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
    if x2 <= x1 or y2 <= y1:
        return ""
    crop = frame[y1:y2, x1:x2]
    h, w = crop.shape[:2]
    scale = max(1.0, 450.0 / max(h, w))
    if scale > 1.0:
        crop = cv2.resize(crop, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    enhanced = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
    try:
        text = pytesseract.image_to_string(enhanced, config="--psm 6")
    except Exception:
        return ""
    text = " ".join(text.split())[:max_len]

    # OCR on a noisy/busy crop (reflections, background clutter) tends to spit
    # out short runs of punctuation and stray letters rather than real words.
    # Require at least one run of 3+ letters/digits to call it a real reading,
    # instead of reporting that noise as a "product name".
    if not re.search(r"[A-Za-z0-9]{3,}", text):
        return ""
    return text


class AppearanceTrack:
    """Same role as Track, but identified by color-histogram similarity
    instead of box position -- for a handheld/moving camera where the same
    physical object lands at a different pixel location every time."""

    def __init__(self, class_name: str, box: tuple, hist: np.ndarray, frame_idx: int, score: float):
        self.class_name = class_name
        self.boxes = [box]
        self.frame_ids = [frame_idx]
        self.score_sum = score
        self._hist_sum = hist.copy()
        self._hist_count = 1
        self.best_alt_similarity = 0.0  # filled in for unmatched tracks during cross-video matching
        self.aliases: list[str] = []  # other labels the detector gave this same physical object

    def add(self, box: tuple, hist: np.ndarray, frame_idx: int, score: float) -> None:
        self.boxes.append(box)
        self.frame_ids.append(frame_idx)
        self.score_sum += score
        self._hist_sum += hist
        self._hist_count += 1

    @property
    def count(self) -> int:
        return len(self.boxes)

    @property
    def box(self) -> tuple:
        return tuple(np.median(np.array(self.boxes), axis=0).tolist())

    @property
    def histogram(self) -> np.ndarray:
        return self._hist_sum / self._hist_count


def build_tracks_by_appearance(per_frame_detections: list[list], frames: list[np.ndarray],
                                hist_thr: float) -> list[AppearanceTrack]:
    """Greedy temporal tracking like build_tracks, but matches detections to
    existing tracks by appearance similarity instead of IoU -- a moving
    camera can put the same object anywhere in frame, even between two
    sampled frames of the *same* video."""
    tracks: list[AppearanceTrack] = []
    for frame_idx, (dets, frame) in enumerate(zip(per_frame_detections, frames)):
        used_indices = set()
        for class_name, box, score in dets:
            hist = crop_histogram(frame, box)
            if hist is None:
                continue
            best_idx, best_sim = None, 0.0
            for i, track in enumerate(tracks):
                if i in used_indices or track.class_name != class_name:
                    continue
                sim = hist_similarity(track.histogram, hist)
                if sim > best_sim:
                    best_idx, best_sim = i, sim
            if best_idx is not None and best_sim >= hist_thr:
                tracks[best_idx].add(box, hist, frame_idx, score)
                used_indices.add(best_idx)
            else:
                tracks.append(AppearanceTrack(class_name, box, hist, frame_idx, score))
    return tracks


def group_alias_tracks(tracks: list[AppearanceTrack], iou_thr: float = 0.4,
                       agree_frac: float = 0.5, min_shared: int = 2) -> list[list[AppearanceTrack]]:
    """A large open vocabulary often gives one physical object several labels
    across frames (a pen is also "pencil", glasses are also "scissors"), and
    each label becomes its own track. Two tracks are the same physical object
    if, in the frames where both were seen, their boxes coincide (IoU) most of
    the time. Groups them with union-find.

    IoU rather than "one box inside the other" on purpose: containment merged a
    pen into the notebook it lies on, and a mouse into the notepad under it,
    hiding both removals behind an object that was still present."""
    parent = list(range(len(tracks)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(tracks)):
        boxes_i = dict(zip(tracks[i].frame_ids, tracks[i].boxes))
        for j in range(i + 1, len(tracks)):
            boxes_j = dict(zip(tracks[j].frame_ids, tracks[j].boxes))
            shared = boxes_i.keys() & boxes_j.keys()
            if len(shared) < min_shared:
                continue
            agree = sum(iou(boxes_i[f], boxes_j[f]) >= iou_thr for f in shared)
            if agree / len(shared) >= agree_frac:
                parent[find(i)] = find(j)

    groups: dict[int, list[AppearanceTrack]] = {}
    for i, track in enumerate(tracks):
        groups.setdefault(find(i), []).append(track)
    return list(groups.values())


def unmatched_objects(groups: list[list[AppearanceTrack]], matched_ids: set) -> list[AppearanceTrack]:
    """A physical object counts as present if *any* of its labels found a
    match. Otherwise report it once, under the label with the highest total
    detection confidence (a frequent-but-weak wrong label shouldn't outvote a
    rarer confident one), with the other labels kept as aliases."""
    result = []
    for group in groups:
        if any(id(t) in matched_ids for t in group):
            continue
        rep = max(group, key=lambda t: t.score_sum)
        rep.aliases = sorted({t.class_name for t in group} - {rep.class_name})
        result.append(rep)
    return result


def match_cross_video_by_appearance(parent_tracks: list[AppearanceTrack],
                                     test_tracks: list[AppearanceTrack], hist_thr: float):
    """Same role as match_cross_video, but the Hungarian algorithm's cost
    matrix is built from appearance similarity instead of IoU -- position is
    meaningless once the camera moves, but a physical object's color/texture
    distribution stays roughly stable across recordings."""
    classes = {t.class_name for t in parent_tracks} | {t.class_name for t in test_tracks}
    matched, missing, added = [], [], []
    for class_name in classes:
        p_tracks = [t for t in parent_tracks if t.class_name == class_name]
        t_tracks = [t for t in test_tracks if t.class_name == class_name]
        if not p_tracks:
            added.extend(t_tracks)
            continue
        if not t_tracks:
            missing.extend(p_tracks)
            continue

        sims = np.zeros((len(p_tracks), len(t_tracks)))
        for i, pt in enumerate(p_tracks):
            for j, tt in enumerate(t_tracks):
                sims[i, j] = hist_similarity(pt.histogram, tt.histogram)

        row_ind, col_ind = linear_sum_assignment(-sims)
        matched_p, matched_t = set(), set()
        for r, c in zip(row_ind, col_ind):
            if sims[r, c] >= hist_thr:
                matched.append((p_tracks[r], t_tracks[c]))
                matched_p.add(r)
                matched_t.add(c)

        for i in range(len(p_tracks)):
            if i not in matched_p:
                p_tracks[i].best_alt_similarity = float(sims[i].max()) if sims.shape[1] > 0 else 0.0
                missing.append(p_tracks[i])
        added.extend(t_tracks[j] for j in range(len(t_tracks)) if j not in matched_t)
    return matched, missing, added


def appearance_confidence(track: AppearanceTrack, hist_thr: float, ambiguous_margin: float = 0.15):
    """A near-miss (best alternative similarity close to the match threshold)
    suggests the object might still be present but looked slightly different
    -- an ambiguous, low-confidence "missing". A clearly low best-alternative
    similarity means nothing in the test video resembled it at all."""
    best_alt = track.best_alt_similarity
    confidence = "low" if best_alt >= hist_thr - ambiguous_margin else "high"
    return best_alt, confidence


def match_cross_video(parent_tracks: list[Track], test_tracks: list[Track], iou_thr: float):
    """Per class, use the Hungarian algorithm to optimally pair parent vs. test
    tracks by IoU. Returns (matched_pairs, missing_parent_tracks, added_test_tracks)."""
    classes = {t.class_name for t in parent_tracks} | {t.class_name for t in test_tracks}
    matched, missing, added = [], [], []
    for class_name in classes:
        p_tracks = [t for t in parent_tracks if t.class_name == class_name]
        t_tracks = [t for t in test_tracks if t.class_name == class_name]
        if not p_tracks:
            added.extend(t_tracks)
            continue
        if not t_tracks:
            missing.extend(p_tracks)
            continue

        cost = np.zeros((len(p_tracks), len(t_tracks)))
        for i, pt in enumerate(p_tracks):
            for j, tt in enumerate(t_tracks):
                cost[i, j] = -iou(pt.box, tt.box)  # minimize negative IoU = maximize IoU

        row_ind, col_ind = linear_sum_assignment(cost)
        matched_p, matched_t = set(), set()
        for r, c in zip(row_ind, col_ind):
            if -cost[r, c] >= iou_thr:
                matched.append((p_tracks[r], t_tracks[c]))
                matched_p.add(r)
                matched_t.add(c)

        missing.extend(p_tracks[i] for i in range(len(p_tracks)) if i not in matched_p)
        added.extend(t_tracks[j] for j in range(len(t_tracks)) if j not in matched_t)
    return matched, missing, added


def pixel_diff_check(parent_frame: np.ndarray, test_frame: np.ndarray, box: tuple,
                      blur_ksize: int = 5, diff_thr: int = 25, change_ratio_thr: float = 0.2,
                      search_radius: int = 8):
    """Compare the same pixel region in a parent vs. test frame. Two separate
    recordings are never pixel-perfectly aligned even with a "fixed" camera, so
    we search small (dx, dy) shifts and keep the best-aligned (lowest-diff)
    match -- otherwise a few pixels of drift at an object's edge reads as
    "changed" even when nothing actually moved. A large remaining changed-pixel
    fraction after that search is real evidence the object is gone."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    H, W = parent_frame.shape[:2]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, W), min(y2, H)
    if x2 <= x1 or y2 <= y1:
        return 0.0, "low"

    p_blur = cv2.GaussianBlur(cv2.cvtColor(parent_frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY),
                               (blur_ksize, blur_ksize), 0).astype(np.int16)

    best_ratio = 1.0
    for dy in range(-search_radius, search_radius + 1):
        for dx in range(-search_radius, search_radius + 1):
            tx1, ty1, tx2, ty2 = x1 + dx, y1 + dy, x2 + dx, y2 + dy
            if tx1 < 0 or ty1 < 0 or tx2 > W or ty2 > H:
                continue
            t_blur = cv2.GaussianBlur(cv2.cvtColor(test_frame[ty1:ty2, tx1:tx2], cv2.COLOR_BGR2GRAY),
                                       (blur_ksize, blur_ksize), 0).astype(np.int16)
            diff = np.abs(p_blur - t_blur)
            ratio = float(np.count_nonzero(diff > diff_thr)) / diff.size
            best_ratio = min(best_ratio, ratio)

    confidence = "high" if best_ratio >= change_ratio_thr else "low"
    return best_ratio, confidence


def best_view(track, frames: list[np.ndarray]) -> tuple[np.ndarray, tuple]:
    """The frame + box where this object was actually seen. A fixed camera can
    use frame 0 and the median box, but a panning camera may not have the
    object in frame 0 at all, so appearance tracks use their own middle sighting."""
    frame_ids = getattr(track, "frame_ids", None)
    if not frame_ids:
        return frames[0], track.box
    mid = len(frame_ids) // 2
    return frames[frame_ids[mid]], track.boxes[mid]


def aliases_of(track) -> str:
    # Fixed-camera Tracks don't carry aliases; only appearance tracks get grouped.
    return "; ".join(getattr(track, "aliases", []))


class ClipMatcher:
    """CLIP image embeddings as the appearance fingerprint, instead of a color
    histogram. Same object across two handheld recordings scored 0.88-0.99 on
    the desk-pan dataset, where histograms dropped as low as 0.44 for an
    object (the watch) that hadn't moved -- lighting and viewpoint shift the
    colors, but not what CLIP thinks the object is. The same model also names
    each object by comparing its crops to the vocabulary text, which is far
    more reliable than YOLO-World's own label vote (glasses: YOLO says
    "scissors", CLIP says "glasses")."""

    def __init__(self, vocab: list[str], model_name: str = "ViT-B/32"):
        import clip
        import torch
        self._torch = torch
        self.model, self.preprocess = clip.load(model_name, device="cpu")
        self.vocab = vocab
        with torch.no_grad():
            text = self.model.encode_text(clip.tokenize([f"a photo of {c}" for c in vocab])).float()
        self.text_embs = (text / text.norm(dim=-1, keepdim=True)).numpy()

    def embed(self, views: list[tuple[np.ndarray, tuple]]) -> np.ndarray:
        """One L2-normalised embedding per (frame, box) crop."""
        from PIL import Image
        crops = []
        for frame, box in views:
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            crop = frame[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)]
            if crop.size:
                crops.append(self.preprocess(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
        if not crops:
            return np.zeros((0, self.text_embs.shape[1]), dtype=np.float32)
        out = []
        with self._torch.no_grad():
            for i in range(0, len(crops), 64):
                e = self.model.encode_image(self._torch.stack(crops[i:i + 64])).float()
                out.append((e / e.norm(dim=-1, keepdim=True)).numpy())
        return np.concatenate(out)

    def embed_group(self, group: list[AppearanceTrack], frames: list[np.ndarray], per_track: int = 4) -> np.ndarray:
        """Average of a few crops spread across each track's sightings."""
        views = []
        for t in group:
            for i in np.linspace(0, len(t.frame_ids) - 1, min(per_track, len(t.frame_ids))).round().astype(int):
                views.append((frames[t.frame_ids[i]], t.boxes[i]))
        mean = self.embed(views).mean(axis=0)
        return mean / (np.linalg.norm(mean) or 1.0)

    def embed_raw(self, per_frame_detections: list[list], frames: list[np.ndarray]) -> np.ndarray:
        """Every detection in a video, whatever its label and however rarely seen."""
        return self.embed([(f, box) for f, dets in zip(frames, per_frame_detections) for _c, box, _s in dets])

    def name(self, emb: np.ndarray) -> str:
        return self.vocab[int(np.argmax(self.text_embs @ emb))]


def match_by_clip(p_groups: list, p_embs: np.ndarray, t_groups: list, t_embs: np.ndarray,
                  t_raw: np.ndarray, thr: float) -> tuple[set, dict]:
    """Class-agnostic Hungarian matching of parent objects to test objects by
    CLIP similarity, ignoring YOLO's labels entirely (they flip between
    recordings). A parent object left unmatched gets a second chance against
    every raw test detection: a stable track can be missing just because the
    detector fired on that object in too few frames, or under a label that
    came and went, even though the object is plainly there.

    Returns (indices of parent groups judged present, best similarity per
    parent group index)."""
    present, best = set(), {}
    if len(p_groups) and len(t_groups):
        sims = p_embs @ t_embs.T
        for r, c in zip(*linear_sum_assignment(-sims)):
            if sims[r, c] >= thr:
                present.add(int(r))
        for i in range(len(p_groups)):
            best[i] = float(sims[i].max())
    for i in range(len(p_groups)):
        if i in present:
            continue
        raw_best = float((t_raw @ p_embs[i]).max()) if len(t_raw) else 0.0
        best[i] = max(best.get(i, 0.0), raw_best)
        if raw_best >= thr:
            present.add(i)
    return present, best


def clip_objects(groups: list, embs: np.ndarray, present: set, best: dict, matcher: ClipMatcher) -> list:
    """Unmatched groups as report rows: one representative track per object,
    named by CLIP, with YOLO's labels kept as aliases for reference."""
    result = []
    for i, group in enumerate(groups):
        if i in present:
            continue
        for t in group:
            # Parent groups are reused for every test video; keep YOLO's own label
            # so a rename on one video doesn't leak into the next one's aliases.
            if not hasattr(t, "yolo_label"):
                t.yolo_label = t.class_name
        rep = max(group, key=lambda t: t.score_sum)
        yolo_labels = {t.yolo_label for t in group}
        rep.class_name = matcher.name(embs[i])
        rep.aliases = sorted(yolo_labels - {rep.class_name})
        rep.best_alt_similarity = best.get(i, 0.0)
        result.append(rep)
    return result


def fmt_box(box: tuple) -> str:
    return "({:.0f}, {:.0f}, {:.0f}, {:.0f})".format(*box)


def draw_box(image: np.ndarray, box: tuple, color: tuple, label: str) -> None:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    cv2.putText(image, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Find objects missing/added vs. a parent video.")
    parser.add_argument("--parent", type=str, default="videos/parent.mp4", help="Reference video")
    parser.add_argument("tests", nargs="*", default=None, help="Test videos (default: all others in videos/)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLO weights file")
    parser.add_argument("--conf", type=float, default=0.3, help="Detection confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="YOLO inference size; raise for small objects in high-res video")
    parser.add_argument("--agnostic-nms", action="store_true",
                         help="Suppress overlapping boxes across classes so one physical object keeps only its best "
                              "label (useful with a large YOLO-World vocabulary, where a pen also fires as 'pencil')")
    parser.add_argument("--vocab", type=str, default=None,
                         help="YOLO-World only: comma-separated class prompts or a text file of them (default: built-in desk/household list)")
    parser.add_argument("--min-frac", type=float, default=0.5, help="Min fraction of frames an object must appear in to count as stable")
    parser.add_argument("--iou-thr", type=float, default=0.3, help="IoU threshold for tracking/matching in fixed-camera mode")
    parser.add_argument("--hist-thr", type=float, default=0.65, help="Color-histogram similarity threshold for tracking/matching in moving-camera mode")
    parser.add_argument("--n-frames", type=int, default=12, help="Frames sampled per video")
    parser.add_argument("--out-dir", type=str, default="results", help="Output directory")
    parser.add_argument("--camera", choices=["fixed", "moving"], default="fixed",
                         help="'fixed': camera didn't move, match objects by position (IoU). "
                              "'moving': handheld/moving camera, match objects by color-histogram appearance instead")
    parser.add_argument("--match", choices=["hist", "clip"], default="hist",
                         help="Moving-camera only. 'hist': per-label color-histogram matching. 'clip': label-agnostic "
                              "CLIP image-embedding matching, plus CLIP naming of reported objects")
    parser.add_argument("--clip-thr", type=float, default=0.855,
                         help="CLIP similarity at or above which two objects count as the same one. Desk-pan data: "
                              "same object 0.87-0.99, removed objects' best candidate 0.82-0.84")
    args = parser.parse_args()
    if args.match == "clip" and args.camera != "moving":
        parser.error("--match clip requires --camera moving")

    if not OCR_AVAILABLE:
        print("Note: pytesseract not installed -- product-name OCR will be skipped (ocr_text column left blank).")

    parent_path = Path(args.parent)
    if args.tests:
        test_paths = [Path(p) for p in args.tests]
    else:
        test_paths = [p for p in sorted(Path("videos").glob("*.mp4")) if p.name != parent_path.name]

    model = YOLO(args.model)
    if "world" in Path(args.model).name.lower():
        vocab = load_vocab(args.vocab)
        model.set_classes(vocab)
        print(f"YOLO-World vocabulary: {len(vocab)} classes")
    else:
        vocab = list(model.names.values())
    moving = args.camera == "moving"
    matcher = ClipMatcher(vocab) if args.match == "clip" else None

    print(f"Sampling + detecting parent: {parent_path.name}")
    parent_frames = read_sampled_frames(parent_path, args.n_frames)
    parent_dets = detect_all_frames(model, parent_frames, args.conf, args.imgsz, args.agnostic_nms)
    if moving:
        parent_tracks = stable_tracks(build_tracks_by_appearance(parent_dets, parent_frames, args.hist_thr),
                                       len(parent_frames), args.min_frac)
    else:
        parent_tracks = stable_tracks(build_tracks(parent_dets, args.iou_thr), len(parent_frames), args.min_frac)
    print(f"  {len(parent_tracks)} stable object(s): " +
          ", ".join(f"{t.class_name}({t.count}/{len(parent_frames)})" for t in parent_tracks))
    if matcher:
        parent_groups = group_alias_tracks(parent_tracks)
        parent_embs = np.stack([matcher.embed_group(g, parent_frames) for g in parent_groups])
        parent_raw = matcher.embed_raw(parent_dets, parent_frames)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = []

    for test_path in test_paths:
        print(f"Sampling + detecting test: {test_path.name}")
        test_frames = read_sampled_frames(test_path, args.n_frames)
        test_dets = detect_all_frames(model, test_frames, args.conf, args.imgsz, args.agnostic_nms)

        if moving:
            test_tracks = stable_tracks(build_tracks_by_appearance(test_dets, test_frames, args.hist_thr),
                                         len(test_frames), args.min_frac)
        else:
            test_tracks = stable_tracks(build_tracks(test_dets, args.iou_thr), len(test_frames), args.min_frac)
        print(f"  {len(test_tracks)} stable object(s): " +
              ", ".join(f"{t.class_name}({t.count}/{len(test_frames)})" for t in test_tracks))

        if matcher:
            test_groups = group_alias_tracks(test_tracks)
            test_embs = np.stack([matcher.embed_group(g, test_frames) for g in test_groups]) if test_groups \
                else np.zeros((0, parent_embs.shape[1]), dtype=np.float32)
            test_raw = matcher.embed_raw(test_dets, test_frames)
            p_present, p_best = match_by_clip(parent_groups, parent_embs, test_groups, test_embs, test_raw, args.clip_thr)
            t_present, t_best = match_by_clip(test_groups, test_embs, parent_groups, parent_embs, parent_raw, args.clip_thr)
            missing = clip_objects(parent_groups, parent_embs, p_present, p_best, matcher)
            added = clip_objects(test_groups, test_embs, t_present, t_best, matcher)
        elif moving:
            matched, _missing, _added = match_cross_video_by_appearance(parent_tracks, test_tracks, args.hist_thr)
            missing = unmatched_objects(group_alias_tracks(parent_tracks), {id(p) for p, _t in matched})
            added = unmatched_objects(group_alias_tracks(test_tracks), {id(t) for _p, t in matched})
        else:
            _matched, missing, added = match_cross_video(parent_tracks, test_tracks, args.iou_thr)

        annotated = test_frames[0].copy()
        for n, track in enumerate(missing):
            view_frame, view_box = best_view(track, parent_frames)
            ocr_text = read_label_text(view_frame, view_box)
            if moving:
                # No meaningful position to mark on the test frame -- the camera
                # moved, so the parent's box location doesn't correspond to
                # anything in this video. Save the parent's crop as evidence instead.
                score, confidence = appearance_confidence(track, args.clip_thr if matcher else args.hist_thr,
                                                          ambiguous_margin=0.03 if matcher else 0.15)
                crop_path = out_dir / f"{test_path.stem}_missing_{n:02d}_{track.class_name.replace(' ', '_')}.jpg"
                x1, y1, x2, y2 = (int(round(v)) for v in view_box)
                cv2.imwrite(str(crop_path), view_frame[max(y1, 0):y2, max(x1, 0):x2])
                confidence_note = f"{confidence} (best alt. similarity {score:.2f})"
            else:
                score, confidence = pixel_diff_check(parent_frames[0], test_frames[0], track.box)
                draw_box(annotated, track.box, (0, 0, 255), f"MISSING {track.class_name}")
                confidence_note = f"{confidence} ({score:.2f})"
            csv_rows.append([test_path.name, "missing", track.class_name, fmt_box(track.box),
                              confidence_note, ocr_text, aliases_of(track)])
        for track in added:
            view_frame, view_box = best_view(track, test_frames)
            ocr_text = read_label_text(view_frame, view_box)
            draw_box(annotated, track.box, (0, 255, 0), f"ADDED {track.class_name}")
            csv_rows.append([test_path.name, "added", track.class_name, fmt_box(track.box), "", ocr_text,
                              aliases_of(track)])

        print(f"  -> {len(missing)} missing, {len(added)} added")
        for t in missing:
            print(f"     MISSING {t.class_name} at {fmt_box(t.box)}" + (f"  (also seen as: {aliases_of(t)})" if aliases_of(t) else ""))
        for t in added:
            print(f"     ADDED   {t.class_name} at {fmt_box(t.box)}" + (f"  (also seen as: {aliases_of(t)})" if aliases_of(t) else ""))

        out_path = out_dir / f"{test_path.stem}_result.jpg"
        cv2.imwrite(str(out_path), annotated)

    csv_path = out_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "status", "class", "box", "confidence", "ocr_text", "aliases"])
        writer.writerows(csv_rows)
    print(f"Saved annotated images and {csv_path}")


if __name__ == "__main__":
    main()
