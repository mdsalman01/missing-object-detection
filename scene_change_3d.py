"""What changed in the room? Compare two handheld recordings of the same space in 3D.

1. Sample frames from both videos.
2. VGGT reconstructs every frame of BOTH videos in one pass, so they share one
   3D coordinate frame (camera poses + per-pixel depth) with no separate
   alignment step.
3. YOLO-World detects objects; CLIP describes and names them (reused from
   find_missing.py).
4. Each object is placed in 3D from the depth inside its boxes.
5. Before/after objects are matched by appearance, then compared by position:
   moved / added / removed / unchanged.
6. "Removed" and "added" are only claimed if the other recording actually
   looked at that spot with nothing in front of it. Otherwise the object is
   reported as "not observed" -- missing from view is not the same as gone.

Outputs: an interactive 3D view, a plain-English report, changes.json, and
before/after evidence crops.
"""
import argparse
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

import find_missing as fm

VGGT_SIZE = 518


# ---------------------------------------------------------------- frames

class FrameMap:
    """Maps boxes between full-resolution video frames and VGGT's resized
    grid (width 518, height rounded to a multiple of 14, centre-cropped to
    518 if taller -- the same rule as vggt.utils.load_fn's "crop" mode)."""

    def __init__(self, full_shape: tuple):
        h0, w0 = full_shape[:2]
        new_h = round(h0 * (VGGT_SIZE / w0) / 14) * 14
        self.sx = VGGT_SIZE / w0
        self.sy = new_h / h0
        self.off_y = (new_h - VGGT_SIZE) // 2 if new_h > VGGT_SIZE else 0

    def to_grid(self, box: tuple) -> tuple:
        x1, y1, x2, y2 = box
        return x1 * self.sx, y1 * self.sy - self.off_y, x2 * self.sx, y2 * self.sy - self.off_y

    def to_full(self, box: tuple) -> tuple:
        x1, y1, x2, y2 = box
        return x1 / self.sx, (y1 + self.off_y) / self.sy, x2 / self.sx, (y2 + self.off_y) / self.sy


def save_frames(frames: list, out_dir: Path, prefix: str) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i, frame in enumerate(frames):
        p = out_dir / f"{prefix}_{i:02d}.jpg"
        cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        paths.append(p)
    return paths


# ---------------------------------------------------------------- 3D reconstruction

def reconstruct(frame_paths: list[Path], cache: Path) -> dict:
    """Run VGGT on all frames of both videos together. Cached, because it is
    by far the slowest step on a CPU."""
    if cache.exists():
        return dict(np.load(cache))
    import torch
    from vggt.models.vggt import VGGT
    from vggt.utils.geometry import unproject_depth_map_to_point_map
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri

    model = VGGT.from_pretrained("facebook/VGGT-1B").eval()
    # Depth + cameras are all we use; skipping the point and track heads saves time.
    model.point_head = None
    model.track_head = None
    images = load_and_preprocess_images([str(p) for p in frame_paths])
    with torch.no_grad():
        pred = model(images)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    extrinsic, intrinsic = extrinsic[0].numpy(), intrinsic[0].numpy()
    depth = pred["depth"][0].numpy()
    recon = {
        "extrinsic": extrinsic,  # camera-from-world, OpenCV convention, (S, 3, 4)
        "intrinsic": intrinsic,  # (S, 3, 3), in VGGT grid pixels
        "depth": depth[..., 0],  # (S, H, W) z-depth
        "conf": pred["depth_conf"][0].numpy(),  # (S, H, W)
        "points": unproject_depth_map_to_point_map(depth, extrinsic, intrinsic).astype(np.float32),
        "colors": (images.permute(0, 2, 3, 1).numpy() * 255).astype(np.uint8),
    }
    np.savez_compressed(cache, **recon)
    return recon


def camera_centers(recon: dict) -> np.ndarray:
    R, t = recon["extrinsic"][:, :, :3], recon["extrinsic"][:, :, 3]
    return -np.einsum("sji,sj->si", R, t)


def confident_mask(recon: dict, g: int) -> np.ndarray:
    conf = recon["conf"][g]
    return conf >= np.median(conf)


def scene_scale(recon: dict) -> float:
    """Diagonal of the 5-95th percentile box of the confident point cloud --
    a size for the whole scene that move thresholds are a fraction of, since
    VGGT's units are arbitrary."""
    pts = np.concatenate([recon["points"][g][confident_mask(recon, g)] for g in range(len(recon["points"]))])
    lo, hi = np.percentile(pts, [5, 95], axis=0)
    return float(np.linalg.norm(hi - lo))


def box_points(recon: dict, g: int, grid_box: tuple, inner: float = 0.6) -> np.ndarray:
    """3D points inside the central part of a box (the outer margin of a
    detection box is mostly background), keeping the more confident half."""
    H, W = recon["depth"].shape[1:]
    x1, y1, x2, y2 = grid_box
    cx, cy, hw, hh = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) * inner / 2, (y2 - y1) * inner / 2
    xs, xe = int(max(0, cx - hw)), int(min(W, cx + hw + 1))
    ys, ye = int(max(0, cy - hh)), int(min(H, cy + hh + 1))
    if xe <= xs or ye <= ys:
        return np.zeros((0, 3), np.float32)
    pts = recon["points"][g, ys:ye, xs:xe].reshape(-1, 3)
    conf = recon["conf"][g, ys:ye, xs:xe].reshape(-1)
    return pts[conf >= np.median(conf)]


def project(recon: dict, g: int, p: np.ndarray):
    """World point -> (u, v, z) in frame g's VGGT grid, or None if behind the camera."""
    E, K = recon["extrinsic"][g], recon["intrinsic"][g]
    pc = E[:, :3] @ p + E[:, 3]
    if pc[2] <= 1e-6:
        return None
    uvw = K @ pc
    return uvw[0] / uvw[2], uvw[1] / uvw[2], pc[2]


def observed_in(recon: dict, frame_ids: list[int], p: np.ndarray, tol: float = 0.12, margin: float = 0.03) -> list[int]:
    """Frames whose camera saw world point p: inside the image (away from the
    border) and not hidden behind something nearer. If the depth at that
    pixel is much smaller than p's depth, something is in front of it."""
    H, W = recon["depth"].shape[1:]
    seen = []
    for g in frame_ids:
        proj = project(recon, g, p)
        if proj is None:
            continue
        u, v, z = proj
        if not (margin * W <= u < (1 - margin) * W and margin * H <= v < (1 - margin) * H):
            continue
        patch = recon["depth"][g, max(0, int(v) - 1):int(v) + 2, max(0, int(u) - 1):int(u) + 2]
        if np.median(patch) >= z * (1 - tol):
            seen.append(g)
    return seen


# ---------------------------------------------------------------- objects

@dataclass
class Object3D:
    video: str  # "before" | "after"
    group: list
    emb: np.ndarray
    name: str = ""
    yolo_labels: list = field(default_factory=list)
    position: np.ndarray = None
    extent: tuple = None  # (lo, hi) corners of a robust 3D box
    n_views: int = 0
    points: np.ndarray = None  # subsample of the 3D surface points seen inside its boxes
    only_partial: bool = False  # every sighting was cut off by the frame edge


def truncated(box: tuple, frame_shape: tuple, margin: float = 0.01) -> bool:
    """Box touching the frame edge: the object is cut off, so the box centre
    is not the object's centre (a notepad half out of frame "moves" with
    the camera)."""
    h, w = frame_shape[:2]
    x1, y1, x2, y2 = box
    return x1 <= margin * w or y1 <= margin * h or x2 >= (1 - margin) * w or y2 >= (1 - margin) * h


def place_objects(groups: list, embs: np.ndarray, video: str, frame_offset: int, fmap: FrameMap,
                  recon: dict, matcher: fm.ClipMatcher, frame_shape: tuple) -> list[Object3D]:
    objects = []
    for group, emb in zip(groups, embs):
        sightings = [(f, b) for t in group for f, b in zip(t.frame_ids, t.boxes)]
        whole = [(f, b) for f, b in sightings if not truncated(b, frame_shape)]
        centres, pooled = [], []
        for local_f, box in whole or sightings:
            pts = box_points(recon, frame_offset + local_f, fmap.to_grid(box))
            if len(pts) >= 10:
                centres.append(np.median(pts, axis=0))
                pooled.append(pts[:: max(1, len(pts) // 200)])
        if not centres:
            continue
        pooled = np.concatenate(pooled)
        objects.append(Object3D(
            video=video, group=group, emb=emb, name=matcher.name(emb),
            yolo_labels=sorted({t.class_name for t in group}),
            position=np.median(np.stack(centres), axis=0),
            extent=tuple(np.percentile(pooled, [15, 85], axis=0)),
            n_views=len(centres),
            points=pooled[:: max(1, len(pooled) // 3000)],
            only_partial=not whole,
        ))
    return objects


def surface_overlap(a: np.ndarray, b: np.ndarray, radius: float) -> float:
    """Fraction of the smaller point set lying within `radius` of the other.
    Two partial views of the same, unmoved object overlap in 3D even when
    their box centres disagree; a moved object's surfaces don't overlap."""
    from scipy.spatial import cKDTree
    if a is None or b is None or not len(a) or not len(b):
        return 0.0
    small, big = (a, b) if len(a) <= len(b) else (b, a)
    d, _ = cKDTree(big).query(small, distance_upper_bound=radius)
    return float(np.mean(np.isfinite(d)))


def box_iou_over_shared_frames(g1: list, g2: list) -> tuple[float, int]:
    """Median best-box IoU between two groups of tracks from the SAME video,
    over the frames where both were detected."""
    def by_frame(group):
        out = {}
        for t in group:
            for f, b in zip(t.frame_ids, t.boxes):
                out.setdefault(f, []).append(b)
        return out
    f1, f2 = by_frame(g1), by_frame(g2)
    shared = f1.keys() & f2.keys()
    if not shared:
        return 0.0, 0
    ious = [max(fm.iou(a, b) for a in f1[f] for b in f2[f]) for f in shared]
    return float(np.median(ious)), len(shared)


def raw_candidates(dets: list, frames: list, frame_offset: int, fmap: FrameMap, recon: dict,
                   matcher: fm.ClipMatcher) -> list[dict]:
    """Every individual detection in a video, embedded and placed in 3D --
    the second chance for objects that never formed a stable track."""
    views, meta = [], []
    for local_f, (frame, frame_dets) in enumerate(zip(frames, dets)):
        for _c, box, _s in frame_dets:
            if truncated(box, frame.shape):
                continue
            views.append((frame, box))
            meta.append((local_f, box))
    embs = matcher.embed(views)
    out = []
    for (local_f, box), emb in zip(meta, embs):
        pts = box_points(recon, frame_offset + local_f, fmap.to_grid(box))
        if len(pts) >= 10:
            out.append({"emb": emb, "position": np.median(pts, axis=0), "frame": local_f, "box": box,
                        "points": pts[:: max(1, len(pts) // 1000)]})
    return out


# ---------------------------------------------------------------- change detection

def describe_place(p: np.ndarray, anchors: list[Object3D], exclude: str) -> str:
    """Where p is, in terms of nearby objects that did not change."""
    cands = sorted(((float(np.linalg.norm(a.position - p)), a.name) for a in anchors if a.name != exclude))
    names = []
    for _d, n in cands:
        if n not in names:
            names.append(n)
    if not cands:
        return "in the scene"
    if len(cands) > 1 and len(names) > 1:
        d0 = cands[0][0]
        d1 = next(d for d, n in cands if n == names[1])
        if d1 < 1.35 * d0:
            return f"between the {names[0]} and the {names[1]}"
    return f"next to the {names[0]}"


def region_embedding(obj: Object3D, frame_ids: list[int], recon: dict, frames_all: list, fmap: FrameMap,
                     matcher: fm.ClipMatcher, max_views: int = 6):
    """Mean CLIP embedding of the object's 3D box as seen from the given
    frames (only frames with a clear view of that spot)."""
    views = []
    for g in observed_in(recon, frame_ids, obj.position)[:max_views]:
        pb = projected_box(recon, g, obj.extent, fmap)
        if pb is not None:
            # The 3D extent comes from the central part of each detection box, so
            # pad it back out to roughly the framing a detection box would have.
            x1, y1, x2, y2 = pb
            px, py = (x2 - x1) * 0.4, (y2 - y1) * 0.4
            views.append((frames_all[g], (x1 - px, y1 - py, x2 + px, y2 + py)))
    if not views:
        return None, 0
    e = matcher.embed(views).mean(axis=0)
    return e / (np.linalg.norm(e) or 1.0), len(views)


def spot_similarity(obj: Object3D, own_ids: list[int], other_ids: list[int], recon: dict, frames_all: list,
                    fmap: FrameMap, matcher: fm.ClipMatcher) -> tuple[float, int]:
    """Did this spot change? Crop the same 3D region from both recordings and
    compare the crops with each other. This is the 3D counterpart of
    comparing the same pixels in two fixed-camera shots, and unlike
    comparing against the object's own fingerprint, it doesn't depend on how
    well (or how partially) the detector tracked the object."""
    here, _ = region_embedding(obj, own_ids, recon, frames_all, fmap, matcher)
    there, n = region_embedding(obj, other_ids, recon, frames_all, fmap, matcher)
    if here is None or there is None:
        return 0.0, n
    return float(here @ there), n


def object_drop(obj: Object3D, own_ids: list[int], other_ids: list[int], recon: dict, frames_all: list,
                fmap: FrameMap, matcher: fm.ClipMatcher) -> tuple[float, float]:
    """How much less the spot resembles THIS object in the other recording.
    Region-vs-region similarity misses a small object on a large,
    distinctive background (a mouse on a yellow notepad: the notepad
    dominates both crops). Comparing each crop with the object's own
    fingerprint asks the sharper question: is the mouse still there?"""
    here, _ = region_embedding(obj, own_ids, recon, frames_all, fmap, matcher)
    there, _ = region_embedding(obj, other_ids, recon, frames_all, fmap, matcher)
    if here is None or there is None:
        return 0.0, 0.0
    return float(obj.emb @ here), float(obj.emb @ there)


def detect_changes(before: list[Object3D], after: list[Object3D], raw_before: list[dict], raw_after: list[dict],
                   recon: dict, before_ids: list[int], after_ids: list[int], scale: float, args,
                   frames_all: list, fmap: FrameMap, matcher: fm.ClipMatcher,
                   log: list | None = None) -> tuple:
    """Classify every object. `log` collects one line per decision, so a
    surprising result can be traced back to the number that caused it."""
    move_thr = args.move_frac * scale
    changes, unchanged = [], []
    log = [] if log is None else log

    def note(o: Object3D, msg: str) -> None:
        log.append(f"[{o.video:6}] {o.name:15} {'/'.join(o.yolo_labels)[:26]:26} {msg}")

    def moved_far(a: Object3D, b_pos: np.ndarray, b_points: np.ndarray | None = None) -> tuple[bool, float]:
        """Moved = centre displaced beyond the scene threshold AND the two
        recordings' surfaces of the object no longer overlap in 3D. The
        overlap test is what keeps a big, often half-out-of-frame object (a
        notepad) from "moving" just because its visible part changed."""
        d = float(np.linalg.norm(a.position - b_pos))
        if d <= move_thr:
            return False, d
        if b_points is not None and surface_overlap(a.points, b_points, args.overlap_radius * scale) >= args.overlap_same:
            return False, d
        return True, d

    def move_confirmed(a: Object3D, b: Object3D) -> bool:
        """A real move changes at least one of the two spots: the old one
        empties or the new one gains the object. If both spots look the same
        in the two recordings, the appearance match paired two different,
        stationary objects."""
        old, _ = spot_similarity(a, before_ids, after_ids, recon, frames_all, fmap, matcher)
        if old < args.spot_thr:
            return True
        new, _ = spot_similarity(b, after_ids, before_ids, recon, frames_all, fmap, matcher)
        return new < args.spot_thr

    sims = np.stack([b.emb for b in before]) @ np.stack([a.emb for a in after]).T if before and after \
        else np.zeros((len(before), len(after)))
    matched_b, matched_a = set(), set()
    if sims.size:
        for i, j in zip(*linear_sum_assignment(-sims)):
            if sims[i, j] < args.clip_thr:
                continue
            matched_b.add(i)
            matched_a.add(j)
            far, d = moved_far(before[i], after[j].position, after[j].points)
            if far and not move_confirmed(before[i], after[j]):
                unchanged.append(before[i])
                note(before[i], f"pairing with '{after[j].name}' rejected: both its old and new spot look the same "
                                f"in the two recordings, so nothing actually moved")
            elif far:
                changes.append({"type": "moved", "obj": before[i], "to": after[j].position, "after_obj": after[j],
                                "similarity": float(sims[i, j]), "distance": d})
                note(before[i], f"MOVED: matched '{after[j].name}' sim {sims[i, j]:.3f}, shift {d / scale:.3f} x scene")
            else:
                unchanged.append(before[i])
                note(before[i], f"unchanged: matched '{after[j].name}' sim {sims[i, j]:.3f}, shift {d / scale:.3f}")

    def second_chance(obj: Object3D, raw: list[dict]):
        """Best raw detection in the other video that looks like obj, if any."""
        if not raw:
            return None, 0.0
        s = np.stack([r["emb"] for r in raw]) @ obj.emb
        k = int(np.argmax(s))
        return raw[k], float(s[k])

    # Second chance against every raw detection in the other recording.
    left_b, left_a = [], []
    best_b = {}
    for i, obj in enumerate(before):
        if i in matched_b:
            continue
        cand, s = second_chance(obj, raw_after)
        best_b[id(obj)] = s
        if cand is not None and s >= args.clip_thr:
            far, d = moved_far(obj, cand["position"], cand["points"])
            if not far:
                unchanged.append(obj)
                note(obj, f"present (raw detection in place): sim {s:.3f}, shift {d / scale:.3f}")
                continue
            if s >= args.clip_thr + args.move_margin:
                changes.append({"type": "moved", "obj": obj, "to": cand["position"], "after_raw": cand,
                                "similarity": s, "distance": d})
                note(obj, f"MOVED (raw detection): sim {s:.3f}, shift {d / scale:.3f}")
                continue
            # A look-alike far away proves nothing either way (a dark earbuds
            # case resembles a dark mouse): let the spot check decide.
            note(obj, f"look-alike only far away (sim {s:.3f}, shift {d / scale:.3f}): undecided")
            left_b.append(obj)
            continue
        note(obj, f"no match in other recording (best raw sim {s:.3f})")
        left_b.append(obj)
    best_a = {}
    for j, obj in enumerate(after):
        if j in matched_a:
            continue
        cand, s = second_chance(obj, raw_before)
        best_a[id(obj)] = s
        if cand is not None and s >= args.clip_thr and not moved_far(obj, cand["position"], cand["points"])[0]:
            note(obj, f"was there before (raw detection sim {s:.3f})")
            continue
        if cand is not None and s >= args.clip_thr:
            note(obj, f"a look-alike was elsewhere before (sim {s:.3f}): kept as a possible move destination")
        if any(c["type"] == "moved" and np.linalg.norm(c["to"] - obj.position) <= move_thr for c in changes):
            note(obj, "destination of a reported move")
            continue
        else:
            note(obj, f"no match in other recording (best raw sim {s:.3f})")
        left_a.append(obj)

    # A leftover whose boxes coincide with an already-explained object's boxes
    # in the same frames is a fragment of it: the same pixels, tracked twice
    # under different labels. Tested in 2D on purpose -- in 3D a pen lying
    # on a notebook shares the notebook's surface and would wrongly count.
    explained_b = [o for o in before if o not in left_b]
    explained_a = [o for o in after if o not in left_a]

    def fragment(obj: Object3D, explained: list[Object3D]) -> bool:
        for u in explained:
            overlap, shared = box_iou_over_shared_frames(obj.group, u.group)
            if shared and overlap >= args.fragment_iou:
                note(obj, f"dropped: same pixels as '{u.name}' (median IoU {overlap:.2f} over {shared} frame(s))")
                return True
        return False
    left_b = [o for o in left_b if not fragment(o, explained_b)]
    left_a = [o for o in left_a if not fragment(o, explained_a)]

    # A removal and an addition of similar-looking objects are more likely one
    # move. Appearance shifts when the background behind an object changes
    # (a pen on a blue notebook vs on a wooden desk), so this pairing uses a
    # lower similarity bar than "same object".
    if left_b and left_a:
        s = np.stack([o.emb for o in left_b]) @ np.stack([o.emb for o in left_a]).T
        used_b, used_a = set(), set()
        for i, j in zip(*linear_sum_assignment(-s)):
            far, d = moved_far(left_b[i], left_a[j].position, left_a[j].points)
            if s[i, j] >= args.pair_thr and far and move_confirmed(left_b[i], left_a[j]):
                changes.append({"type": "moved", "obj": left_b[i], "to": left_a[j].position,
                                "after_obj": left_a[j], "similarity": float(s[i, j]), "distance": d})
                used_b.add(i)
                used_a.add(j)
                note(left_b[i], f"MOVED (paired with leftover '{left_a[j].name}'): sim {s[i, j]:.3f}, shift {d / scale:.3f}")
            else:
                note(left_b[i], f"not paired with '{left_a[j].name}': sim {s[i, j]:.3f}, shift {d / scale:.3f}")
        left_b = [o for i, o in enumerate(left_b) if i not in used_b]
        left_a = [o for j, o in enumerate(left_a) if j not in used_a]

    for objs, own_ids, other_ids, verb, best in [(left_b, before_ids, after_ids, "removed", best_b),
                                                 (left_a, after_ids, before_ids, "added", best_a)]:
        for obj in objs:
            if obj.only_partial:
                note(obj, "skipped: only ever seen cut off at the frame edge, too unreliable to report")
                continue
            seen = observed_in(recon, other_ids, obj.position)
            if len(seen) < args.min_observed:
                changes.append({"type": "not observed", "obj": obj, "seen_in": seen,
                                "similarity": best.get(id(obj), 0.0)})
                note(obj, f"NOT OBSERVED: its spot is clearly visible in only {len(seen)} frame(s) of the other recording")
                continue
            spot, n = spot_similarity(obj, own_ids, other_ids, recon, frames_all, fmap, matcher)
            obj_here, obj_there = object_drop(obj, own_ids, other_ids, recon, frames_all, fmap, matcher)
            drop = obj_here - obj_there
            evidence_txt = (f"region sim {spot:.3f}; looks like the {obj.name}: {obj_here:.3f} here vs "
                            f"{obj_there:.3f} there (drop {drop:.3f}); {n} views")
            if spot >= args.spot_thr and drop < args.drop_thr:
                unchanged.append(obj)
                note(obj, f"present: its spot looks the same in both recordings ({evidence_txt})")
                continue
            changes.append({"type": verb, "obj": obj, "seen_in": seen, "similarity": max(spot, best.get(id(obj), 0.0))})
            note(obj, f"{verb.upper()}: spot visible in {len(seen)} frame(s) and it changed ({evidence_txt})")

    # One report per physical object: a split group can surface the same
    # object twice at the same spot.
    deduped = []
    for c in changes:
        p = c["obj"].position
        if any(o["type"] == c["type"] and o["obj"].name == c["obj"].name
               and np.linalg.norm(o["obj"].position - p) <= move_thr for o in deduped):
            continue
        deduped.append(c)
    return deduped, unchanged


def sentence(c: dict, anchors: list[Object3D]) -> str:
    name = c["obj"].name
    if c["type"] == "moved":
        return (f"The {name} moved from {describe_place(c['obj'].position, anchors, name)} "
                f"to {describe_place(c['to'], anchors, name)}.")
    if c["type"] == "removed":
        return f"The {name} was removed from {describe_place(c['obj'].position, anchors, name)}."
    if c["type"] == "added":
        return f"A {name} was added {describe_place(c['obj'].position, anchors, name)}."
    other = "second" if c["obj"].video == "before" else "first"
    return (f"The {name} {describe_place(c['obj'].position, anchors, name)} was not seen in the {other} recording, "
            f"but that recording never had a clear view of its spot, so this is not reported as a change.")


# ---------------------------------------------------------------- evidence images

def crop_full(frame: np.ndarray, box: tuple, pad: float = 0.25) -> np.ndarray:
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = box
    px, py = (x2 - x1) * pad, (y2 - y1) * pad
    x1, y1, x2, y2 = int(max(0, x1 - px)), int(max(0, y1 - py)), int(min(W, x2 + px)), int(min(H, y2 + py))
    return frame[y1:y2, x1:x2]


def projected_box(recon: dict, g: int, extent: tuple, fmap: FrameMap):
    """Project an object's 3D box into frame g; returns a full-res 2D box."""
    lo, hi = extent
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    uv = [project(recon, g, c) for c in corners]
    if any(p is None for p in uv):
        return None
    us, vs = [p[0] for p in uv], [p[1] for p in uv]
    return fmap.to_full((min(us), min(vs), max(us), max(vs)))


def best_frame(recon: dict, frame_ids: list[int], p: np.ndarray) -> int | None:
    """Of the frames that saw p, the one where it sits closest to the centre."""
    H, W = recon["depth"].shape[1:]
    best, best_d = None, None
    for g in observed_in(recon, frame_ids, p):
        u, v, _z = project(recon, g, p)
        d = (u / W - 0.5) ** 2 + (v / H - 0.5) ** 2
        if best_d is None or d < best_d:
            best, best_d = g, d
    return best


def evidence(c: dict, frames_all: list, recon: dict, fmap: FrameMap, before_ids: list, after_ids: list,
             out_path: Path) -> None:
    """Side-by-side crop: the object in one recording, and the same 3D spot
    (or where it went) in the other."""
    obj = c["obj"]
    own_ids = before_ids if obj.video == "before" else after_ids
    other_ids = after_ids if obj.video == "before" else before_ids
    offset = own_ids[0]
    frame, box = fm.best_view(obj.group[0] if len(obj.group) == 1 else max(obj.group, key=lambda t: t.score_sum),
                              frames_all[offset:offset + len(own_ids)])
    left = crop_full(frame, box)

    target = c["to"] if c["type"] == "moved" else obj.position
    if c["type"] == "moved" and "after_obj" in c:
        a = c["after_obj"]
        rframe, rbox = fm.best_view(max(a.group, key=lambda t: t.score_sum),
                                    frames_all[after_ids[0]:after_ids[0] + len(after_ids)])
        right = crop_full(rframe, rbox)
    elif c["type"] == "moved" and "after_raw" in c:
        r = c["after_raw"]
        right = crop_full(frames_all[after_ids[0] + r["frame"]], r["box"])
    else:
        g = best_frame(recon, other_ids, target)
        right = None
        if g is not None:
            shift = target - obj.position
            pb = projected_box(recon, g, (obj.extent[0] + shift, obj.extent[1] + shift), fmap)
            if pb is not None:
                right = crop_full(frames_all[g], pb, pad=0.6)
    tiles = [left] + ([right] if right is not None and right.size else [])
    h = 360
    tiles = [cv2.resize(t, (max(1, int(t.shape[1] * h / t.shape[0])), h)) for t in tiles if t.size]
    if not tiles:
        return
    labels = ["before", "after"] if obj.video == "before" else ["after", "before"]
    for t, lab in zip(tiles, labels):
        cv2.putText(t, lab, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
        cv2.putText(t, lab, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (20, 20, 20), 1)
    gap = np.full((h, 12, 3), 255, np.uint8)
    cv2.imwrite(str(out_path), np.hstack([x for t in tiles for x in (t, gap)][:-1]))


# ---------------------------------------------------------------- 3D view

COLORS = {"moved": "#e69500", "removed": "#d6402f", "added": "#2e9e5b", "not observed": "#8a8f98"}


def write_3d_view(recon: dict, before_ids: list, after_ids: list, changes: list, unchanged: list,
                  out_path: Path, max_points: int = 60000) -> None:
    import plotly.graph_objects as go

    def cloud(ids, name, visible):
        pts, cols = [], []
        for g in ids:
            m = confident_mask(recon, g)
            pts.append(recon["points"][g][m])
            cols.append(recon["colors"][g][m])
        pts, cols = np.concatenate(pts), np.concatenate(cols)
        step = max(1, len(pts) // (max_points // 2))
        pts, cols = pts[::step], cols[::step]
        return go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers", name=name, visible=visible,
                            marker=dict(size=1.6, color=[f"rgb({r},{g},{b})" for r, g, b in cols]),
                            hoverinfo="skip")

    centres = camera_centers(recon)
    traces = [cloud(before_ids, "scene: before", True), cloud(after_ids, "scene: after", "legendonly")]
    for ids, name, col in [(before_ids, "camera path: before", "#3b6fd1"), (after_ids, "camera path: after", "#b04fc4")]:
        c = centres[ids]
        traces.append(go.Scatter3d(x=c[:, 0], y=c[:, 1], z=c[:, 2], mode="lines+markers", name=name,
                                   line=dict(color=col, width=4), marker=dict(size=3, color=col)))
    if unchanged:
        u = np.stack([o.position for o in unchanged])
        traces.append(go.Scatter3d(x=u[:, 0], y=u[:, 1], z=u[:, 2], mode="markers+text", name="unchanged",
                                   text=[o.name for o in unchanged], textfont=dict(size=10, color="#5c6370"),
                                   marker=dict(size=4, color="#5c6370", symbol="circle-open")))
    for c in changes:
        col, p = COLORS[c["type"]], c["obj"].position
        label = f"{c['type'].upper()}: {c['obj'].name}"
        traces.append(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers+text", name=label,
                                   text=[label], textposition="top center", textfont=dict(color=col, size=13),
                                   marker=dict(size=9, color=col, symbol="diamond"),
                                   hovertext=c["sentence"], hoverinfo="text"))
        lo, hi = c["obj"].extent
        xs, ys, zs = [], [], []
        for a, b in [((0, 0, 0), (1, 0, 0)), ((0, 1, 0), (1, 1, 0)), ((0, 0, 1), (1, 0, 1)), ((0, 1, 1), (1, 1, 1)),
                     ((0, 0, 0), (0, 1, 0)), ((1, 0, 0), (1, 1, 0)), ((0, 0, 1), (0, 1, 1)), ((1, 0, 1), (1, 1, 1)),
                     ((0, 0, 0), (0, 0, 1)), ((1, 0, 0), (1, 0, 1)), ((0, 1, 0), (0, 1, 1)), ((1, 1, 0), (1, 1, 1))]:
            for corner in (a, b):
                q = [hi[k] if corner[k] else lo[k] for k in range(3)]
                xs.append(q[0]); ys.append(q[1]); zs.append(q[2])
            xs.append(None); ys.append(None); zs.append(None)
        traces.append(go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=col, width=3),
                                   showlegend=False, hoverinfo="skip"))
        if c["type"] == "moved":
            q = c["to"]
            traces.append(go.Scatter3d(x=[p[0], q[0]], y=[p[1], q[1]], z=[p[2], q[2]], mode="lines",
                                       line=dict(color=col, width=6, dash="dash"), showlegend=False, hoverinfo="skip"))
            traces.append(go.Cone(x=[q[0]], y=[q[1]], z=[q[2]], u=[q[0] - p[0]], v=[q[1] - p[1]], w=[q[2] - p[2]],
                                  sizemode="absolute", sizeref=float(np.linalg.norm(q - p)) * 0.25, anchor="tip",
                                  colorscale=[[0, col], [1, col]], showscale=False, hoverinfo="skip"))
    fig = go.Figure(traces)
    fig.update_layout(
        title="What changed: shared 3D reconstruction of both recordings",
        scene=dict(aspectmode="data", xaxis_visible=False, yaxis_visible=False, zaxis_visible=False,
                   camera=dict(up=dict(x=0, y=-1, z=0), eye=dict(x=0.0, y=-0.9, z=-1.6))),
        legend=dict(itemsizing="constant"), margin=dict(l=0, r=0, t=40, b=0),
    )
    fig.write_html(out_path, include_plotlyjs="cdn")


def write_change_map(recon: dict, before_ids: list, after_ids: list, changes: list, unchanged: list,
                     out_path: Path, title: str) -> None:
    """Static bird's-eye view for places that can't show the interactive one
    (a GitHub README). "Down" is the flattest direction of the point cloud,
    which for a room or desk scene is the floor/table normal."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pts, cols = [], []
    for g in before_ids + after_ids:
        m = confident_mask(recon, g)
        pts.append(recon["points"][g][m][::7])
        cols.append(recon["colors"][g][m][::7])
    pts, cols = np.concatenate(pts), np.concatenate(cols) / 255
    centre = np.median(pts, axis=0)
    _u, _s, vt = np.linalg.svd(pts[::5] - centre, full_matrices=False)
    e1, e2, normal = vt[0], vt[1], vt[2]
    cams = camera_centers(recon)
    if np.dot(np.median(cams, axis=0) - centre, normal) < 0:
        normal = -normal  # look from the side the cameras were on
    # Keep the image the way a person filming would see it: e1 left->right,
    # e2 pointing away from the camera.
    if np.dot(np.median(cams, axis=0) - centre, e2) > 0:
        e2 = -e2
    if np.dot(np.cross(e1, e2), normal) < 0:
        e1 = -e1

    def flat(p):
        q = np.atleast_2d(p) - centre
        return np.stack([q @ e1, q @ e2], axis=1)

    above = (pts - centre) @ normal
    order = np.argsort(above)  # draw higher points last so objects sit on top of the desk
    xy = flat(pts)[order]
    fig, ax = plt.subplots(figsize=(11, 8))
    ax.scatter(xy[:, 0], xy[:, 1], c=cols[order], s=0.6, linewidths=0)
    for ids, col, lab in [(before_ids, "#3b6fd1", "camera path, recording 1"),
                          (after_ids, "#b04fc4", "camera path, recording 2")]:
        c = flat(cams[ids])
        ax.plot(c[:, 0], c[:, 1], "-o", color=col, ms=3, lw=2, label=lab)
    for o in unchanged:
        q = flat(o.position)[0]
        ax.annotate(o.name, q, fontsize=7, color="#3a3f47", ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6))
    for c in changes:
        if c["type"] == "not observed":
            continue
        col = COLORS[c["type"]]
        q = flat(c["obj"].position)[0]
        if c["type"] == "moved":
            r = flat(c["to"])[0]
            ax.annotate("", xy=r, xytext=q, arrowprops=dict(arrowstyle="-|>", color=col, lw=2.5, ls="--"))
            ax.scatter(*q, s=160, facecolors="none", edgecolors=col, linewidths=2.5)
            ax.scatter(*r, s=160, color=col, edgecolors="white", linewidths=1.5)
            q = r
        else:
            ax.scatter(*q, s=220, color=col, marker="X" if c["type"] == "removed" else "P",
                       edgecolors="white", linewidths=1.5)
        ax.annotate(f"{c['type'].upper()}: {c['obj'].name}", q, xytext=(10, 10), textcoords="offset points",
                    fontsize=11, fontweight="bold", color=col,
                    bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=col, alpha=0.9))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    ax.set_title(title, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description="Detect moved / added / removed objects between two recordings, in 3D.")
    ap.add_argument("--before", required=True, help="First recording (reference)")
    ap.add_argument("--after", required=True, help="Second recording")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-frames", type=int, default=12, help="Frames sampled per video (VGGT sees 2x this)")
    ap.add_argument("--model", default="yolov8s-worldv2.pt")
    ap.add_argument("--vocab", default=None)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--min-frac", type=float, default=0.2)
    ap.add_argument("--hist-thr", type=float, default=0.65)
    ap.add_argument("--clip-thr", type=float, default=0.855, help="CLIP similarity for 'same object'")
    ap.add_argument("--move-frac", type=float, default=0.06, help="Displacement, as a fraction of scene size, that counts as moved")
    ap.add_argument("--move-margin", type=float, default=0.04,
                    help="Extra similarity a single stray detection needs before it can claim an object moved")
    ap.add_argument("--min-observed", type=int, default=2, help="Clear views of a spot needed to call it removed/added")
    ap.add_argument("--pair-thr", type=float, default=0.82,
                    help="Similarity at which a would-be removal and a would-be addition are paired as one move")
    ap.add_argument("--spot-thr", type=float, default=0.855,
                    help="Similarity of the same 3D region across recordings at or above which nothing changed there")
    ap.add_argument("--drop-thr", type=float, default=0.05,
                    help="Drop in resemblance to the object, between its spot in each recording, that counts as changed")
    ap.add_argument("--fragment-iou", type=float, default=0.5,
                    help="2D box overlap with an explained object at which a leftover counts as a fragment of it")
    ap.add_argument("--overlap-radius", type=float, default=0.015,
                    help="Distance (fraction of scene size) within which two surface points count as the same place")
    ap.add_argument("--overlap-same", type=float, default=0.3,
                    help="Surface overlap at or above which an object counts as not moved")
    ap.add_argument("--explain", action="store_true", help="Print the decision taken for every object (always saved to decisions.log)")
    args = ap.parse_args()

    out = Path(args.out_dir)
    (out / "evidence").mkdir(parents=True, exist_ok=True)
    for stale in (out / "evidence").glob("*.jpg"):
        stale.unlink()

    print("Sampling frames")
    before_frames = fm.read_sampled_frames(Path(args.before), args.n_frames)
    after_frames = fm.read_sampled_frames(Path(args.after), args.n_frames)
    frames_all = before_frames + after_frames
    before_ids = list(range(len(before_frames)))
    after_ids = list(range(len(before_frames), len(frames_all)))
    paths = save_frames(before_frames, out / "frames", "before") + save_frames(after_frames, out / "frames", "after")
    fmap = FrameMap(before_frames[0].shape)
    if FrameMap(after_frames[0].shape).sy != fmap.sy:
        raise SystemExit("Both recordings must have the same resolution and orientation.")

    print(f"Reconstructing {len(paths)} frames jointly with VGGT (slow on CPU; cached afterwards)")
    recon = reconstruct(paths, out / "reconstruction.npz")
    scale = scene_scale(recon)

    print("Detecting objects")
    det_cache = out / "detections.pkl"
    if det_cache.exists():
        before_dets, after_dets = pickle.loads(det_cache.read_bytes())
    else:
        from ultralytics import YOLO
        model = YOLO(args.model)
        model.set_classes(fm.load_vocab(args.vocab))
        before_dets = fm.detect_all_frames(model, before_frames, args.conf, args.imgsz)
        after_dets = fm.detect_all_frames(model, after_frames, args.conf, args.imgsz)
        det_cache.write_bytes(pickle.dumps((before_dets, after_dets)))

    print("Describing objects with CLIP and placing them in 3D")
    matcher = fm.ClipMatcher(fm.load_vocab(args.vocab))
    objects = {}
    for video, frames, dets, offset in [("before", before_frames, before_dets, 0),
                                        ("after", after_frames, after_dets, len(before_frames))]:
        tracks = fm.stable_tracks(fm.build_tracks_by_appearance(dets, frames, args.hist_thr), len(frames), args.min_frac)
        groups = fm.group_alias_tracks(tracks)
        embs = np.stack([matcher.embed_group(g, frames) for g in groups]) if groups else np.zeros((0, 512))
        objects[video] = place_objects(groups, embs, video, offset, fmap, recon, matcher, frames[0].shape)
    raw_before = raw_candidates(before_dets, before_frames, 0, fmap, recon, matcher)
    raw_after = raw_candidates(after_dets, after_frames, len(before_frames), fmap, recon, matcher)

    log: list[str] = []
    changes, unchanged = detect_changes(objects["before"], objects["after"], raw_before, raw_after,
                                        recon, before_ids, after_ids, scale, args, frames_all, fmap, matcher, log)
    (out / "decisions.log").write_text("\n".join(log) + "\n", encoding="utf-8")
    if args.explain:
        print("\n".join(log))
    for c in changes:
        c["sentence"] = sentence(c, unchanged)

    print("Writing outputs")
    order = {"moved": 0, "removed": 1, "added": 2, "not observed": 3}
    changes.sort(key=lambda c: order[c["type"]])
    records = []
    for n, c in enumerate(changes):
        img = out / "evidence" / f"{n:02d}_{c['type'].replace(' ', '_')}_{c['obj'].name.replace(' ', '_')}.jpg"
        evidence(c, frames_all, recon, fmap, before_ids, after_ids, img)
        rec = {
            "change": c["type"], "object": c["obj"].name, "detector_labels": c["obj"].yolo_labels,
            "description": c["sentence"],
            "position": [round(float(v), 4) for v in c["obj"].position],
            "best_appearance_match": round(float(c.get("similarity", 0.0)), 3),
            "evidence": str(img.relative_to(out)).replace("\\", "/") if img.exists() else None,
        }
        if c["type"] == "moved":
            rec["moved_to"] = [round(float(v), 4) for v in c["to"]]
            rec["distance_fraction_of_scene"] = round(c["distance"] / scale, 3)
        if "seen_in" in c:
            rec["clear_views_in_other_recording"] = len(c["seen_in"])
        records.append(rec)

    (out / "changes.json").write_text(json.dumps({
        "before": args.before, "after": args.after, "frames_per_video": args.n_frames,
        "unchanged_objects": sorted({o.name for o in unchanged}), "changes": records}, indent=2))
    write_3d_view(recon, before_ids, after_ids, changes, unchanged, out / "scene_3d.html")
    write_change_map(recon, before_ids, after_ids, changes, unchanged, out / "change_map.png",
                     f"Top-down view of the shared 3D reconstruction: {Path(args.before).name} vs {Path(args.after).name}")

    lines = [f"# Changes: `{Path(args.before).name}` -> `{Path(args.after).name}`", ""]
    real = [r for r in records if r["change"] != "not observed"]
    lines += [f"- **{r['change']}**: {r['description']}" for r in real] or ["- No changes detected."]
    unseen = [r for r in records if r["change"] == "not observed"]
    if unseen:
        lines += ["", "Not reported as changes (no clear view in the other recording):", ""]
        lines += [f"- {r['description']}" for r in unseen]
    lines += ["", "Interactive 3D view: `scene_3d.html` · top-down map: `change_map.png` · "
                  "details: `changes.json` · crops: `evidence/` · reasoning: `decisions.log`"]
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
