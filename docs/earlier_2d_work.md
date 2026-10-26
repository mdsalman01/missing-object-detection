# Earlier stage: 2D missing-object detection (`find_missing.py`)

> This is the write-up of the project's first stage, kept for context. It
> worked purely in 2D and handled removals only. The 3D system in the main
> [README](../README.md) reuses its detection, alias-grouping and CLIP
> components. The videos it refers to (`vid1.mp4`, `vid2.mp4`) are not
> included in the repository.

Given one parent (reference) video of a scene and one or more test videos of
the same scene, report which objects present in the parent are missing from
each test video.

## Problem

The goal is to detect removed objects automatically from the video pixels
alone, with no prior knowledge of what was removed.

**Current dataset (handheld).** `videos/parent.mp4`, `vid1.mp4`, `vid2.mp4`:
3840x2160, ~60fps, 2.7-3.4s each. A handheld left-to-right pan across a desk
with a laptop, skeleton figurine, green jar, watch, spray bottle, earbuds case,
power bank, Pond's tin, wallet, glasses, notebook + pen, notepad, mouse, coin
jar, water bottle and spray can. Ground truth (`ground_truth.csv`), confirmed
by inspecting sampled frames:

- `vid1.mp4`: the **glasses** are missing
- `vid2.mp4`: the **glasses**, the **mouse**, and the **pen** are missing

Two of the three removed object types (glasses, pen) are **not COCO classes**,
so standard YOLOv8 cannot see them at all. That is what motivated the
YOLO-World mode below.

**Earlier dataset (fixed camera, files no longer present).** Short (~3.7s)
tripod clips of a different desk arrangement; `video1.mp4` was missing a
phone, `video2.mp4` a water bottle. Its ground truth is kept in
`ground_truth_old_desk_scene.csv`, and its results are in the evaluation
section below for reference.

## Approach

1. **`video_info.py`** — print resolution/fps/frame count/duration per video,
   as a sanity check that files load correctly.
2. **`sample_frames.py`** — evenly sample N frames per video (default 12) and
   save them for visual inspection. A short, fixed-camera clip doesn't need
   dense sampling; 12 frames spread across the clip is enough to judge
   whether an object is stably present or stably absent.
3. **`detect_one.py`** — run YOLOv8 on a single frame to sanity-check object
   detection before building the full pipeline.
4. **`find_missing.py`** — the full pipeline:
   - Sample frames per video, run YOLO on each frame.
   - **Dedupe same-class boxes per frame**: YOLO occasionally fires twice on
     one physical object with the same class label (e.g. a bottle body and
     its hanging strap both called "bottle"). Boxes of the same class where
     one sits almost entirely inside the other are merged, keeping the
     higher-confidence box.
   - **Temporal voting**: group each video's detections into tracks (same
     class + overlapping box across frames = same physical object), then
     keep only tracks seen in at least `min_frac` of sampled frames. This
     filters out one-off detector noise and keeps real, stable objects.
   - **Cross-video matching**: for each class, use the Hungarian algorithm
     (`scipy.optimize.linear_sum_assignment`) to find the IoU-maximizing
     pairing between the parent's stable tracks and the test video's stable
     tracks. Unmatched parent tracks are reported missing; unmatched test
     tracks are reported added.
   - **Pixel-diff cross-check**: for each box reported missing, crop the same
     region from a parent and a test frame, convert to grayscale, blur, and
     compare. Because two separate recordings are never pixel-perfectly
     aligned even with a "fixed" camera, the check searches small (dx, dy)
     shifts and keeps the best-aligned comparison before thresholding — this
     is what lets it tell "a few pixels of camera drift" apart from "this
     region actually changed." It is a supplementary human-readable signal,
     not a hard filter on the reported results.
   - **Product-name OCR**: each flagged box is cropped and passed through
     Tesseract (`pytesseract`) to try to read any printed text (e.g.
     "LLOYDS", "120W"), appended to the report as an `ocr_text` column. This
     is best-effort — see Limitations.
   - Saves annotated images (missing = red box, added = green box) to
     `results/` and a `results.csv` with one row per flagged object.
   - **`--camera moving` mode**: for a handheld/moving camera, position-based
     matching (IoU) breaks, because the same object lands at a different
     pixel location in every recording. This mode replaces IoU with a
     color-histogram appearance comparison for *both* within-video tracking
     and cross-video matching (see Approach: moving camera, below).
5. **`evaluate.py`** — compares `results.csv` against `ground_truth.csv` at
   **(video, class)** granularity (not per-box — ground truth is naturally
   "the phone is missing," not a specific box), and reports TP/FP/FN,
   precision, and recall per video and overall.

### Approach: moving camera (`--camera moving`)

A handheld camera means the same physical object sits at a different pixel
position in every recording — and even between two sampled frames of the
*same* recording, since 12 frames spread across a few seconds of handheld
motion can already be a meaningfully different viewpoint. Position (IoU)
stops meaning anything.

The fix is to describe each object by **appearance instead of position**: a
normalized HSV color histogram of its crop. This replaces IoU everywhere —
both `build_tracks_by_appearance` (grouping one video's own detections into
stable tracks) and `match_cross_video_by_appearance` (the Hungarian-algorithm
step pairing parent vs. test tracks) use histogram similarity
(`cv2.HISTCMP_BHATTACHARYYA`, converted to a 0-1 similarity score) as the
cost instead of IoU.

This was calibrated empirically on the real objects in this scene: the same
physical object scored 0.85-0.96 similarity across two separate recordings,
while different objects in the same frame topped out around 0.55 (the
power bank and the watch — both dark and nondescript). `--hist-thr 0.65`
sits in that gap.

### Approach: open vocabulary (YOLO-World)

Passing a `*world*` model (e.g. `--model yolov8s-worldv2.pt`) switches to
YOLO-World, which detects whatever text prompts it is given instead of COCO's
fixed 80 classes. The prompt list (`DEFAULT_VOCAB` in `find_missing.py`, or
`--vocab`) is a **generic list of ~40 everyday desk/household objects**, not
the specific items in this scene, so the tool stays general-purpose.
Honest caveat: the list was written by someone who had seen this scene, and
"glasses", "pen" and "mouse" are on it. They are ordinary desk objects that
belong in any such list, but this is not a blind test of the vocabulary.

A bigger vocabulary brings a new problem: **one physical object gets several
labels**. The glasses come out as `scissors` in 27 of 30 frames and as
`glasses` in only 12; the pen is also `pencil`; the mouse is also
`cell phone`. Each label becomes its own track, so a present object can
look "missing" under whichever label didn't happen to fire in the test video.

**Alias grouping** (`group_alias_tracks`, moving-camera mode) fixes this.
Tracks with different labels are merged into one physical object when, in the
frames where both were seen, their boxes overlap (IoU >= 0.4) at least half
the time. An object then counts as present if *any* of its labels matched,
and a missing object is reported once, under the label with the highest
total detection confidence, with the others listed in an `aliases` column.

IoU (rather than "one box inside the other") is deliberate. A containment
test merged the pen into the notebook it lies on, and the mouse into the
notepad under it. Both removals were then hidden behind an object that was
still present.

Because the reported label is often wrong while the object is right,
`evaluate.py --use-aliases` counts a report as correct if its class *or any
alias* matches the ground truth. Both scores are reported below.

### Approach: CLIP matching (`--match clip`, current best)

Colour histograms and per-label matching caused every remaining false alarm
in the YOLO-World run. `--match clip` (moving camera only) replaces both:

- **Appearance = CLIP image embedding.** Each object (alias group) is
  described by the average CLIP ViT-B/32 embedding of a few of its crops.
  The same object across two recordings scored 0.87-0.99; the watch, which
  colour histograms scored 0.44 because lighting shifted, scores 0.97.
- **Label-agnostic matching.** Parent and test objects are paired by CLIP
  similarity with the Hungarian algorithm, ignoring YOLO's labels.
- **Second chance against raw detections.** Before reporting a parent
  object missing, it is compared with *every* raw detection in the test
  video, not only stable tracks. This cleared the background bag and the
  keyboard, which were present but detected too rarely or under a
  different label to form a stable track.
- **CLIP naming.** The reported name is CLIP's closest vocabulary prompt
  ("a photo of glasses"), not YOLO's vote; YOLO's labels go to `aliases`.

`--clip-thr 0.855`: removed objects' best candidate scored 0.82-0.84,
present objects 0.87 and up. The window that gives a perfect result is only
0.84-0.87, chosen on the test videos themselves.

**Trade-off worth knowing**: fixed-camera mode's `pixel_diff_check` can
independently verify "is the exact pixel region where the object used to be
still unchanged" — a real pixel-level sanity check. Moving-camera mode has
no equivalent, because there's no shared pixel region to check once the
camera has moved. Its confidence signal (`appearance_confidence`) can only
say "how close was the *best available* alternative match" — a near-miss
(close to `--hist-thr`) is flagged low-confidence as possibly-still-present;
a clear absence of any similar-looking candidate is flagged high-confidence.
That's weaker evidence than a real pixel comparison. Concretely: the wallet
false positive from the fixed-camera run was correctly flagged "low
confidence" by pixel-diff, but the same case under appearance-matching came
back "high confidence" — appearance-matching has no way to notice the wallet
is actually sitting untouched, it only knows nothing else looked similar
enough to be it.

## Parameters

All thresholds are CLI arguments on `find_missing.py`, defaults chosen as below:

| Flag | Default | Why |
|---|---|---|
| `--model` | `yolov8n.pt` | Smallest YOLOv8 model; CPU-only machine, no GPU |
| `--conf` | `0.3` | Low enough to catch the real objects (0.6-0.9 confidence observed), high enough to drop the weakest background noise (tv/chair blips around 0.3-0.4) |
| `--n-frames` | `12` | Enough coverage of a ~3.7s clip without excessive CPU inference time |
| `--iou-thr` | `0.3` | Used both for within-video temporal tracking and cross-video matching; loose enough to tolerate a few pixels of camera drift between recordings, tight enough to not merge genuinely different objects |
| `--min-frac` | `0.5` | A track must appear in at least half the sampled frames to count as a stable object, not detector noise |
| `--camera` | `fixed` | `fixed` matches by IoU (position); `moving` matches by color-histogram appearance for handheld footage |
| `--hist-thr` | `0.65` | Only used in `--camera moving`. Empirically calibrated: same object across recordings scored 0.85-0.96 similarity, different objects topped out at 0.55 |
| `--imgsz` | `640` | YOLO inference size. Use `1280` for 4K footage so thin objects (a pen) survive downscaling |
| `--vocab` | built-in list | YOLO-World only: comma-separated prompts or a text file of them |
| `--agnostic-nms` | off | Cross-class NMS. Tried for the multi-label problem; it made recall *worse* (the glasses' label flips between frames, so no single-label track stays stable). Alias grouping is the fix used instead |

For a **handheld pan**, the defaults of `--n-frames 12 --min-frac 0.5` are
wrong: each object is on screen for only part of the pan (the mouse is in 3
of the parent's 12 sampled frames), so it gets thrown away as "unstable".
Use denser sampling with a lower presence threshold: `--n-frames 30
--min-frac 0.2`.

## Evaluation results

### Handheld dataset (current)

All runs use `--camera moving --n-frames 30`. "Alias-aware" =
`evaluate.py --use-aliases`.

| config | strict P / R | alias-aware P / R |
|---|---|---|
| YOLOv8n (COCO), `--min-frac 0.1` | 0.20 / 0.25 | 0.20 / 0.25 |
| YOLO-World, no alias grouping, `--conf 0.15 --imgsz 1280 --min-frac 0.1` | 0.22 / 1.00 | (n/a: no aliases yet) |
| YOLO-World + alias grouping, `--conf 0.15 --imgsz 1280 --min-frac 0.2` | 0.00 / 0.00 | 0.36 / 1.00 |
| **Same + `--match clip`** | **1.00 / 1.00** | **1.00 / 1.00** |

With `--match clip`, vid1 reports only `glasses` and vid2 reports
`glasses`, `pen` and `mouse`, with no false alarms and nothing spuriously
"added". The analysis below is for the alias-grouping run before CLIP
matching. Treat the perfect score as a result on two videos with a
threshold tuned on them, not as a general accuracy estimate.

**All four removals are found** with the final config, and the saved
evidence crops (`results_world/*_missing_*.jpg`) show the right objects.
Strict scoring is 0/4 only because every hit carries the wrong primary
label: the glasses are reported as `scissors` (alias `glasses`), the pen as
`pencil` (alias `pen`), the mouse as `cell phone` (alias `mouse`).

The 7 false positives, by inspecting the crops:
- watch labelled `bottle` (both videos): still on the desk; the appearance
  match across recordings fell short of `--hist-thr` (best 0.44)
- the Domino's bag in the background (both videos): background clutter
- the laptop keyboard (vid1)
- a teal sliver labelled `pencil` (both videos): actually the reflection on
  the **glasses** lens, i.e. the same missing glasses reported a second time
  as a separate small object, not a different object

Thresholds here were chosen against these same two test videos (a 4-config
sweep, then a sweep over the grouping rule on cached detections). With this
little data the numbers are optimistic, not a held-out estimate.

### Fixed-camera dataset (earlier, for reference)

Run on that dataset's ground truth (`video1.mp4` missing the phone,
`video2.mp4` missing the bottle):

| video | TP | FP | FN | precision | recall |
|---|---|---|---|---|---|
| video1.mp4 | 1 | 1 | 0 | 0.50 | 1.00 |
| video2.mp4 | 1 | 1 | 0 | 0.50 | 1.00 |
| **overall** | 2 | 2 | 0 | **0.50** | **1.00** |

**Recall is perfect** — both real removals were caught in every run.
**Precision is 50%**, dragged down by two explainable false positives (see
Limitations). Both are flagged "high confidence" by the pixel-diff check for
reasons unrelated to real object removal, which is itself a limitation of
that specific check rather than of the overall pipeline — see below.

## Known limitations

- **COCO's 80 classes are a hard ceiling** for standard YOLOv8. Glasses and
  pens are outside COCO, so in the handheld dataset 3 of the 4 removals are
  invisible to it. YOLO-World mode lifts this ceiling, but only for objects
  someone thought to put in the prompt list.
- **YOLO-World's labels are often wrong even when its boxes are right.**
  Glasses come out mostly as "scissors", the mouse as "cell phone". Alias
  grouping recovers the right *object*, but the primary label in
  `results.csv` can't be trusted on its own; read the `aliases` column and
  the evidence crop.
- **Alias grouping can chain.** Union-find is transitive, so A~B and B~C
  merges A with C even if A and C never overlap. With a looser rule
  (containment) this hid the mouse behind the notepad; the IoU rule avoids
  that here, but it isn't guaranteed in general.
- **CLIP matching has a narrow margin** (0.84 vs 0.87 on this data), and
  its raw-detection second chance can be fooled by a near-identical object
  that stayed in the scene: the removed one would be judged present.
- **Objects at the frame edge** during a pan are cut off and can get odd
  labels or unstable tracks.
- **Thresholds were tuned on the test set itself** (two videos). Expect
  worse precision on new footage.
- **Visually similar COCO classes create false positives.** In this scene,
  the wallet, the power bank, and the watch were all intermittently
  misclassified as `cell phone` by YOLOv8n — COCO's "cell phone" class is
  visually broad (any small dark rectangle). This doesn't break the pipeline
  (class + position matching still finds the right box), but it does mean
  `results.csv` can show more `cell phone` rows than there are real phones in
  frame; `evaluate.py`'s class-granularity scoring absorbs this without
  inflating false-positive counts, but a per-object (not per-class) ground
  truth would expose it directly.
- **Transparent/reflective objects defeat pixel-diff.** The water bottle's
  appearance is dominated by reflections and whatever's visible through the
  glass, which changes between two separate recordings regardless of whether
  the bottle itself moved. Pixel-diff cross-check reports "high confidence"
  for the bottle even when it's genuinely still there — a known, documented
  weakness of appearance-based diffing on specular/transparent surfaces, not
  something the alignment-search fix (below) can solve.
- **Camera framing is never pixel-identical between two separate
  recordings**, even with a tripod. `pixel_diff_check` searches a small
  (dx, dy) shift window to compensate, which fixed one false positive
  (a wallet, now correctly scored "low confidence") but didn't fully resolve
  the bottle case above.
- **Background clutter that happens to match a COCO class** (e.g. a
  mirror/bag in the corner classified as `tv`) can get flagged missing if
  its detection is unstable between recordings, purely because it's
  irrelevant content rather than a true scene object. No class allowlist was
  added by design choice — the tool stays general-purpose rather than tuned
  to the specific objects in this dataset — so this shows up as an
  explainable false positive rather than being silently filtered.
- **Identical objects of the same class, swapped or duplicated, are not
  distinguishable.** If two real phones were in the scene and one was
  removed, class+IoU matching alone can't guarantee it picks the *right* one
  if they're close together.
- **Occlusion and multiple simultaneously-missing objects of the same class
  per video are untested** — the current ground truth only has one missing
  object per video, each a different class.
- **OCR product-naming is unreliable on real product photography.** Tested
  directly on this scene's own crops: "LLOYDS" (printed on the bottle) and
  "120W FAST CHARGING" (embossed, grey-on-white, on the power bank) are both
  clearly legible to a human eye, but Tesseract could not reliably read
  either one even after upscaling and CLAHE contrast enhancement — it's
  tuned for flat, high-contrast document scans, not small or low-contrast
  text at an angle on a curved/textured surface. The `ocr_text` column will
  often be blank; a word-pattern filter blocks most outright garbage, but
  occasional nonsense can still slip through if background clutter happens
  to contain letter-like shapes.
- **Moving-camera mode has no pixel-level sanity check.** Fixed-camera mode's
  `pixel_diff_check` can directly verify "is this exact spot still
  unchanged." Appearance-matching can only report "how close was the best
  available alternative," which is strictly weaker evidence — see the
  moving-camera approach section above for a concrete example of this
  producing a *worse* confidence call than the fixed-camera version did on
  the same underlying false positive.
- **Appearance matching depends on color/texture being genuinely
  distinctive.** Two different objects that happen to share a dominant
  color (e.g. two black rectangular objects) will be harder to tell apart by
  histogram than they would be by position in a fixed-camera setup. This is
  also why the user was advised to prefer visually/texturally distinct
  objects when reshooting for `--camera moving`.

## How to run

```powershell
# 1. Check video metadata
python video_info.py

# 2. Sample frames for visual inspection (optional, writes to debug/)
python sample_frames.py

# 3. Sanity-check detection on a single frame (optional)
python detect_one.py debug/parent/frame_00_src0000.jpg

# 4. Run the full pipeline (writes results/results.csv + annotated images)
python find_missing.py                     # fixed camera (default)
python find_missing.py --camera moving     # handheld/moving camera

# Handheld pan + open vocabulary + CLIP matching (the current dataset's best config)
python find_missing.py --camera moving --match clip --model yolov8s-worldv2.pt --imgsz 1280 `
    --n-frames 30 --conf 0.15 --min-frac 0.2 --out-dir results_clip

# 5. Score results.csv against ground_truth.csv
python evaluate.py
python evaluate.py --results results_world/results.csv --use-aliases
```

YOLO-World downloads `yolov8s-worldv2.pt` (~25 MB) on first use, plus the
CLIP text encoder, and needs the `clip` package. `requirements.txt` installs
it from a GitHub zip archive, because pip's `git+https://` form fails on
machines without git.

Install Python dependencies with `pip install -r requirements.txt` inside the
project's virtual environment. The product-naming feature also needs the
Tesseract OCR engine itself (not just the `pytesseract` Python wrapper) —
on Windows: `winget install UB-Mannheim.TesseractOCR`. If it's not installed,
`find_missing.py` prints a note and simply leaves `ocr_text` blank rather
than failing.

Output folders: `results_clip/` is the best handheld run (`--match clip`);
`results_world/` is the YOLO-World run before CLIP matching;
`results_handheld/` is the COCO-only handheld baseline; `results/` still
holds output from the earlier fixed-camera dataset.
