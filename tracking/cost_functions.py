from typing import Optional, Tuple


BoundingBox = Tuple[float, float, float, float]  # x1, y1, x2, y2


# =====================================================
# Single-Camera Tracking Framework milestone, Phase 5 -- the two plain
# geometric similarity measures the baseline tracker matches with.
# Deliberately pure functions of two bounding boxes: no state, no
# learned model, no motion prediction -- "a clean engineering baseline",
# not DeepSORT/ByteTrack/StrongSORT/OCSORT (all explicitly out of
# scope). A future tracker can replace SimpleSingleCameraTracker
# entirely with a smarter matching strategy without this module, or
# anything importing it, needing to change.
# =====================================================


def iou(box_a: BoundingBox, box_b: BoundingBox) -> float:

    # Standard intersection-over-union of two axis-aligned boxes in the
    # same (x1, y1, x2, y2) image-pixel-space convention
    # human_detection.yolo_backend.BoundingBoxDetection already
    # establishes. 0.0 for non-overlapping or degenerate (zero-area)
    # boxes -- never negative, never > 1.0.

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    intersection = inter_width * inter_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - intersection

    if union <= 0.0:
        return 0.0

    return intersection / union


def centroid(box: BoundingBox) -> Tuple[float, float]:

    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def centroid_distance(box_a: BoundingBox, box_b: BoundingBox) -> float:

    ax, ay = centroid(box_a)
    bx, by = centroid(box_b)

    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


def match_score(
    box_a: Optional[BoundingBox],
    box_b: Optional[BoundingBox],
    *,
    iou_threshold: float,
    max_centroid_distance: float,
) -> Optional[float]:

    # Returns a single "how good a match" score, higher is better, or
    # None if the two boxes do not qualify as the same physical track
    # by either criterion -- Phase 5's own "IoU matching, or centroid
    # distance, or both" instruction, implemented as: prefer an
    # IoU-qualifying match (scored on IoU itself, always preferred over
    # a centroid-only match, matching this codebase's own "geometric
    # overlap is stronger evidence than raw proximity" convention);
    # fall back to a centroid-distance-qualifying match only when IoU
    # does not clear its own threshold (handles small/fast-moving boxes
    # with little or no frame-to-frame overlap). A detection or a
    # coasting track with no bounding box at all (Optional, honestly
    # absent) can never be geometrically matched -- always None.

    if box_a is None or box_b is None:
        return None

    overlap = iou(box_a, box_b)

    if overlap >= iou_threshold:
        # Scored above any possible centroid-only score (which is
        # normalized into (0, 1) below) so an IoU-qualifying pair is
        # always preferred during greedy assignment.
        return 1.0 + overlap

    distance = centroid_distance(box_a, box_b)

    if distance <= max_centroid_distance:
        return 1.0 - (distance / max_centroid_distance) if max_centroid_distance > 0 else 1.0

    return None
