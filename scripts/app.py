"""ParkVision AI.

One Streamlit file. Upload an overhead shot of a lined parking lot, or open a
sample frame. Bays that match the sample camera use that camera's map. Any
other overhead shot is read from a parking-space model, or from the painted
stall lines when the shot is straight down. On a diagonal or herringbone lot,
each stall is boxed along its row. An open bay between two cars is marked
empty, and so is an open stall whose painted sides are still visible.
"""

from __future__ import annotations

import base64
import io
import wave
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from PIL import Image

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "samples"
MODELS = ROOT / "models"


# RGB colours required by the brief: green empty, red occupied.
EMPTY_RGB = (34, 197, 94)
OCCUPIED_RGB = (239, 68, 68)

_EMPTY_HINTS = ("empty", "free", "vacant", "available", "0")
_OCCUPIED_HINTS = ("occupied", "busy", "taken", "full", "car", "vehicle", "1")


@dataclass(frozen=True)
class Slot:
    status: str
    confidence: float
    xyxy: tuple[int, int, int, int]
    # Four corners [(x1,y1), (x2,y2), (x3,y3), (x4,y4)] when the bay is slanted
    # or extracted via OBB. If None, drawn from xyxy.
    quad: tuple[tuple[int, int], tuple[int, int], tuple[int, int], tuple[int, int]] | None = None


def slot_status(class_name: str) -> str:
    """Map a detector class name to ``empty`` or ``occupied``."""
    name = class_name.lower().replace("_", " ").replace("-", " ").strip()
    if any(hint in name for hint in _EMPTY_HINTS):
        return "empty"
    if any(hint in name for hint in _OCCUPIED_HINTS):
        return "occupied"
    return "occupied"


def warp_slot_crop(image: np.ndarray, quad: tuple[tuple[int, int], ...], width: int = 64, height: int = 128) -> np.ndarray | None:
    """Warp an arbitrary quadrilateral parking stall to an upright rectified rectangle.
    
    Prevents diagonal stalls from bleeding into adjacent bays or surrounding sidewalks.
    """
    src_pts = np.array(quad, dtype=np.float32)
    if len(src_pts) != 4:
        return None
    
    # Order points: top-left, top-right, bottom-right, bottom-left
    rect = np.zeros((4, 2), dtype=np.float32)
    s = src_pts.sum(axis=1)
    rect[0] = src_pts[np.argmin(s)]
    rect[2] = src_pts[np.argmax(s)]
    diff = np.diff(src_pts, axis=1)
    rect[1] = src_pts[np.argmin(diff)]
    rect[3] = src_pts[np.argmax(diff)]

    dst_pts = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    matrix = cv2.getPerspectiveTransform(rect, dst_pts)
    warped = cv2.warpPerspective(image, matrix, (width, height), flags=cv2.INTER_LINEAR)
    return warped


def annotate(image_rgb: np.ndarray, slots: list[Slot]) -> np.ndarray:
    """Return a copy of ``image_rgb`` with colour-coded slot boxes."""
    canvas = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    height = canvas.shape[0]
    thickness = max(2, height // 320)
    font_scale = max(0.4, height / 1400)

    for slot in slots:
        rgb = EMPTY_RGB if slot.status == "empty" else OCCUPIED_RGB
        color = (rgb[2], rgb[1], rgb[0])
        x1, y1, x2, y2 = slot.xyxy

        if slot.quad is not None:
            corners = np.array(slot.quad, dtype=np.int32)
            cv2.polylines(canvas, [corners], isClosed=True, color=color, thickness=thickness)
            x1 = int(min(point[0] for point in slot.quad))
            y1 = int(min(point[1] for point in slot.quad))
            x2 = int(max(point[0] for point in slot.quad))
        else:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), color, thickness)

        label = "occ" if slot.status == "occupied" else "empty"
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )
        if (x2 - x1) >= tw + 4:
            top = max(0, y1 - th - baseline - 4)
            cv2.rectangle(
                canvas, (x1, top), (x1 + tw + 6, top + th + baseline + 4), color, -1
            )
            cv2.putText(
                canvas,
                label,
                (x1 + 3, top + th + 1),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    return cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)


def count_slots(slots: list[Slot]) -> tuple[int, int]:
    occupied = sum(1 for slot in slots if slot.status == "occupied")
    empty = sum(1 for slot in slots if slot.status == "empty")
    return occupied, empty


_FALLBACK_OPEN_FRACTION = 46 / 144
_SAMPLE_RATE = 22050


@dataclass(frozen=True)
class ParkingInsights:
    total: int
    occupied: int
    available: int
    occupancy_pct: float
    level: str
    recommendation: str


def summarize(occupied: int, empty: int) -> ParkingInsights:
    """Count slots and classify congestion."""
    if occupied < 0 or empty < 0:
        raise ValueError("Slot counts cannot be negative.")

    total = occupied + empty
    if total == 0:
        return ParkingInsights(
            total=0,
            occupied=0,
            available=0,
            occupancy_pct=0.0,
            level="No slots detected",
            recommendation=(
                "No parking slots were detected in this frame. "
                "Try a clearer camera view or verify alignment."
            ),
        )

    occupancy = occupied / total * 100.0
    share_open = empty / total
    average_open = open_bay_average().fraction
    
    if share_open >= average_open * 2:
        level = "Low"
        recommendation = "Slots available — proceed."
    elif share_open <= average_open * 0.5:
        level = "High"
        recommendation = "Parking full — try another area."
    else:
        level = "Moderate"
        recommendation = "Some slots are open — proceed, and take the nearest empty bay."

    return ParkingInsights(
        total=total,
        occupied=occupied,
        available=empty,
        occupancy_pct=round(occupancy, 1),
        level=level,
        recommendation=recommendation,
    )


ROW_BANDS: tuple[tuple[float, float], ...] = (
    (0.000, 0.070),
    (0.075, 0.150),
    (0.160, 0.245),
    (0.255, 0.340),
    (0.360, 0.430),
    (0.455, 0.540),
    (0.555, 0.650),
)
LOWER_BAND = (0.692, 0.782)
BOTTOM_BAND = (0.860, 0.970)
X_START = 0.00
X_STOP = 0.80
LOWER_X_STOP = 0.78
X_STEP = 0.05

OCCUPIED_BELOW = 0.55
LOWER_OCCUPIED_BELOW = 0.50
BOTTOM_MATCH = 0.28
BOTTOM_STD_MATCH = 0.55
BOTTOM_STD = 38.0
TEXTURE_RATIO = 4.5
TEXTURE_MARGIN = 0.23

REFERENCE_NAME = "lot-11.jpg"
_ROW_COUNT = len(ROW_BANDS) + 2


@dataclass(frozen=True)
class LotReading:
    slots: list[Slot]
    same_camera: bool
    aligned_bgr: np.ndarray
    source: str = "camera"


def _x_edges(width: int, x_stop: float) -> list[int]:
    edges: list[float] = []
    x = X_START
    while x < x_stop - 1e-6:
        edges.append(x)
        x = round(x + X_STEP, 5)
    edges.append(x_stop)
    return [int(round(value * width)) for value in edges]


def slot_boxes(width: int, height: int) -> list[tuple[int, int, int, int]]:
    """Pixel boxes for every lined bay, left to right, top to bottom."""
    boxes: list[tuple[int, int, int, int]] = []
    spans = [((top, bottom), X_STOP) for top, bottom in ROW_BANDS]
    spans.append((LOWER_BAND, LOWER_X_STOP))
    spans.append((BOTTOM_BAND, LOWER_X_STOP))
    for (y1, y2), x_stop in spans:
        top = int(round(y1 * height))
        bottom = int(round(y2 * height))
        edges = _x_edges(width, x_stop)
        for left, right in zip(edges, edges[1:]):
            if right - left >= 2 and bottom - top >= 2:
                boxes.append((left, top, right, bottom))
    return boxes


def _crop(image: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = box
    dx = max(1, (x2 - x1) // 10)
    dy = max(1, (y2 - y1) // 8)
    return image[y1 + dy : y2 - dy, x1 + dx : x2 - dx]


def _correlation(image: np.ndarray, reference: np.ndarray, box: tuple[int, int, int, int]) -> float:
    current = _crop(image, box)
    baseline = _crop(reference, box)
    if current.size < 16 or current.shape != baseline.shape:
        return 1.0
    current = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    baseline = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
    score = cv2.matchTemplate(current, baseline, cv2.TM_CCOEFF_NORMED)
    return float(score[0, 0])


def _laplacian(image: np.ndarray, box: tuple[int, int, int, int]) -> float:
    gray = cv2.cvtColor(_crop(image, box), cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_32F).var())


def _texture_ratio(image: np.ndarray, reference: np.ndarray, box: tuple[int, int, int, int]) -> float:
    return _laplacian(image, box) / (_laplacian(reference, box) + 1.0)


def _parked_on_plate(image: np.ndarray, box: tuple[int, int, int, int]) -> bool:
    height, width = image.shape[:2]
    x1, y1, x2, y2 = box
    if x1 <= 1 or y1 <= 1 or x2 >= width - 1 or y2 >= height - 1:
        return False
    if _laplacian(image, box) < 140:
        return False
    gray = cv2.cvtColor(image[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    dark = (gray.astype(np.int16) < np.median(gray) - 18).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    if count <= 1:
        return False
    index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    area = stats[index, cv2.CC_STAT_AREA] / float(dark.size)
    box_w = max(1, x2 - x1)
    box_h = max(1, y2 - y1)
    return (
        area > 0.18
        and stats[index, cv2.CC_STAT_WIDTH] / box_w > 0.45
        and stats[index, cv2.CC_STAT_HEIGHT] / box_h > 0.45
    )


def _clear_parked_vehicles(plate: np.ndarray) -> np.ndarray:
    boxes = slot_boxes(plate.shape[1], plate.shape[0])
    if not boxes:
        return plate
    per_row = len(boxes) // _ROW_COUNT
    energy = [_laplacian(plate, box) for box in boxes]
    cleaned = plate.copy()
    for index, box in enumerate(boxes):
        if not _parked_on_plate(plate, box):
            continue
        row, col = divmod(index, per_row)
        calm = float(np.median(energy[row * per_row : (row + 1) * per_row]))
        donor = None
        for other_col in range(per_row):
            other = row * per_row + other_col
            if other == index or energy[other] > calm * 1.4:
                continue
            distance = abs(other_col - col)
            if donor is None or distance < donor[0]:
                donor = (distance, other)
        if donor is None:
            continue
        x1, y1, x2, y2 = box
        sx1, sy1, sx2, sy2 = boxes[donor[1]]
        patch = plate[sy1:sy2, sx1:sx2]
        if patch.shape[0] != y2 - y1 or patch.shape[1] != x2 - x1:
            patch = cv2.resize(patch, (x2 - x1, y2 - y1), interpolation=cv2.INTER_AREA)
        cleaned[y1:y2, x1:x2] = patch
    return cleaned


def _grass_correlation(image: np.ndarray, reference: np.ndarray) -> float:
    height, width = reference.shape[:2]
    box = (
        int(0.20 * width),
        int(0.655 * height),
        int(0.60 * width),
        int(0.705 * height),
    )
    return _correlation(image, reference, box)


def _prepare(gray: np.ndarray) -> np.ndarray:
    return cv2.createCLAHE(3.0, (8, 8)).apply(gray)


def _align_to_reference(image: np.ndarray, reference: np.ndarray) -> np.ndarray | None:
    reference_max = max(reference.shape[:2])
    image_max = max(image.shape[:2])
    match_image = image
    shrink = 1.0
    if image_max > reference_max * 2:
        shrink = (reference_max * 2) / float(image_max)
        match_image = cv2.resize(
            image,
            (max(1, int(image.shape[1] * shrink)), max(1, int(image.shape[0] * shrink))),
            interpolation=cv2.INTER_AREA,
        )

    gray_image = _prepare(cv2.cvtColor(match_image, cv2.COLOR_BGR2GRAY))
    gray_reference = _prepare(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY))
    orb = cv2.ORB_create(8000, fastThreshold=8)
    keys_image, desc_image = orb.detectAndCompute(gray_image, None)
    keys_reference, desc_reference = orb.detectAndCompute(gray_reference, None)
    if desc_image is None or desc_reference is None or len(keys_image) < 20 or len(keys_reference) < 20:
        return None

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    pairs = matcher.knnMatch(desc_image, desc_reference, k=2)
    good = [m for m, n in pairs if len((m, n)) == 2 and m.distance < 0.75 * n.distance]
    if len(good) < 12:
        return None

    source = np.float32([keys_image[match.queryIdx].pt for match in good])
    destination = np.float32([keys_reference[match.trainIdx].pt for match in good])
    homography, mask = cv2.findHomography(source, destination, cv2.RANSAC, 5.0)
    if homography is not None and shrink != 1.0:
        homography = homography @ np.array(
            [[shrink, 0.0, 0.0], [0.0, shrink, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
    inliers = 0 if mask is None else int(mask.sum())
    if homography is None or inliers < 12:
        return None
    if float(np.linalg.det(homography[:2, :2])) <= 0:
        return None
    scales = np.linalg.svd(homography[:2, :2], compute_uv=False)
    if float(scales.min()) < 0.08 or float(scales.max()) > 12.0:
        return None

    height, width = reference.shape[:2]
    warped = cv2.warpPerspective(
        image,
        homography,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )
    covered = np.count_nonzero(cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY) > 8) / float(height * width)
    grass = _grass_correlation(warped, reference)
    strong = inliers >= 28 and covered >= 0.18
    weak = covered >= 0.45 and grass >= 0.18
    if not (strong or weak):
        return None
    return warped


def _registered(image: np.ndarray, reference: np.ndarray) -> bool:
    if image.shape[:2] != reference.shape[:2]:
        return False
    return _grass_correlation(image, reference) >= 0.20


def read_lot(image_bgr: np.ndarray, reference_bgr: np.ndarray, occupied_below: float = OCCUPIED_BELOW) -> LotReading:
    matched = _registered(image_bgr, reference_bgr)
    if not matched:
        aligned = _align_to_reference(image_bgr, reference_bgr)
        if aligned is None:
            return LotReading(slots=[], same_camera=False, aligned_bgr=image_bgr)
        image_bgr = aligned
        matched = True
    grass = _grass_correlation(image_bgr, reference_bgr)
    boxes = slot_boxes(image_bgr.shape[1], image_bgr.shape[0])
    height = image_bgr.shape[0]
    lower_cut = int(0.68 * height)
    bottom_cut = int(0.82 * height)
    slots: list[Slot] = []
    for box in boxes:
        x1, y1, x2, y2 = box
        bay = image_bgr[y1:y2, x1:x2]
        if bay.size == 0:
            continue
        gray = cv2.cvtColor(bay, cv2.COLOR_BGR2GRAY)
        covered = np.count_nonzero(gray > 8) / float(bay.shape[0] * bay.shape[1])
        if covered < 0.65:
            continue
        match = _correlation(image_bgr, reference_bgr, box)
        if y1 >= bottom_cut:
            spread = float(gray.std())
            occupied = match < BOTTOM_MATCH or (match < BOTTOM_STD_MATCH and spread > BOTTOM_STD)
        else:
            limit = min(occupied_below, LOWER_OCCUPIED_BELOW) if y1 >= lower_cut else occupied_below
            soft_below = min(0.85, limit + TEXTURE_MARGIN)
            busy = _texture_ratio(image_bgr, reference_bgr, box)
            occupied = match < limit or (match < soft_below and busy > TEXTURE_RATIO)
        confidence = (1.0 - match) if occupied else match
        confidence = float(np.clip(confidence, 0.0, 0.99))
        slots.append(
            Slot(
                status="occupied" if occupied else "empty",
                confidence=confidence,
                xyxy=box,
            )
        )
    return LotReading(slots=slots, same_camera=matched or grass >= 0.20, aligned_bgr=image_bgr)


def load_reference(samples_dir: Path) -> np.ndarray:
    path = samples_dir / REFERENCE_NAME
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Empty-lot plate not found at {path}")
    return _clear_parked_vehicles(image)


_LINED_MIN_BAYS = 6


def _segments(gray: np.ndarray) -> list[tuple[float, float, float, float, float, float]]:
    raw = cv2.createLineSegmentDetector(0).detect(gray)[0]
    if raw is None:
        return []
    found = []
    for x1, y1, x2, y2 in raw.reshape(-1, 4):
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length < 16:
            continue
        angle = abs(np.degrees(np.arctan2(dy, dx)))
        if angle > 90:
            angle = 180 - angle
        found.append((float(x1), float(y1), float(x2), float(y2), angle, length))
    return found


def _divider_weight(gray: np.ndarray, vertical: bool) -> float:
    total = 0.0
    for *_, angle, length in _segments(gray):
        steep = angle >= 55
        if steep == vertical and 16 <= length <= 120:
            total += length
    return total


def _dividers(gray: np.ndarray) -> list[dict]:
    divs = []
    for x1, y1, x2, y2, angle, length in _segments(gray):
        if length > 110 or angle < 55:
            continue
        divs.append(
            {
                "x": (x1 + x2) / 2,
                "y": (y1 + y2) / 2,
                "y1": min(y1, y2),
                "y2": max(y1, y2),
            }
        )
    return divs


def _row_centers(divs: list[dict], height: int) -> list[float]:
    if len(divs) < 8:
        return []
    ys = np.array([d["y"] for d in divs])
    bins = np.arange(0, height + 8, 8)
    hist, edges = np.histogram(ys, bins=bins)
    smooth = np.convolve(hist.astype(float), np.ones(3) / 3, mode="same")
    peaks = []
    for index in range(1, len(smooth) - 1):
        if smooth[index] >= smooth[index - 1] and smooth[index] >= smooth[index + 1] and hist[index] >= 5:
            peaks.append(float((edges[index] + edges[index + 1]) / 2))
    merged: list[float] = []
    for peak in peaks:
        if merged and peak - merged[-1] < 70:
            merged[-1] = (merged[-1] + peak) / 2
        else:
            merged.append(peak)
    return merged


def _boxes_for_row(divs: list[dict], center: float, height: int, green: np.ndarray) -> list[tuple[int, int, int, int]]:
    items = [d for d in divs if abs(d["y"] - center) < 48]
    if len(items) < 6:
        return []
    items.sort(key=lambda d: d["x"])
    cols: list[dict] = []
    for item in items:
        if cols and item["x"] - cols[-1]["x"] < 8:
            col = cols[-1]
            col["x"] = (col["x"] * col["n"] + item["x"]) / (col["n"] + 1)
            col["n"] += 1
        else:
            cols.append({"x": item["x"], "n": 1})
    gaps = np.diff([col["x"] for col in cols])
    plausible = gaps[(gaps > 12) & (gaps < 70)]
    if len(plausible) < 3:
        return []
    pitch = float(np.median(plausible))
    half = int(pitch * 1.7)
    y1, y2 = int(center - half), int(center + half)
    boxes = []
    for index in range(len(cols) - 1):
        gap = cols[index + 1]["x"] - cols[index]["x"]
        count = int(round(gap / pitch))
        if not 1 <= count <= 2:
            continue
        step = gap / count
        if step < pitch * 0.6 or step > pitch * 1.45:
            continue
        for slot_index in range(count):
            left = cols[index]["x"] + step * slot_index
            right = left + step
            box = (
                int(round(left)) + 2,
                max(0, y1 + 2),
                int(round(right)) - 2,
                min(height - 1, y2 - 2),
            )
            if box[2] - box[0] < 8 or box[3] - box[1] < 14:
                continue
            patch = green[box[1] : box[3], box[0] : box[2]]
            if patch.size and float(patch.mean()) > 0.22:
                continue
            boxes.append(box)
    return boxes


def _vehicle_fraction(image: np.ndarray, box: tuple[int, int, int, int], asphalt: float) -> float:
    x1, y1, x2, y2 = box
    dx = max(1, (x2 - x1) // 10)
    dy = max(1, (y2 - y1) // 8)
    crop = image[y1 + dy : y2 - dy, x1 + dx : x2 - dx]
    if crop.size < 16:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    span = max(3, min(9, (x2 - x1) // 3))
    if span % 2 == 0:
        span += 1
    paint = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (span, span)))
    differ = np.abs(gray.astype(np.int16) - asphalt) > 30
    colored = hsv[:, :, 1] > 38
    vehicle = (differ | colored) & (paint <= 20)
    vehicle = cv2.morphologyEx(vehicle.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return float(vehicle.mean())


def _asphalt_level(image: np.ndarray, boxes: list[tuple[int, int, int, int]]) -> float:
    spreads = []
    for box in boxes:
        gray = cv2.cvtColor(image[box[1] : box[3], box[0] : box[2]], cv2.COLOR_BGR2GRAY)
        spreads.append(float(gray.std()))
    if not spreads:
        return 140.0
    spreads = np.array(spreads)
    calm = spreads <= np.percentile(spreads, 30)
    levels = []
    for box, quiet in zip(boxes, calm):
        if not quiet:
            continue
        gray = cv2.cvtColor(image[box[1] : box[3], box[0] : box[2]], cv2.COLOR_BGR2GRAY)
        levels.append(float(np.median(gray)))
    return float(np.median(levels)) if levels else 140.0


def _keep_runs(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    if not boxes:
        return []
    widths = [box[2] - box[0] for box in boxes]
    gap_limit = float(np.median(widths)) * 2.4
    rows: list[list[tuple[int, int, int, int]]] = []
    for box in sorted(boxes, key=lambda item: (item[1] + item[3]) / 2):
        center = (box[1] + box[3]) / 2
        if rows and abs(center - np.mean([(item[1] + item[3]) / 2 for item in rows[-1]])) < 24:
            rows[-1].append(box)
        else:
            rows.append([box])
    kept = []
    for row in rows:
        row.sort(key=lambda item: item[0])
        run = [row[0]]
        runs = []
        for box in row[1:]:
            if box[0] - run[-1][2] > gap_limit:
                runs.append(run)
                run = [box]
            else:
                run.append(box)
        runs.append(run)
        for run in runs:
            if len(run) >= 3:
                kept.extend(run)
    return kept


def _slots_upright(image: np.ndarray) -> list[Slot]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    green = (hsv[:, :, 0] > 35) & (hsv[:, :, 0] < 95) & (hsv[:, :, 1] > 40)
    divs = _dividers(gray)
    boxes: list[tuple[int, int, int, int]] = []
    for center in _row_centers(divs, image.shape[0]):
        boxes.extend(_boxes_for_row(divs, center, image.shape[0], green))
    boxes = _keep_runs(boxes)
    if len(boxes) < _LINED_MIN_BAYS:
        return []
    asphalt = _asphalt_level(image, boxes)
    slots = []
    for box in boxes:
        fraction = _vehicle_fraction(image, box, asphalt)
        occupied = fraction > 0.18
        slots.append(
            Slot(
                status="occupied" if occupied else "empty",
                confidence=float(np.clip(fraction if occupied else 1.0 - fraction, 0.0, 0.99)),
                xyxy=box,
            )
        )
    return slots


def _unmap_clockwise(box: tuple[int, int, int, int], height: int) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    corners = ((x1, y1), (x2, y1), (x2, y2), (x1, y2))
    original = [(y, height - 1 - x) for x, y in corners]
    xs = [point[0] for point in original]
    ys = [point[1] for point in original]
    return int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))


def _rotate_bound(image: np.ndarray, angle: float) -> tuple[np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    center = (width / 2, height / 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    cosine, sine = abs(matrix[0, 0]), abs(matrix[0, 1])
    bound_w = int(height * sine + width * cosine)
    bound_h = int(height * cosine + width * sine)
    matrix[0, 2] += bound_w / 2 - center[0]
    matrix[1, 2] += bound_h / 2 - center[1]
    warped = cv2.warpAffine(image, matrix, (bound_w, bound_h), flags=cv2.INTER_LINEAR)
    return warped, matrix


def _line_angle(dx: float, dy: float) -> float:
    return float(np.degrees(np.arctan2(dy, dx)) % 180.0)


def _angle_delta(a: float, b: float) -> float:
    delta = abs(a - b) % 180.0
    return min(delta, 180.0 - delta)


def _ridge_strength(gray: np.ndarray, x1: float, y1: float, x2: float, y2: float) -> float:
    height, width = gray.shape
    ang = np.arctan2(y2 - y1, x2 - x1)
    nx, ny = -np.sin(ang), np.cos(ang)
    scores = []
    for t in (0.25, 0.5, 0.75):
        mx = x1 + (x2 - x1) * t
        my = y1 + (y2 - y1) * t
        vals = []
        for step in range(-6, 7):
            x = int(round(mx + nx * step))
            y = int(round(my + ny * step))
            if not (0 <= x < width and 0 <= y < height):
                vals = []
                break
            vals.append(int(gray[y, x]))
        if len(vals) < 13:
            continue
        center = float(np.mean(vals[5:8]))
        sides = 0.5 * (float(np.mean(vals[0:3])) + float(np.mean(vals[10:13])))
        scores.append(center - sides)
    if len(scores) < 2:
        return -99.0
    return float(np.median(scores))


def _paint_segments(gray: np.ndarray):
    kept = []
    for x1, y1, x2, y2, _folded, length in _segments(gray):
        if length < 20 or _ridge_strength(gray, x1, y1, x2, y2) < 5:
            continue
        kept.append((x1, y1, x2, y2, _line_angle(x2 - x1, y2 - y1), length))
    return kept


def _slant_peaks(gray: np.ndarray) -> list[float]:
    hist = np.zeros(180, dtype=np.float64)
    for *_, ang, length in _paint_segments(gray):
        hist[int(ang) % 180] += length
    smooth = np.convolve(hist, np.ones(9) / 9, mode="same")
    for degree in range(180):
        if _angle_delta(degree, 0) < 15 or _angle_delta(degree, 90) < 15:
            smooth[degree] = 0
    found = []
    for degree in range(180):
        if smooth[degree] < 180:
            continue
        if smooth[degree] >= smooth[(degree - 1) % 180] and smooth[degree] > smooth[(degree + 1) % 180]:
            found.append((float(smooth[degree]), float(degree)))
    found.sort(reverse=True)
    kept: list[float] = []
    for _weight, degree in found:
        if all(_angle_delta(degree, other) > 18 for other in kept):
            kept.append(degree)
    return [degree for degree in kept[:4] if _angle_delta(degree, 0) >= 20 and _angle_delta(degree, 90) >= 20]


def _quad_from_upright(inverse, cx, cy, half_w, half_l, shape):
    height, width = shape[:2]

    def original(x, y):
        return (
            inverse[0, 0] * x + inverse[0, 1] * y + inverse[0, 2],
            inverse[1, 0] * x + inverse[1, 1] * y + inverse[1, 2],
        )

    center = original(cx, cy)
    if not (6 < center[0] < width - 6 and 6 < center[1] < height - 6):
        return None
    corners = [original(cx + dx, cy + dy) for dx in (-half_w, half_w) for dy in (-half_l, half_l)]
    order = np.argsort([np.arctan2(y - center[1], x - center[0]) for x, y in corners])
    ordered = [corners[index] for index in order]
    return tuple((int(round(np.clip(x, 0, width - 1))), int(round(np.clip(y, 0, height - 1)))) for x, y in ordered)


def _foliage_fraction(hsv) -> float:
    pixels = np.asarray(hsv).reshape(-1, 3)
    if pixels.size == 0:
        return 1.0
    hue, sat = pixels[:, 0], pixels[:, 1]
    return float(((hue >= 18) & (hue <= 48) & (sat > 70)).mean())


def _core_sample(gray, hsv, probe, cx, cy, width, length):
    half_w, half_l = width * 0.32, length * 0.32
    x1, x2 = int(cx - half_w), int(cx + half_w)
    y1, y2 = int(cy - half_l), int(cy + half_l)
    height, width_px = gray.shape
    if x1 < 2 or y1 < 2 or x2 >= width_px - 2 or y2 >= height - 2:
        return None
    return (
        float(probe[y1:y2, x1:x2].mean()),
        float(gray[y1:y2, x1:x2].std()),
        _foliage_fraction(hsv[y1:y2, x1:x2]),
        float(np.median(gray[y1:y2, x1:x2])),
    )


def _upright_vehicles(image, peak):
    turned, matrix = _rotate_bound(image, 90 - peak)
    inverse = np.linalg.inv(np.vstack([matrix, [0, 0, 1]]))[:2]
    gray = cv2.cvtColor(turned, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(turned, cv2.COLOR_BGR2HSV)
    scale = min(image.shape[:2]) / 960.0
    span = 81 if scale >= 0.72 else (max(31, int(min(image.shape[:2]) * 0.1)) | 1)
    if span % 2 == 0:
        span += 1
    area_lo = int(np.clip(450 * scale, 280, 520))
    short_lo = float(np.clip(20 * scale, 14, 22))
    long_lo = float(np.clip(48 * scale, 36, 52))
    delta = gray.astype(np.int16) - cv2.medianBlur(gray, span).astype(np.int16)
    probe = ((delta < -16) | (delta > 12)).astype(np.uint8)
    blobs = []
    for darker, limit in ((True, -24), (False, 18)):
        mask = (delta < limit).astype(np.uint8) if darker else (delta > limit).astype(np.uint8)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        count, labels, _stats, _ = cv2.connectedComponentsWithStats(closed)
        for index in range(1, count):
            ys, xs = np.where(labels == index)
            if not area_lo < len(xs) < 9500 or _foliage_fraction(hsv[ys, xs]) > 0.22:
                continue
            sample = np.stack([xs, ys], 1)
            if len(sample) > 1100:
                sample = sample[np.linspace(0, len(sample) - 1, 1100).astype(int)]
            rect = cv2.minAreaRect(sample.astype(np.float32))
            box = cv2.boxPoints(rect)
            edges = []
            for start, end in zip(box, np.roll(box, -1, 0)):
                dx, dy = float(end[0] - start[0]), float(end[1] - start[1])
                edges.append((float(np.hypot(dx, dy)), float(np.degrees(np.arctan2(dy, dx)) % 180)))
            edges.sort(key=lambda edge: -edge[0])
            long_edge, axis = edges[0]
            short_edge = edges[2][0]
            if short_edge < short_lo or long_edge < long_lo or long_edge > 180 or not 1.25 <= long_edge / max(short_edge, 1) <= 3.8:
                continue
            if _angle_delta(axis, 90) > 18:
                continue
            blobs.append([float(rect[0][0]), float(rect[0][1]), short_edge, long_edge, float(len(xs))])
    blobs.sort(key=lambda blob: -blob[4])
    kept = []
    for blob in blobs:
        if any(
            abs(blob[0] - other[0]) < 0.65 * max(blob[2], other[2])
            and abs(blob[1] - other[1]) < 0.5 * max(blob[3], other[3])
            for other in kept
        ):
            continue
        kept.append(blob)
    return kept, inverse, gray, hsv, probe


def _row_pitch_and_slopes(cars):
    if len(cars) < 3:
        return None
    med_w = float(np.median([car[2] for car in cars]))
    med_l = float(np.median([car[3] for car in cars]))
    pairs = []
    for index, car in enumerate(cars):
        best = None
        for other_index, other in enumerate(cars):
            dx = other[0] - car[0]
            if not (0.80 * med_w <= dx <= 2.5 * med_w):
                continue
            dy = other[1] - car[1]
            if abs(dy) > 0.85 * dx or abs(dy) > 0.65 * med_l:
                continue
            if best is None or dx < best[0]:
                best = (dx, dy, other_index)
        if best is not None:
            pairs.append((index, best[2], best[0], best[1] / best[0]))
    pitch = float(np.median([pair[2] for pair in pairs])) if pairs else med_w * 1.45
    slopes = np.array([pair[3] for pair in pairs]) if pairs else np.array([0.0])
    centers = [float(np.median(slopes))]
    return med_w, med_l, pitch, pairs, centers


def _poly_iou(quad_a, quad_b) -> float:
    area, _ = cv2.intersectConvexConvex(np.array(quad_a, np.float32), np.array(quad_b, np.float32))
    if area <= 0:
        return 0.0
    union = cv2.contourArea(np.array(quad_a, np.float32)) + cv2.contourArea(np.array(quad_b, np.float32)) - area
    return float(area) / float(max(1.0, union))


def _asphalt_level_frame(image: np.ndarray) -> float:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    plain = gray[(hsv[:, :, 1] < 32) & (gray > 40) & (gray < 190)]
    return float(np.median(plain)) if plain.size >= 40 else 140.0


def _slant_car_slots(image: np.ndarray, peaks: list[float]) -> list[Slot]:
    if not peaks:
        return []
    found: list[tuple] = []
    for peak in peaks:
        cars, inverse, gray, hsv, probe = _upright_vehicles(image, peak)
        fitted = _row_pitch_and_slopes(cars)
        if fitted is None:
            continue
        med_w, med_l, pitch, pairs, centers = fitted
        asphalt = _asphalt_level_frame(image)
        stall_l = float(np.clip(med_l, 38, med_l * 1.05))
        stall_w = float(np.clip(min(med_w * 1.1, pitch * 0.82), med_w * 0.72, pitch * 0.88))
        slope = centers[0]
        
        for car in cars:
            stats = _core_sample(gray, hsv, probe, car[0], car[1], stall_w, stall_l)
            if stats is None or stats[2] > 0.14 or stats[0] < 0.14:
                continue
            quad = _quad_from_upright(inverse, car[0], car[1], stall_w / 2, stall_l / 2, image.shape)
            if quad is not None:
                found.append((quad, False))
                
            for direction in (1, -1):
                cx = car[0] + direction * pitch
                cy = car[1] + direction * pitch * slope
                if any(abs(cx - other[0]) < pitch * 0.42 and abs(cy - other[1]) < med_l * 0.42 for other in cars):
                    continue
                stats = _core_sample(gray, hsv, probe, cx, cy, stall_w, stall_l)
                if stats is None or stats[2] > 0.08 or not (stats[0] <= 0.10 and stats[1] <= 18):
                    continue
                if abs(stats[3] - asphalt) > 26:
                    continue
                quad = _quad_from_upright(inverse, cx, cy, stall_w / 2, stall_l / 2, image.shape)
                if quad is not None:
                    found.append((quad, True))

    found.sort(key=lambda item: item[1])
    kept: list[tuple] = []
    for quad, empty in found:
        limit = 0.08 if empty else 0.20
        if any(_poly_iou(quad, other[0]) > limit for other in kept):
            continue
        kept.append((quad, empty))

    slots = []
    for quad, empty in kept:
        xs = [point[0] for point in quad]
        ys = [point[1] for point in quad]
        slots.append(
            Slot(
                status="empty" if empty else "occupied",
                confidence=0.82 if empty else 0.86,
                xyxy=(min(xs), min(ys), max(xs), max(ys)),
                quad=quad,
            )
        )
    return slots


def read_lined(image_bgr: np.ndarray) -> tuple[list[Slot], np.ndarray]:
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    peaks = _slant_peaks(gray)
    if peaks:
        slanted = _slant_car_slots(image_bgr, peaks)
        if len(slanted) >= _LINED_MIN_BAYS:
            return slanted, image_bgr

    vertical = _divider_weight(gray, vertical=True)
    horizontal = _divider_weight(gray, vertical=False)
    if horizontal > vertical * 1.25 and horizontal > 200:
        turned = cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
        slots = _slots_upright(turned)
        if len(slots) >= _LINED_MIN_BAYS:
            height = image_bgr.shape[0]
            mapped = [
                Slot(slot.status, slot.confidence, _unmap_clockwise(slot.xyxy, height))
                for slot in slots
            ]
            return mapped, image_bgr
    return _slots_upright(image_bgr), image_bgr


# Model paths supporting standard detection and oriented bounding boxes (OBB)
MODEL_OBB_PATH = MODELS / "parking-yolov8n-obb.pt"
MODEL_PATH = MODELS / "pklot-yolov8n.pt"
_CONFIDENCE = 0.25
_IMAGE_SIZE = 960
_MODEL_MIN_BAYS = 4


@lru_cache(maxsize=2)
def load_model(obb: bool = False):
    """Load parking-space weights (either oriented bounding box or standard PKLot)."""
    target = MODEL_OBB_PATH if obb and MODEL_OBB_PATH.is_file() else MODEL_PATH
    if not target.is_file():
        return None
    from ultralytics import YOLO
    return YOLO(str(target))


def _iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1, (bx2 - bx1) * (by2 - by1))
    return inter / float(area_a + area_b - inter)


def _dedupe(slots: list[Slot]) -> list[Slot]:
    ordered = sorted(slots, key=lambda slot: slot.confidence, reverse=True)
    kept: list[Slot] = []
    for slot in ordered:
        if any(_iou(slot.xyxy, other.xyxy) > 0.52 for other in kept):
            continue
        kept.append(slot)
    return kept


def _detect(image_bgr: np.ndarray) -> list[Slot]:
    """Run model inference, parsing oriented bounding boxes (obb) when present."""
    model = load_model(obb=True) or load_model(obb=False)
    if model is None:
        return []

    result = model.predict(
        image_bgr,
        verbose=False,
        conf=_CONFIDENCE,
        iou=0.45,
        imgsz=_IMAGE_SIZE,
    )[0]
    
    names = result.names
    height, width = image_bgr.shape[:2]
    frame = float(height * width)
    slots: list[Slot] = []

    # Check for YOLO OBB (Oriented Bounding Boxes: xyxyxyxy)
    if hasattr(result, "obb") and result.obb is not None and len(result.obb) > 0:
        for obb in result.obb:
            class_id = int(obb.cls[0])
            status = slot_status(str(names[class_id]))
            conf = float(obb.conf[0])
            poly = obb.xyxyxyxy[0].cpu().numpy().astype(int)
            quad = tuple((int(pt[0]), int(pt[1])) for pt in poly)
            xs = [pt[0] for pt in quad]
            ys = [pt[1] for pt in quad]
            xyxy = (max(0, min(xs)), max(0, min(ys)), min(width - 1, max(xs)), min(height - 1, max(ys)))
            slots.append(Slot(status=status, confidence=conf, xyxy=xyxy, quad=quad))
        return _dedupe(slots)

    if result.boxes is None:
        return slots

    for box in result.boxes:
        class_id = int(box.cls[0])
        status = slot_status(str(names[class_id]))
        x1, y1, x2, y2 = (int(round(v)) for v in box.xyxy[0].tolist())
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(width - 1, x2), min(height - 1, y2)
        area = max(0, x2 - x1) * max(0, y2 - y1)
        if area < frame * 0.00015 or area > frame * 0.035:
            continue
        if x2 - x1 < 6 or y2 - y1 < 6:
            continue
        slots.append(Slot(status=status, confidence=float(box.conf[0]), xyxy=(x1, y1, x2, y2)))
        
    return _dedupe(slots)


def _orientations(image_bgr: np.ndarray):
    yield image_bgr
    yield cv2.rotate(image_bgr, cv2.ROTATE_180)
    yield cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
    yield cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)


def read_overhead(image_bgr: np.ndarray) -> LotReading:
    best_slots: list[Slot] = []
    best_image = image_bgr
    for view in _orientations(image_bgr):
        slots = _detect(view)
        if len(slots) > len(best_slots):
            best_slots = slots
            best_image = view
        if len(slots) >= 15:
            best_slots = slots
            best_image = view
            break

    lined_slots, lined_image = read_lined(image_bgr)
    if len(lined_slots) > len(best_slots) and len(lined_slots) >= _LINED_MIN_BAYS:
        return LotReading(slots=lined_slots, same_camera=False, aligned_bgr=lined_image, source="lines")
    if len(best_slots) >= _MODEL_MIN_BAYS:
        return LotReading(slots=best_slots, same_camera=False, aligned_bgr=best_image, source="model")
    if len(lined_slots) >= _MODEL_MIN_BAYS:
        return LotReading(slots=lined_slots, same_camera=False, aligned_bgr=lined_image, source="lines")
    return LotReading(slots=[], same_camera=False, aligned_bgr=image_bgr, source="lines")


def list_samples() -> list[Path]:
    if not SAMPLES.is_dir():
        return []
    return sorted(SAMPLES.glob("*.jpg")) + sorted(SAMPLES.glob("*.png"))


@dataclass(frozen=True)
class OpenAverage:
    fraction: float
    open_bays: float
    frames: int


@lru_cache(maxsize=1)
def open_bay_average() -> OpenAverage:
    paths = list_samples()
    if not paths:
        return OpenAverage(_FALLBACK_OPEN_FRACTION, 46.0, 0)
    plate = load_reference(SAMPLES)
    fractions: list[float] = []
    opens: list[int] = []
    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        occupied, empty = count_slots(read_lot(image, plate).slots)
        total = occupied + empty
        if total == 0:
            continue
        fractions.append(empty / total)
        opens.append(empty)
    if not fractions:
        return OpenAverage(_FALLBACK_OPEN_FRACTION, 46.0, 0)
    return OpenAverage(
        fraction=sum(fractions) / len(fractions),
        open_bays=sum(opens) / len(opens),
        frames=len(fractions),
    )


def band_counts(total: int) -> tuple[int, int]:
    average = open_bay_average().fraction
    full_at = int(average * 0.5 * total)
    free_line = average * 2 * total
    free_at = int(free_line) if abs(free_line - round(free_line)) < 1e-6 else int(free_line) + 1
    return full_at, free_at


def read_rgb(file) -> np.ndarray:
    return np.array(Image.open(file).convert("RGB"))


def level_class(level: str) -> str:
    return {"Low": "low", "Moderate": "moderate", "High": "high"}.get(level, "none")


def cue_for(level: str) -> str | None:
    return {"Low": "clear", "Moderate": "warning", "High": "alarm"}.get(level)


def _silence(seconds: float) -> np.ndarray:
    return np.zeros(int(_SAMPLE_RATE * seconds), dtype=np.float64)


def _beep(frequency: float, seconds: float, volume: float, decay: float) -> np.ndarray:
    count = max(1, int(_SAMPLE_RATE * seconds))
    time = np.arange(count) / _SAMPLE_RATE
    tone = np.sin(2 * np.pi * frequency * time)
    tone += 0.22 * np.sin(2 * np.pi * frequency * 2 * time)
    edge = min(int(0.012 * _SAMPLE_RATE), count // 4)
    envelope = np.ones(count)
    if edge:
        envelope[:edge] = np.linspace(0.0, 1.0, edge)
        envelope[-edge:] = np.linspace(1.0, 0.0, edge)
    envelope *= np.exp(-decay * time / max(seconds, 0.05))
    return tone * envelope * volume


def _wav(samples: np.ndarray) -> bytes:
    clipped = np.clip(samples, -1.0, 1.0)
    pcm = (clipped * 32767).astype(np.int16)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(_SAMPLE_RATE)
        handle.writeframes(pcm.tobytes())
    return buffer.getvalue()


@lru_cache(maxsize=3)
def cue_audio(cue: str) -> bytes:
    if cue == "alarm":
        pattern = np.concatenate(
            [
                _beep(880, 0.16, 0.55, decay=0.4),
                _silence(0.06),
                _beep(1175, 0.16, 0.55, decay=0.4),
                _silence(0.12),
            ]
        )
        target = int(3.0 * _SAMPLE_RATE)
        repeats = int(np.ceil(target / len(pattern)))
        samples = np.tile(pattern, repeats)[:target]
        fade = int(0.08 * _SAMPLE_RATE)
        samples[-fade:] *= np.linspace(1.0, 0.0, fade)
    elif cue == "warning":
        samples = np.concatenate(
            [
                _beep(784, 0.14, 0.5, decay=2.2),
                _silence(0.04),
                _beep(523, 0.28, 0.5, decay=2.4),
            ]
        )
    elif cue == "clear":
        samples = np.concatenate(
            [
                _beep(523, 0.12, 0.45, decay=2.0),
                _beep(659, 0.12, 0.45, decay=2.0),
                _beep(784, 0.14, 0.48, decay=2.0),
                _beep(1047, 0.32, 0.42, decay=2.6),
            ]
        )
    else:
        raise ValueError(f"Unknown cue {cue}")
    return _wav(samples)


def _play_cue(cue: str) -> None:
    st.session_state.cue_play = st.session_state.get("cue_play", 0) + 1
    encoded = base64.b64encode(cue_audio(cue)).decode("ascii")
    st.markdown(
        f'<audio autoplay playsinline id="parkvision-cue-{st.session_state.cue_play}" '
        f'src="data:audio/wav;base64,{encoded}"></audio>',
        unsafe_allow_html=True,
    )


@st.cache_resource(show_spinner=False)
def reference():
    return load_reference(SAMPLES)


def _frame_label(name: str) -> str:
    stem = Path(name).stem
    if stem.startswith("lot-"):
        return stem[4:]
    return stem


def _mosaic_html(slots: list[Slot]) -> str:
    def center(slot: Slot) -> tuple[float, float]:
        if slot.quad:
            xs = [point[0] for point in slot.quad]
            ys = [point[1] for point in slot.quad]
            return (sum(ys) / len(ys), sum(xs) / len(xs))
        x1, y1, x2, y2 = slot.xyxy
        return ((y1 + y2) / 2, (x1 + x2) / 2)

    cells = []
    for slot in sorted(slots, key=center):
        kind = "open" if slot.status == "empty" else "taken"
        title = "Empty" if slot.status == "empty" else "Occupied"
        cells.append(f'<span class="cell {kind}" title="{title}"></span>')
    return (
        '<div class="mosaic" aria-label="Bays from the top of the frame">'
        + "".join(cells)
        + "</div>"
    )


def _rail_html(insights: ParkingInsights, slots: list[Slot], full_at: int, free_at: int, usual_open: float) -> str:
    tone = level_class(insights.level)
    color = {"low": "#3ddc84", "moderate": "#e0a23a", "high": "#ff5d5d"}.get(tone, "#8ea0b5")
    occupancy = max(0.0, min(100.0, insights.occupancy_pct))
    span = max(insights.total, 1)
    now = min(100.0, 100.0 * insights.available / span)
    full_mark = min(100.0, 100.0 * full_at / span)
    free_mark = min(100.0, 100.0 * free_at / span)
    return f"""
    <div class="rail {tone}">
      <div class="rail-top">
        <div class="ring" style="background: conic-gradient({color} {occupancy:.1f}%, rgba(232,238,246,0.08) 0);">
          <div class="ring-hole">
            <strong>{insights.occupancy_pct:.1f}%</strong>
            <span>full</span>
          </div>
        </div>
        <div class="verdict">
          <p class="chip">{insights.level}</p>
          <h2>{insights.available} open</h2>
          <p class="say">{insights.recommendation}</p>
        </div>
      </div>
      <div class="stats">
        <div><span>Total</span><strong>{insights.total}</strong></div>
        <div><span>Occupied</span><strong>{insights.occupied}</strong></div>
        <div><span>Available</span><strong>{insights.available}</strong></div>
        <div><span>Occupancy</span><strong>{insights.occupancy_pct:.1f}%</strong></div>
      </div>
      {_mosaic_html(slots)}
      <p class="mosaic-note">Each square is one bay, from the top of the frame.</p>
      <div class="band">
        <div class="band-track">
          <div class="band-now" style="left:{now:.1f}%"></div>
          <div class="band-tick" style="left:{full_mark:.1f}%"></div>
          <div class="band-tick free" style="left:{free_mark:.1f}%"></div>
        </div>
        <div class="band-labels">
          <span>Full at {full_at}</span>
          <span>{insights.available} open now</span>
          <span>Free at {free_at}</span>
        </div>
      </div>
      <p class="usual">A normal frame of the sample camera leaves {usual_open:.0f} bays open. The sound for this band plays on its own.</p>
    </div>
    """


_PAGE_CSS = """
<style>
  html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {
    background: #090c11;
    color: #e8eef6;
    font-family: Inter, "Segoe UI", sans-serif;
  }
  header[data-testid="stHeader"],
  [data-testid="stToolbar"],
  [data-testid="stDecoration"],
  [data-testid="stStatusWidget"] {
    display: none !important;
    height: 0 !important;
  }
  #MainMenu, footer { visibility: hidden; height: 0; }
  .block-container {
    max-width: 1180px;
    padding-top: 1.15rem;
    padding-bottom: 2.6rem;
  }
  .brand { display: flex; align-items: center; gap: 0.85rem; margin: 0 0 0.35rem; }
  .brand img { width: 3.1rem; height: 3.1rem; border-radius: 0.8rem; }
  .brand .eyebrow {
    margin: 0;
    color: #8fb4ff;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    font-size: 0.68rem;
  }
  .brand h1 { margin: 0; font-size: 1.55rem; letter-spacing: -0.03em; line-height: 1.1; }
  .lede { color: #9aabbe; margin: 0 0 0.85rem; max-width: 40rem; font-size: 0.95rem; }
  .st-key-stage, .st-key-railbox {
    background: #12181f;
    border: 1px solid rgba(232, 238, 246, 0.08);
    border-radius: 18px;
    padding: 0.85rem 0.9rem 0.35rem;
  }
  .view-label {
    margin: 0 0 0.45rem;
    color: #8ea0b5;
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
  }
  [data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid rgba(232, 238, 246, 0.08);
  }
  .legend { display: flex; gap: 0.9rem; color: #c5d0de; font-size: 0.82rem; margin: 0.15rem 0 0.2rem; }
  .swatch {
    display: inline-block; width: 0.7rem; height: 0.7rem;
    border-radius: 2px; margin-right: 0.35rem; vertical-align: -1px;
  }
  .rail { display: flex; flex-direction: column; gap: 0.85rem; }
  .rail-top { display: flex; gap: 0.9rem; align-items: center; }
  .ring {
    width: 148px; height: 148px; border-radius: 50%;
    display: grid; place-items: center; flex: none;
  }
  .ring-hole {
    width: 108px; height: 108px; border-radius: 50%;
    background: #12181f;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
  }
  .ring-hole strong {
    font-family: "JetBrains Mono", ui-monospace, monospace;
    font-size: 1.35rem; letter-spacing: -0.04em; line-height: 1;
  }
  .ring-hole span { color: #8ea0b5; font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; }
  .chip {
    display: inline-block; margin: 0 0 0.25rem;
    padding: 0.12rem 0.45rem; border-radius: 999px;
    font-size: 0.68rem; letter-spacing: 0.12em; text-transform: uppercase;
    border: 1px solid transparent;
  }
  .low .chip { color: #3ddc84; border-color: rgba(61, 220, 132, 0.45); background: rgba(61, 220, 132, 0.1); }
  .moderate .chip { color: #e0a23a; border-color: rgba(224, 162, 58, 0.45); background: rgba(224, 162, 58, 0.1); }
  .high .chip { color: #ff5d5d; border-color: rgba(255, 93, 93, 0.5); background: rgba(255, 93, 93, 0.1); }
  .none .chip { color: #8ea0b5; border-color: rgba(142, 160, 181, 0.4); }
  .verdict h2 {
    margin: 0; font-size: 1.85rem; letter-spacing: -0.04em; line-height: 1;
    font-family: "JetBrains Mono", ui-monospace, monospace;
  }
  .verdict .say { margin: 0.4rem 0 0; color: #d5deea; font-size: 0.95rem; }
  .stats { display: grid; grid-template-columns: repeat(4, 1fr); gap: 0.45rem; }
  .stats div {
    background: rgba(232, 238, 246, 0.04);
    border-radius: 10px; padding: 0.45rem 0.5rem 0.4rem;
  }
  .stats span { display: block; color: #8ea0b5; font-size: 0.68rem; letter-spacing: 0.06em; text-transform: uppercase; }
  .stats strong {
    font-family: "JetBrains Mono", ui-monospace, monospace;
    font-size: 1.15rem; font-weight: 560;
  }
  .mosaic { display: flex; flex-wrap: wrap; gap: 3px; }
  .cell { width: 12px; height: 16px; border-radius: 2px; display: inline-block; }
  .cell.open { background: #3ddc84; }
  .cell.taken { background: #ff5d5d; }
  .mosaic-note, .usual, .fineprint { color: #8ea0b5; font-size: 0.8rem; margin: 0; }
  .band-track {
    position: relative; height: 6px; border-radius: 999px;
    background: rgba(232, 238, 246, 0.1); margin: 0.2rem 0 0.35rem;
  }
  .band-now {
    position: absolute; top: -4px; width: 14px; height: 14px; border-radius: 50%;
    background: #e8eef6; transform: translateX(-50%);
    box-shadow: 0 0 0 3px rgba(232, 238, 246, 0.16);
  }
  .low .band-now { background: #3ddc84; }
  .moderate .band-now { background: #e0a23a; }
  .high .band-now { background: #ff5d5d; }
  .band-tick {
    position: absolute; top: -2px; width: 2px; height: 10px;
    background: rgba(232, 238, 246, 0.45); transform: translateX(-50%);
  }
  .band-labels { display: flex; justify-content: space-between; gap: 0.4rem; color: #8ea0b5; font-size: 0.72rem; }
  .empty-stage {
    border: 1px dashed rgba(143, 180, 255, 0.35);
    border-radius: 18px; padding: 2.2rem 1.4rem; text-align: center;
    background: #10161e;
  }
  .empty-stage h2 { margin: 0.4rem 0 0.3rem; font-size: 1.25rem; }
  .empty-stage p { margin: 0 auto; max-width: 34rem; color: #9aabbe; }
  [data-testid="stFileUploaderDropzone"] {
    background: #10161e;
    border: 1px dashed rgba(143, 180, 255, 0.45);
    border-radius: 16px;
  }
  .cue { height: 0; overflow: hidden; }
  @media (max-width: 720px) {
    .rail-top { flex-direction: column; align-items: flex-start; }
    .stats { grid-template-columns: repeat(2, 1fr); }
    .brand h1 { font-size: 1.3rem; }
    .verdict h2 { font-size: 1.55rem; }
  }
</style>
"""

_LOGO_PNG = (
    "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"
)


def _logo_bytes() -> bytes:
    path = ROOT / "assets" / "logo.png"
    if path.is_file():
        return path.read_bytes()
    return base64.b64decode(_LOGO_PNG)


def main() -> None:
    logo = _logo_bytes()
    icon_path = ROOT / "assets" / "logo.png"
    st.set_page_config(
        page_title="ParkVision AI",
        page_icon=str(icon_path) if icon_path.is_file() else Image.open(io.BytesIO(logo)),
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.markdown(_PAGE_CSS, unsafe_allow_html=True)

    mark = base64.b64encode(logo).decode("ascii")
    st.markdown(
        f"""
        <div class="brand">
          <img src="data:image/png;base64,{mark}" alt="ParkVision">
          <div>
            <p class="eyebrow">Overhead lot reading</p>
            <h1>ParkVision AI</h1>
          </div>
        </div>
        <p class="lede">
          An overhead shot of a lined lot. Empty bays are green, occupied bays
          are red, and the call tells a driver whether to pull in.
        </p>
        """,
        unsafe_allow_html=True,
    )

    samples = list_samples()
    source_options = ["Sample lot", "Upload"] if samples else ["Upload"]
    with st.container(key="controls"):
        source = st.segmented_control(
            "Frame",
            source_options,
            default=source_options[0],
            selection_mode="single",
            key="source_mode",
        )
        upload = None
        sample_path = None
        if source == "Upload":
            upload = st.file_uploader(
                "Overhead photo",
                type=["jpg", "jpeg", "png", "webp"],
                help="An overhead photo of a lined parking lot. It does not have to be the sample camera.",
            )
        elif samples and source == "Sample lot":
            labels = [path.name for path in samples]
            default = "lot-01.jpg" if "lot-01.jpg" in labels else labels[0]
            choice = st.pills(
                "Saved frames",
                labels,
                default=default,
                format_func=_frame_label,
                selection_mode="single",
                key="frame",
            )
            if choice:
                sample_path = samples[labels.index(choice)]

    image = None
    if upload is not None:
        image = read_rgb(upload)
    elif sample_path is not None:
        image = read_rgb(sample_path)

    if image is None:
        if source != "Upload":
            st.markdown(
                """
                <div class="empty-stage">
                  <h2>Choose a frame</h2>
                  <p>Pick a saved lot. The chips above are the frames from this camera.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.stop()

    with st.spinner("Reading this frame…"):
        plate = reference()
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        reading = read_lot(bgr, plate)
        if not reading.slots:
            reading = read_overhead(bgr)

    if not reading.slots:
        st.image(image, width="stretch")
        st.markdown(
            """
            <div class="empty-stage">
              <h2>No bays in this photo</h2>
              <p>Use an overhead shot where the stall lines are visible, or pick a saved frame.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    shown = cv2.cvtColor(reading.aligned_bgr, cv2.COLOR_BGR2RGB)
    occupied, empty = count_slots(reading.slots)
    insights = summarize(occupied, empty)
    annotated = annotate(shown, reading.slots)
    usual = open_bay_average()
    full_at, free_at = band_counts(insights.total)

    stage, rail = st.columns([1.42, 0.78], gap="medium")
    with stage:
        with st.container(key="stage"):
            view = st.segmented_control(
                "View",
                ["Split", "Overlay", "Camera"],
                default="Split",
                selection_mode="single",
                key="view_mode",
            )
            if view is None:
                view = "Split"
            st.markdown(
                """
                <div class="legend">
                  <span><i class="swatch" style="background:#3ddc84"></i>Empty</span>
                  <span><i class="swatch" style="background:#ff5d5d"></i>Occupied</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if view == "Split":
                camera, overlay = st.columns(2, gap="small")
                with camera:
                    st.markdown('<p class="view-label">Camera</p>', unsafe_allow_html=True)
                    st.image(shown, width="stretch")
                with overlay:
                    st.markdown('<p class="view-label">Overlay</p>', unsafe_allow_html=True)
                    st.image(annotated, width="stretch")
            elif view == "Camera":
                st.markdown('<p class="view-label">Camera</p>', unsafe_allow_html=True)
                st.image(shown, width="stretch")
            else:
                st.markdown('<p class="view-label">Overlay</p>', unsafe_allow_html=True)
                st.image(annotated, width="stretch")
    with rail:
        with st.container(key="railbox"):
            st.markdown(
                _rail_html(insights, reading.slots, full_at, free_at, usual.open_bays),
                unsafe_allow_html=True,
            )

    if reading.same_camera:
        note = (
            "This frame matched the sample camera, so the bays come from that view's map. "
            "The empty plate is samples/lot-11.jpg (Almeida et al., 2015, CC BY 4.0)."
        )
    elif reading.source == "lines":
        note = "These bays were traced from the painted lines. Empty asphalt is green. A bay with a vehicle in it is red."
    else:
        note = (
            "These bays were found in the photo itself, so the shot does not have to be the sample camera. "
            "The reader is a YOLOv8n model trained on PKLot."
        )
    st.markdown(f'<p class="fineprint">{note}</p>', unsafe_allow_html=True)

    with st.expander("How the call is made"):
        st.markdown(
            """
            A photo of the sample camera uses a fixed grid of its lined rows.
            Each bay is compared with an empty plate of the same lot. A poor
            match means the bay is occupied. A bay that still roughly matches,
            but is much busier than the empty asphalt, is counted too.

            Any other overhead shot is read from the painted stall lines when
            they are clear, and otherwise by a parking-space model trained on
            lined lots. On a diagonal or herringbone lot, each stall is boxed
            along its row. An open stall is marked when it sits between two
            cars, or when both painted sides are visible and the middle is
            bare asphalt.

            The average number of open bays is taken across the saved frames.
            Twice that average, scaled to this lot, is free and plays a short
            chime. Half of it is full and plays a 3 second alarm. Anywhere
            between is busy and plays a warning. The sound starts on its own,
            including when the next lot falls in the same band.
            """
        )

    cue = cue_for(insights.level)
    if cue is not None:
        _play_cue(cue)


if __name__ == "__main__":
    main()
