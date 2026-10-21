ROOT = Path(__file__).resolve().parents[1]
def asphalt(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = rng.integers(38, 72, size=(h, w, 3), dtype=np.uint8)
    noise = rng.integers(-12, 13, size=(h, w, 3))
    img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # lighting / weather tint
    weather = seed % 5
    if weather == 0:  # sunny
        img = np.clip(img.astype(np.int16) + 18, 0, 255).astype(np.uint8)
    elif weather == 1:  # cloudy
        img = (img.astype(np.float32) * 0.78).astype(np.uint8)
    elif weather == 2:  # rainy / wet
        img = np.clip(img.astype(np.int16) - 10, 0, 255).astype(np.uint8)
        for _ in range(40):
            x = rng.integers(0, w)
            y = rng.integers(0, h)
            cv2.line(img, (x, y), (x + int(rng.integers(4, 14)), y + int(rng.integers(8, 22))), (190, 200, 210), 1)
    return img
def draw_empty_slot(img: np.ndarray, seed: int) -> np.ndarray:
    h, w = img.shape[:2]
    color = (230, 230, 230)
    cv2.rectangle(img, (18, 12), (w - 18, h - 12), color, 3)
    for y in range(28, h - 20, 18):
        cv2.line(img, (w // 2, y), (w // 2, y + 8), color, 2)
    if seed % 7 == 0:
        # shadow on empty asphalt
        overlay = img.copy()
        cv2.ellipse(overlay, (w // 2, h // 2 + 20), (w // 3, h // 6), 0, 0, 360, (20, 20, 20), -1)
        img = cv2.addWeighted(overlay, 0.25, img, 0.75, 0)
    return img
def draw_car(img: np.ndarray, seed: int) -> np.ndarray:
    h, w = img.shape[:2]
    rng = random.Random(seed)
    colors = [
        (40, 80, 190),
        (30, 30, 30),
        (200, 200, 205),
        (40, 140, 80),
        (20, 40, 160),
        (170, 50, 40),
    ]
    color = colors[seed % len(colors)]
    x1, y1 = 28 + rng.randint(-6, 6), 30 + rng.randint(-8, 8)
    x2, y2 = w - 28 + rng.randint(-6, 6), h - 28 + rng.randint(-8, 8)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    cv2.rectangle(img, (x1 + 10, y1 + 12), (x2 - 10, y1 + 42), (180, 220, 235), -1)
    cv2.rectangle(img, (x1 + 12, y2 - 38), (x2 - 12, y2 - 14), (30, 30, 30), -1)
    cv2.circle(img, (x1 + 16, y2 - 6), 10, (15, 15, 15), -1)
    cv2.circle(img, (x2 - 16, y2 - 6), 10, (15, 15, 15), -1)
    if seed % 4 == 0:
        # partial occlusion: crop the car visually at the border
        cv2.rectangle(img, (0, 0), (w // 5, h), (50, 50, 50), -1)
    return img
def make_slot(label: str, idx: int, size: int = 224) -> np.ndarray:
    img = asphalt(size, size, idx)
    if label == "empty":
        img = draw_empty_slot(img, idx)
    else:
        img = draw_empty_slot(img, idx)
        img = draw_car(img, idx)
    # brightness jitter
    alpha = 0.75 + (idx % 9) * 0.05
    img = np.clip(img.astype(np.float32) * alpha, 0, 255).astype(np.uint8)
    return img
def write_slots(out_dir: Path, per_class: int) -> None:
    for label in ("empty", "occupied"):
        folder = out_dir / label
        folder.mkdir(parents=True, exist_ok=True)
        offset = 0 if label == "empty" else 10_000
        for i in range(per_class):
            image = make_slot(label, offset + i)
            cv2.imwrite(str(folder / f"{label}_{i:04d}.jpg"), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
def make_full_lot(rows: int, cols: int, occupied_ratio: float, seed: int, slot: int = 160) -> np.ndarray:
    rng = random.Random(seed)
    tiles = []
    for r in range(rows):
        row_tiles = []
        for c in range(cols):
            idx = seed * 100 + r * cols + c
            label = "occupied" if rng.random() < occupied_ratio else "empty"
            row_tiles.append(make_slot(label, idx, size=slot))
        tiles.append(np.concatenate(row_tiles, axis=1))
    lot = np.concatenate(tiles, axis=0)
    return lot
def write_samples(sample_dir: Path) -> None:
    sample_dir.mkdir(parents=True, exist_ok=True)
    lots = [
        ("lot_low_congestion.jpg", 3, 4, 0.25, 1),
        ("lot_moderate_congestion.jpg", 3, 4, 0.55, 2),
        ("lot_high_congestion.jpg", 3, 4, 0.9, 3),
        ("lot_sunny.jpg", 2, 5, 0.4, 4),
        ("lot_cloudy.jpg", 4, 4, 0.6, 11),
    ]
    for name, rows, cols, ratio, seed in lots:
        image = make_full_lot(rows, cols, ratio, seed)
        cv2.imwrite(str(sample_dir / name), cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--per-class", type=int, default=120)
    args = parser.parse_args()
    write_slots(ROOT / "data" / "raw", args.per_class)
    write_samples(ROOT / "data" / "samples")
    print(f"Wrote {args.per_class} images per class to data/raw/{{empty,occupied}}")
    print("Wrote demo parking-lot images to data/samples/")
if __name__ == "__main__":
    main()
