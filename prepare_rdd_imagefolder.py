import argparse
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Common RDD2022 class IDs used in YOLO labels.
DEFAULT_CLASS_MAP = {
    0: "D00_longitudinal_crack",
    1: "D10_transverse_crack",
    2: "D20_alligator_crack",
    3: "D40_pothole",
    4: "D43_crosswalk_blur",
}


def parse_label_file(label_path: Path):
    boxes = []
    if not label_path.exists() or label_path.stat().st_size == 0:
        return boxes

    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            cls_id = int(float(parts[0]))
            w = float(parts[3])
            h = float(parts[4])
            boxes.append((cls_id, w * h))
    return boxes


def pick_image_label(boxes):
    if not boxes:
        return None
    # Use the class with the largest bbox area as the image-level label.
    return max(boxes, key=lambda item: item[1])[0]


def materialize(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return

    if mode == "copy":
        shutil.copy2(src, dst)
        return

    if mode == "hardlink":
        try:
            dst.hardlink_to(src)
            return
        except OSError:
            shutil.copy2(src, dst)
            return

    # symlink mode
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def convert_split(src_root: Path, out_root: Path, split: str, class_map: dict, mode: str, include_empty: bool, limit: int | None):
    img_dir = src_root / split / "images"
    lbl_dir = src_root / split / "labels"

    images = [p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    images.sort()

    stats = {name: 0 for name in class_map.values()}
    skipped_empty = 0
    unknown_class = 0

    processed = 0
    for image_path in images:
        if limit is not None and processed >= limit:
            break

        label_path = lbl_dir / f"{image_path.stem}.txt"
        boxes = parse_label_file(label_path)
        label_id = pick_image_label(boxes)

        if label_id is None:
            if include_empty:
                label_name = "no_damage"
            else:
                skipped_empty += 1
                continue
        else:
            if label_id not in class_map:
                unknown_class += 1
                continue
            label_name = class_map[label_id]

        dst = out_root / split / label_name / image_path.name
        materialize(image_path, dst, mode)
        stats[label_name] = stats.get(label_name, 0) + 1
        processed += 1

    return {
        "processed": processed,
        "skipped_empty": skipped_empty,
        "unknown_class": unknown_class,
        "stats": stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Convert RDD YOLO labels into ImageFolder classification layout")
    parser.add_argument("--source", type=str, default="archive/RDD_SPLIT", help="Path with train/val/test images+labels")
    parser.add_argument("--output", type=str, default="data/rdd_imagefolder", help="Output ImageFolder root")
    parser.add_argument("--splits", nargs="+", default=["train", "val"], help="Splits to convert")
    parser.add_argument("--mode", choices=["symlink", "hardlink", "copy"], default="symlink", help="How to materialize images")
    parser.add_argument("--include-empty", action="store_true", help="Include images with empty labels as no_damage")
    parser.add_argument("--limit", type=int, default=None, help="Optional max images per split (for smoke tests)")
    args = parser.parse_args()

    src_root = Path(args.source)
    out_root = Path(args.output)

    if not src_root.exists():
        raise FileNotFoundError(f"Source dataset path not found: {src_root}")

    class_map = dict(DEFAULT_CLASS_MAP)
    if args.include_empty:
        class_map[999] = "no_damage"

    print(f"Source: {src_root}")
    print(f"Output: {out_root}")
    print(f"Splits: {args.splits}")
    print(f"Mode: {args.mode}")

    for split in args.splits:
        split_root = src_root / split
        if not split_root.exists():
            print(f"Skipping missing split: {split}")
            continue

        result = convert_split(
            src_root=src_root,
            out_root=out_root,
            split=split,
            class_map=class_map,
            mode=args.mode,
            include_empty=args.include_empty,
            limit=args.limit,
        )

        print(f"\n[{split}] processed={result['processed']} skipped_empty={result['skipped_empty']} unknown_class={result['unknown_class']}")
        for cls_name, count in sorted(result["stats"].items()):
            print(f"  {cls_name}: {count}")


if __name__ == "__main__":
    main()
