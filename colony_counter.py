
streamlit run app.py

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import numpy as np

try:
    import cv2
except ModuleNotFoundError as exc:
    raise SystemExit(
        "OpenCV is required for this script. Install dependencies with:\n"
        "  python3 -m pip install -r requirements.txt"
    ) from exc


@dataclass
class ContourDecision:
    contour_id: int
    area: float
    circularity: float
    solidity: float
    aspect_ratio: float
    bbox: tuple[int, int, int, int]
    decision: str
    reasons: List[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Count well-isolated colonies automatically and flag suspicious "
            "contours for human review."
        )
    )
    parser.add_argument("image", type=Path, help="Path to the plate image.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for annotated outputs and review tables.",
    )
    parser.add_argument(
        "--min-area",
        type=float,
        default=80.0,
        help="Ignore contours smaller than this many pixels.",
    )
    parser.add_argument(
        "--area-multiplier",
        type=float,
        default=1.8,
        help=(
            "Flag contours for review when area is much larger than the median "
            "single-colony area. Higher values flag fewer colonies."
        ),
    )
    parser.add_argument(
        "--review-area-threshold",
        type=float,
        default=None,
        help=(
            "Optional manual review threshold in pixels. If omitted, the script "
            "estimates one from the colony area distribution."
        ),
    )
    parser.add_argument(
        "--min-circularity",
        type=float,
        default=0.70,
        help="Flag contours below this circularity for review.",
    )
    parser.add_argument(
        "--min-solidity",
        type=float,
        default=0.90,
        help="Flag contours below this solidity for review.",
    )
    parser.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=1.60,
        help="Flag contours more elongated than this for review.",
    )
    parser.add_argument(
        "--invert-mask",
        action="store_true",
        help="Use this if colonies are brighter than the background.",
    )
    parser.add_argument(
        "--save-mask",
        action="store_true",
        help="Also save the binary mask used for contour detection.",
    )
    return parser.parse_args()


def build_mask(image: np.ndarray, invert_mask: bool) -> np.ndarray:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    threshold_mode = cv2.THRESH_BINARY if invert_mask else cv2.THRESH_BINARY_INV
    _, mask = cv2.threshold(
        blurred,
        0,
        255,
        threshold_mode + cv2.THRESH_OTSU,
    )

    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    return mask


def find_candidate_contours(mask: np.ndarray, min_area: float) -> list[np.ndarray]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return [contour for contour in contours if cv2.contourArea(contour) >= min_area]


def compute_circularity(area: float, perimeter: float) -> float:
    if perimeter == 0:
        return 0.0
    return float((4.0 * math.pi * area) / (perimeter * perimeter))


def compute_solidity(area: float, contour: np.ndarray) -> float:
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if hull_area == 0:
        return 0.0
    return float(area / hull_area)


def suggest_review_area_threshold(
    areas: Iterable[float],
    area_multiplier: float,
    manual_threshold: float | None,
) -> tuple[float, dict[str, float]]:
    area_array = np.array(list(areas), dtype=np.float32)
    if area_array.size == 0:
        return float("inf"), {
            "median": 0.0,
            "q1": 0.0,
            "q3": 0.0,
            "iqr": 0.0,
            "mad": 0.0,
            "suggested": float("inf"),
        }

    q1, median, q3 = np.percentile(area_array, [25, 50, 75])
    iqr = q3 - q1
    mad = np.median(np.abs(area_array - median))
    robust_sigma = 1.4826 * mad

    # Human-review threshold strategy:
    # Use the median contour area as the "typical single colony" size, then
    # flag anything much larger than that as suspicious. This keeps the script
    # conservative: likely overlaps are reviewed by a person instead of being
    # silently added to the automatic count.
    multiplier_threshold = median * area_multiplier
    iqr_threshold = q3 + (1.5 * iqr) if iqr > 0 else multiplier_threshold
    mad_threshold = median + (3.0 * robust_sigma) if robust_sigma > 0 else multiplier_threshold

    suggested_threshold = max(multiplier_threshold, iqr_threshold, mad_threshold)
    if manual_threshold is not None:
        suggested_threshold = manual_threshold

    return float(suggested_threshold), {
        "median": float(median),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(iqr),
        "mad": float(mad),
        "suggested": float(suggested_threshold),
    }


def touches_border(bbox: tuple[int, int, int, int], image_shape: tuple[int, int, int]) -> bool:
    x, y, w, h = bbox
    image_height, image_width = image_shape[:2]
    return x <= 0 or y <= 0 or (x + w) >= image_width - 1 or (y + h) >= image_height - 1


def classify_contour(
    contour: np.ndarray,
    contour_id: int,
    review_area_threshold: float,
    min_circularity: float,
    min_solidity: float,
    max_aspect_ratio: float,
    image_shape: tuple[int, int, int],
) -> ContourDecision:
    area = float(cv2.contourArea(contour))
    perimeter = float(cv2.arcLength(contour, True))
    circularity = compute_circularity(area, perimeter)
    solidity = compute_solidity(area, contour)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(max(w / max(h, 1), h / max(w, 1)))

    reasons: List[str] = []

    # IMPORTANT DESIGN DECISION FOR THE WORKSHEET:
    # We only auto-count contours that look like one clear, isolated colony.
    # If a contour is unusually large, irregular, elongated, or touches the edge,
    # we DO NOT auto-count it because it may contain merged / overlapping colonies
    # or only part of a colony. Those cases are flagged for human review instead.
    if area > review_area_threshold:
        reasons.append(f"area {area:.1f} > review threshold {review_area_threshold:.1f}")
    if circularity < min_circularity:
        reasons.append(
            f"circularity {circularity:.2f} < minimum {min_circularity:.2f}"
        )
    if solidity < min_solidity:
        reasons.append(f"solidity {solidity:.2f} < minimum {min_solidity:.2f}")
    if aspect_ratio > max_aspect_ratio:
        reasons.append(
            f"aspect ratio {aspect_ratio:.2f} > maximum {max_aspect_ratio:.2f}"
        )
    if touches_border((x, y, w, h), image_shape):
        reasons.append("touches image border")

    decision = "review" if reasons else "count"
    return ContourDecision(
        contour_id=contour_id,
        area=area,
        circularity=circularity,
        solidity=solidity,
        aspect_ratio=aspect_ratio,
        bbox=(x, y, w, h),
        decision=decision,
        reasons=reasons,
    )


def annotate_image(
    image: np.ndarray,
    contours: list[np.ndarray],
    decisions: list[ContourDecision],
) -> np.ndarray:
    annotated = image.copy()

    for contour, decision in zip(contours, decisions):
        color = (0, 180, 0) if decision.decision == "count" else (0, 165, 255)
        label = str(decision.contour_id) if decision.decision == "count" else f"R{decision.contour_id}"
        x, y, _, _ = decision.bbox

        cv2.drawContours(annotated, [contour], -1, color, 2)
        cv2.putText(
            annotated,
            label,
            (x, max(y - 6, 16)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return annotated


def write_review_csv(path: Path, decisions: list[ContourDecision]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "contour_id",
                "decision",
                "area_px",
                "circularity",
                "solidity",
                "aspect_ratio",
                "bbox_x",
                "bbox_y",
                "bbox_w",
                "bbox_h",
                "review_reasons",
            ]
        )
        for decision in decisions:
            x, y, w, h = decision.bbox
            writer.writerow(
                [
                    decision.contour_id,
                    decision.decision,
                    f"{decision.area:.2f}",
                    f"{decision.circularity:.3f}",
                    f"{decision.solidity:.3f}",
                    f"{decision.aspect_ratio:.3f}",
                    x,
                    y,
                    w,
                    h,
                    " | ".join(decision.reasons),
                ]
            )


def main() -> None:
    args = parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")

    mask = build_mask(image, invert_mask=args.invert_mask)
    contours = find_candidate_contours(mask, min_area=args.min_area)
    areas = [float(cv2.contourArea(contour)) for contour in contours]

    review_area_threshold, area_stats = suggest_review_area_threshold(
        areas,
        area_multiplier=args.area_multiplier,
        manual_threshold=args.review_area_threshold,
    )

    decisions: list[ContourDecision] = []
    for contour_id, contour in enumerate(contours, start=1):
        decisions.append(
            classify_contour(
                contour=contour,
                contour_id=contour_id,
                review_area_threshold=review_area_threshold,
                min_circularity=args.min_circularity,
                min_solidity=args.min_solidity,
                max_aspect_ratio=args.max_aspect_ratio,
                image_shape=image.shape,
            )
        )

    auto_count = sum(decision.decision == "count" for decision in decisions)
    review_count = sum(decision.decision == "review" for decision in decisions)

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    annotated = annotate_image(image, contours, decisions)
    annotated_path = output_dir / f"{args.image.stem}_annotated.png"
    review_table_path = output_dir / f"{args.image.stem}_review.csv"

    cv2.imwrite(str(annotated_path), annotated)
    write_review_csv(review_table_path, decisions)

    if args.save_mask:
        mask_path = output_dir / f"{args.image.stem}_mask.png"
        cv2.imwrite(str(mask_path), mask)

    print(f"Image: {args.image}")
    print(f"Candidate contours: {len(contours)}")
    print(f"Auto-counted colonies: {auto_count}")
    print(f"Flagged for review: {review_count}")
    print()
    print("Area threshold guidance")
    print(f"  median area: {area_stats['median']:.1f} px")
    print(f"  q1-q3 area: {area_stats['q1']:.1f} px to {area_stats['q3']:.1f} px")
    print(f"  IQR: {area_stats['iqr']:.1f} px")
    print(f"  MAD: {area_stats['mad']:.1f} px")
    print(f"  review area threshold used: {area_stats['suggested']:.1f} px")
    print()
    print(
        "Tip: if obvious single colonies are being flagged, raise --area-multiplier "
        "or pass --review-area-threshold manually."
    )
    print(
        "Tip: if merged colonies are slipping into the auto-count, lower "
        "--area-multiplier or tighten the circularity / solidity cutoffs."
    )
    print()
    print(f"Annotated image saved to: {annotated_path}")
    print(f"Review table saved to: {review_table_path}")

    if review_count:
        print()
        print("Review list")
        for decision in decisions:
            if decision.decision == "review":
                print(
                    f"  R{decision.contour_id}: "
                    f"{'; '.join(decision.reasons)}"
                )


if __name__ == "__main__":
    main()

