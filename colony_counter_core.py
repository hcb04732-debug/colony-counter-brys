from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np


@dataclass
class ColonySettings:
    plate_radius_fraction: float = 0.43
    lightness_threshold: int = 180
    warm_tone_threshold: int = 145
    blob_min_threshold: int = 140
    min_contour_area: int = 300
    area_multiplier: float = 1.8
    min_review_circularity: float = 0.55
    min_review_solidity: float = 0.85
    max_review_aspect_ratio: float = 1.7
    edge_artifact_margin: float = 120.0


@dataclass
class CircleDetection:
    circle_id: int
    x: int
    y: int
    radius: int


@dataclass
class ReviewRegion:
    review_id: int
    area: float
    circularity: float
    solidity: float
    aspect_ratio: float
    edge_margin: float
    bbox: tuple[int, int, int, int]
    circle_ids: list[int]
    reasons: list[str] = field(default_factory=list)


@dataclass
class AnalysisResult:
    image_path: Path
    auto_count: int
    raw_circle_count: int
    review_count: int
    ignored_artifact_count: int
    review_area_threshold: float
    median_contour_area: float
    plate_center: tuple[int, int]
    plate_radius: int
    circles: list[CircleDetection]
    review_regions: list[ReviewRegion]
    annotated_image: np.ndarray
    mask_image: np.ndarray


def review_regions_to_rows(result: AnalysisResult) -> list[dict[str, str | int | float]]:
    rows: list[dict[str, str | int | float]] = []
    for region in result.review_regions:
        rows.append(
            {
                "review_id": f"R{region.review_id}",
                "circle_ids": ", ".join(str(circle_id) for circle_id in region.circle_ids)
                or "-",
                "area_px": round(region.area, 1),
                "circularity": round(region.circularity, 2),
                "solidity": round(region.solidity, 2),
                "aspect_ratio": round(region.aspect_ratio, 2),
                "edge_margin_px": round(region.edge_margin, 1),
                "reasons": " | ".join(region.reasons),
            }
        )
    return rows


def _build_plate_mask(
    image: np.ndarray,
    settings: ColonySettings,
) -> tuple[np.ndarray, tuple[int, int], int]:
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    radius = int(min(height, width) * settings.plate_radius_fraction)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    return mask, center, radius


def _detect_clear_circles(
    image: np.ndarray,
    plate_mask: np.ndarray,
    settings: ColonySettings,
) -> list[CircleDetection]:
    lightness = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]
    blurred = cv2.medianBlur(lightness, 9)
    masked = cv2.bitwise_and(blurred, blurred, mask=plate_mask)

    params = cv2.SimpleBlobDetector_Params()
    params.filterByColor = True
    params.blobColor = 255
    params.minThreshold = settings.blob_min_threshold
    params.maxThreshold = 255
    params.thresholdStep = 10
    params.filterByArea = True
    # Blob centers are smaller than the full contour of a colony, so tie the
    # blob-area floor to the user-facing minimum contour area at a lower scale.
    # This lets smaller colonies become countable when the minimum-area slider
    # is reduced, without opening the detector to tiny speck noise.
    params.minArea = max(30, int(settings.min_contour_area * 0.4))
    params.maxArea = 18000
    params.filterByCircularity = True
    params.minCircularity = 0.52
    params.filterByConvexity = True
    params.minConvexity = 0.70
    params.filterByInertia = False

    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(masked)

    circles: list[CircleDetection] = []
    for circle_id, keypoint in enumerate(keypoints, start=1):
        circles.append(
            CircleDetection(
                circle_id=circle_id,
                x=int(round(keypoint.pt[0])),
                y=int(round(keypoint.pt[1])),
                radius=max(1, int(round(keypoint.size / 2))),
            )
        )
    return circles


def _build_contour_mask(
    image: np.ndarray,
    plate_mask: np.ndarray,
    settings: ColonySettings,
) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    lightness = lab[:, :, 0]
    warm_tone = lab[:, :, 2]

    mask = np.zeros(lightness.shape, dtype=np.uint8)
    mask[
        (lightness >= settings.lightness_threshold)
        & (warm_tone >= settings.warm_tone_threshold)
        & (plate_mask > 0)
    ] = 255

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9)),
    )
    return mask


def _compute_area_threshold(
    contour_areas: list[float],
    settings: ColonySettings,
) -> tuple[float, float]:
    if not contour_areas:
        return float("inf"), 0.0

    area_array = np.array(contour_areas, dtype=np.float32)
    q1, median, q3 = np.percentile(area_array, [25, 50, 75])
    iqr = q3 - q1
    mad = np.median(np.abs(area_array - median))
    robust_sigma = 1.4826 * mad

    # Human-review threshold strategy:
    # Treat the median contour area as the typical "single colony" size,
    # then push unusually large contours to human review instead of
    # silently auto-counting a merged region as one colony.
    multiplier_threshold = median * settings.area_multiplier
    iqr_threshold = q3 + (1.5 * iqr) if iqr > 0 else multiplier_threshold
    mad_threshold = (
        median + (3.0 * robust_sigma)
        if robust_sigma > 0
        else multiplier_threshold
    )

    return float(max(multiplier_threshold, iqr_threshold, mad_threshold)), float(median)


def _find_circle_ids_in_contour(
    contour: np.ndarray,
    circles: list[CircleDetection],
) -> list[int]:
    contained: list[int] = []
    for circle in circles:
        inside = cv2.pointPolygonTest(contour, (circle.x, circle.y), False)
        if inside >= 0:
            contained.append(circle.circle_id)
    return contained


def analyze_image_array(
    image: np.ndarray,
    image_name: str = "uploaded_image.png",
    settings: ColonySettings | None = None,
) -> AnalysisResult:
    settings = settings or ColonySettings()
    image_path = Path(image_name)

    if image is None or image.size == 0:
        raise ValueError("Image data is empty.")

    plate_mask, plate_center, plate_radius = _build_plate_mask(image, settings)
    circles = _detect_clear_circles(image, plate_mask, settings)
    contour_mask = _build_contour_mask(image, plate_mask, settings)

    contours, _ = cv2.findContours(
        contour_mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= settings.min_contour_area
    ]

    areas = [float(cv2.contourArea(contour)) for contour in contours]
    review_area_threshold, median_contour_area = _compute_area_threshold(areas, settings)

    review_regions: list[ReviewRegion] = []
    review_circle_ids: set[int] = set()
    ignored_artifact_count = 0

    for contour in contours:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        circularity = (
            0.0
            if perimeter == 0
            else float((4.0 * math.pi * area) / (perimeter * perimeter))
        )

        hull = cv2.convexHull(contour)
        hull_area = float(cv2.contourArea(hull))
        solidity = 0.0 if hull_area == 0 else float(area / hull_area)

        x, y, width, height = cv2.boundingRect(contour)
        aspect_ratio = float(max(width / max(height, 1), height / max(width, 1)))

        moments = cv2.moments(contour)
        if moments["m00"]:
            center_x = moments["m10"] / moments["m00"]
            center_y = moments["m01"] / moments["m00"]
        else:
            center_x = x + (width / 2)
            center_y = y + (height / 2)

        edge_margin = plate_radius - float(
            np.linalg.norm(np.array([center_x, center_y]) - np.array(plate_center))
        )

        circle_ids = _find_circle_ids_in_contour(contour, circles)
        reasons: list[str] = []

        # IMPORTANT DESIGN DECISION FOR THE WORKSHEET:
        # Only auto-count circles that look like clear, separate colonies.
        # If a contour is unusually large, irregular, or likely overlapping,
        # do NOT auto-count that region. Flag it for human review instead so a
        # person can decide whether it is one colony, multiple merged colonies,
        # or a non-colony artifact.
        if area > review_area_threshold:
            reasons.append(
                f"area {area:.1f} px > review threshold {review_area_threshold:.1f} px"
            )
        if area > 2000 and circularity < settings.min_review_circularity:
            reasons.append(
                f"circularity {circularity:.2f} < {settings.min_review_circularity:.2f}"
            )
        if area > 2000 and solidity < settings.min_review_solidity:
            reasons.append(
                f"solidity {solidity:.2f} < {settings.min_review_solidity:.2f}"
            )
        if area > 2000 and aspect_ratio > settings.max_review_aspect_ratio:
            reasons.append(
                f"aspect ratio {aspect_ratio:.2f} > {settings.max_review_aspect_ratio:.2f}"
            )
        if len(circle_ids) > 1:
            reasons.append("contains multiple clear colony centers")

        if not reasons:
            continue

        if not circle_ids and edge_margin < settings.edge_artifact_margin:
            ignored_artifact_count += 1
            continue

        review_id = len(review_regions) + 1
        review_regions.append(
            ReviewRegion(
                review_id=review_id,
                area=area,
                circularity=circularity,
                solidity=solidity,
                aspect_ratio=aspect_ratio,
                edge_margin=edge_margin,
                bbox=(x, y, width, height),
                circle_ids=circle_ids,
                reasons=reasons,
            )
        )
        review_circle_ids.update(circle_ids)

    auto_circle_ids = {
        circle.circle_id for circle in circles if circle.circle_id not in review_circle_ids
    }

    annotated = image.copy()
    cv2.circle(annotated, plate_center, plate_radius, (120, 120, 120), 3)

    draw_count = 1
    for circle in circles:
        if circle.circle_id not in auto_circle_ids:
            continue
        cv2.circle(
            annotated,
            (circle.x, circle.y),
            circle.radius,
            (0, 180, 0),
            3,
        )
        cv2.putText(
            annotated,
            str(draw_count),
            (circle.x - circle.radius, max(32, circle.y - circle.radius - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 180, 0),
            2,
            cv2.LINE_AA,
        )
        draw_count += 1

    for region in review_regions:
        x, y, width, height = region.bbox
        cv2.rectangle(
            annotated,
            (x, y),
            (x + width, y + height),
            (0, 165, 255),
            3,
        )
        cv2.putText(
            annotated,
            f"R{region.review_id}",
            (x, max(36, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 165, 255),
            2,
            cv2.LINE_AA,
        )
        for circle in circles:
            if circle.circle_id not in region.circle_ids:
                continue
            cv2.circle(
                annotated,
                (circle.x, circle.y),
                circle.radius,
                (0, 165, 255),
                3,
            )

    return AnalysisResult(
        image_path=image_path,
        auto_count=len(auto_circle_ids),
        raw_circle_count=len(circles),
        review_count=len(review_regions),
        ignored_artifact_count=ignored_artifact_count,
        review_area_threshold=review_area_threshold,
        median_contour_area=median_contour_area,
        plate_center=plate_center,
        plate_radius=plate_radius,
        circles=circles,
        review_regions=review_regions,
        annotated_image=annotated,
        mask_image=contour_mask,
    )


def analyze_image(
    image_path: str | Path,
    settings: ColonySettings | None = None,
) -> AnalysisResult:
    image_path = Path(image_path)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    return analyze_image_array(image=image, image_name=image_path.name, settings=settings)


def save_analysis_outputs(result: AnalysisResult, output_dir: str | Path) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = result.image_path.stem

    annotated_path = output_dir / f"{stem}_annotated.png"
    mask_path = output_dir / f"{stem}_mask.png"
    review_csv_path = output_dir / f"{stem}_review.csv"
    notes_path = output_dir / f"{stem}_notes.txt"

    cv2.imwrite(str(annotated_path), result.annotated_image)
    cv2.imwrite(str(mask_path), result.mask_image)

    with review_csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "review_id",
                "circle_ids",
                "area_px",
                "circularity",
                "solidity",
                "aspect_ratio",
                "edge_margin_px",
                "reasons",
            ]
        )
        for region in result.review_regions:
            writer.writerow(
                [
                    region.review_id,
                    "|".join(str(circle_id) for circle_id in region.circle_ids),
                    f"{region.area:.1f}",
                    f"{region.circularity:.2f}",
                    f"{region.solidity:.2f}",
                    f"{region.aspect_ratio:.2f}",
                    f"{region.edge_margin:.1f}",
                    " | ".join(region.reasons),
                ]
            )

    notes_path.write_text(
        "\n".join(
            [
                f"Auto-counted colonies: {result.auto_count}",
                f"Review regions: {result.review_count}",
                f"Raw clear-circle detections: {result.raw_circle_count}",
                f"Ignored edge artifacts: {result.ignored_artifact_count}",
                f"Median contour area: {result.median_contour_area:.1f} px",
                f"Review area threshold: {result.review_area_threshold:.1f} px",
            ]
        ),
        encoding="utf-8",
    )
