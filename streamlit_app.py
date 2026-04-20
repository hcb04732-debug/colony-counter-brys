from __future__ import annotations

import csv
import io

import cv2
import numpy as np
import streamlit as st

from colony_counter_core import (
    ColonySettings,
    analyze_image_array,
    review_regions_to_rows,
)


st.set_page_config(
    page_title="Colony Counter",
    layout="wide",
    initial_sidebar_state="expanded",
)


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            background:
                radial-gradient(circle at top left, rgba(244, 224, 182, 0.45), transparent 28%),
                radial-gradient(circle at top right, rgba(116, 166, 127, 0.20), transparent 24%),
                linear-gradient(180deg, #f6f1e7 0%, #f2ecdf 100%);
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 3rem;
        }
        .hero {
            padding: 1.3rem 1.5rem;
            border: 1px solid rgba(44, 75, 63, 0.14);
            border-radius: 20px;
            background: rgba(255, 252, 245, 0.82);
            box-shadow: 0 12px 36px rgba(58, 54, 46, 0.08);
            margin-bottom: 1.25rem;
        }
        .hero h1 {
            margin: 0;
            color: #1f4738;
            letter-spacing: -0.02em;
            font-size: 2.2rem;
        }
        .hero p {
            margin: 0.65rem 0 0 0;
            color: #3e4d46;
            font-size: 1rem;
            line-height: 1.55;
        }
        .callout {
            padding: 1rem 1.1rem;
            border-left: 6px solid #c96f2d;
            border-radius: 14px;
            background: rgba(255, 247, 235, 0.92);
            color: #573a1f;
            margin-bottom: 1rem;
        }
        .metric-card {
            border: 1px solid rgba(44, 75, 63, 0.12);
            border-radius: 18px;
            background: rgba(255, 251, 244, 0.88);
            padding: 1rem 1.05rem;
            box-shadow: 0 10px 28px rgba(58, 54, 46, 0.06);
        }
        .metric-label {
            font-size: 0.82rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: #5d6a63;
            margin-bottom: 0.3rem;
        }
        .metric-value {
            font-size: 2rem;
            line-height: 1;
            color: #183f31;
            font-weight: 700;
        }
        .metric-sub {
            margin-top: 0.45rem;
            font-size: 0.92rem;
            color: #516058;
        }
        .section-title {
            margin-top: 0.25rem;
            margin-bottom: 0.65rem;
            color: #234637;
            font-size: 1.2rem;
            font-weight: 700;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def encode_png(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise ValueError("Could not encode image as PNG.")
    return encoded.tobytes()


def crop_review_region(
    image: np.ndarray,
    bbox: tuple[int, int, int, int],
    padding: int = 70,
) -> np.ndarray:
    x, y, width, height = bbox
    image_height, image_width = image.shape[:2]

    x0 = max(0, x - padding)
    y0 = max(0, y - padding)
    x1 = min(image_width, x + width + padding)
    y1 = min(image_height, y + height + padding)

    return cv2.cvtColor(image[y0:y1, x0:x1], cv2.COLOR_BGR2RGB)


def review_csv_bytes(review_rows: list[dict[str, str | int | float]]) -> bytes:
    handle = io.StringIO()
    if review_rows:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0].keys()))
        writer.writeheader()
        writer.writerows(review_rows)
    else:
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
    return handle.getvalue().encode("utf-8")


def notes_bytes(result, manual_review_total: int, final_count: int) -> bytes:
    text = "\n".join(
        [
            f"Auto-counted colonies: {result.auto_count}",
            f"Flagged for human review: {result.review_count}",
            f"Manual review colonies added: {manual_review_total}",
            f"Final combined count: {final_count}",
            f"Raw clear-circle detections: {result.raw_circle_count}",
            f"Ignored edge artifacts: {result.ignored_artifact_count}",
            f"Median contour area: {result.median_contour_area:.1f} px",
            f"Review area threshold: {result.review_area_threshold:.1f} px",
        ]
    )
    return text.encode("utf-8")


def render_metric_card(label: str, value: str, subtext: str) -> None:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-sub">{subtext}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def slider_descriptor(text: str) -> None:
    with st.expander("See more"):
        st.caption(text)


def manual_count_key(image_name: str, review_id: int) -> str:
    safe_name = image_name.replace(" ", "_")
    return f"manual_review_count::{safe_name}::{review_id}"


def clear_manual_count_state(image_name: str) -> None:
    prefix = f"manual_review_count::{image_name.replace(' ', '_')}::"
    for key in list(st.session_state.keys()):
        if key.startswith(prefix):
            del st.session_state[key]


def build_review_rows_with_manual_counts(result, review_rows):
    rows_with_manual: list[dict[str, str | int | float]] = []
    manual_review_total = 0

    for region, row in zip(result.review_regions, review_rows):
        suggested_count = max(1, len(region.circle_ids))
        manual_count = int(
            st.session_state.get(
                manual_count_key(result.image_path.name, region.review_id),
                suggested_count,
            )
        )
        manual_review_total += manual_count

        row_with_manual = dict(row)
        row_with_manual["manual_count"] = manual_count
        rows_with_manual.append(row_with_manual)

    return rows_with_manual, manual_review_total


def main() -> None:
    inject_styles()

    st.markdown(
        """
        <div class="hero">
            <h1>Colony Counter</h1>
            <p>
                Upload a plate image, count clear isolated colonies automatically,
                and push suspicious regions into a human-review queue instead of
                silently overcounting merged growth.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="callout">
            This app is intentionally conservative. If a contour is unusually large,
            irregular, elongated, or likely overlapping, it is flagged for review
            instead of being auto-counted.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("## Controls")
        uploaded_file = st.file_uploader(
            "Plate image",
            type=["png", "jpg", "jpeg", "webp", "bmp", "tif", "tiff"],
        )

        st.markdown("### Detection settings")
        plate_radius_fraction = st.slider(
            "Plate radius fraction",
            min_value=0.30,
            max_value=0.49,
            value=0.43,
            step=0.005,
        )
        slider_descriptor(
            "Adjusts how much of the image is treated as the plate area for "
            "analysis. Increase it to include more of the plate, or decrease it "
            "to focus more tightly on the center."
        )
        lightness_threshold = st.slider(
            "Lightness threshold",
            min_value=0,
            max_value=255,
            value=180,
            step=1,
        )
        slider_descriptor(
            "Controls how bright a region must be to be considered part of a "
            "possible colony. Higher values are more selective, while lower "
            "values include more faint regions."
        )
        warm_tone_threshold = st.slider(
            "Warm-tone threshold",
            min_value=0,
            max_value=255,
            value=145,
            step=1,
        )
        slider_descriptor(
            "Controls how much warm coloring a region needs before it is "
            "flagged as colony-like. Raise it to focus more on warmer-toned "
            "colonies, or lower it to include less strongly colored regions."
        )
        blob_min_threshold = st.slider(
            "Blob brightness cutoff",
            min_value=80,
            max_value=220,
            value=140,
            step=1,
        )
        slider_descriptor(
            "Sets the brightness cutoff for detected blobs. Lower values allow "
            "dimmer, subtler blobs to be considered, while higher values make "
            "detection more selective."
        )
        min_contour_area = st.slider(
            "Minimum colony area",
            min_value=50,
            max_value=600,
            value=300,
            step=10,
        )
        slider_descriptor(
            "Sets the smallest detected region that can still be treated as a "
            "possible colony. Lower values allow smaller colonies to be counted, "
            "while higher values filter out tiny specks and image noise."
        )
        area_multiplier = st.slider(
            "Area multiplier",
            min_value=1.0,
            max_value=5.0,
            value=1.8,
            step=0.1,
        )
        slider_descriptor(
            "Changes the size range used when deciding whether a detected region "
            "should count as a colony. Higher values allow larger regions, while "
            "lower values keep detection more strict."
        )
        min_review_circularity = st.slider(
            "Review circularity",
            min_value=0.0,
            max_value=1.0,
            value=0.55,
            step=0.05,
        )
        slider_descriptor(
            "Controls how round an object must be before it is accepted "
            "automatically. Higher values favor more circular colonies, while "
            "lower values allow more irregular shapes."
        )
        min_review_solidity = st.slider(
            "Review solidity",
            min_value=0.0,
            max_value=1.0,
            value=0.85,
            step=0.05,
        )
        slider_descriptor(
            "Controls how solid and filled-in a region must be during review. "
            "Higher values favor smoother, less fragmented colonies, while lower "
            "values allow more uneven shapes."
        )
        max_review_aspect_ratio = st.slider(
            "Review aspect ratio",
            min_value=1.0,
            max_value=6.0,
            value=1.7,
            step=0.1,
        )
        slider_descriptor(
            "Controls how stretched a region can be before it is flagged for "
            "review. Lower values favor more evenly shaped colonies, while "
            "higher values allow more elongated ones."
        )
        analyze_clicked = st.button("Analyze plate", type="primary", use_container_width=True)

    if not uploaded_file:
        st.info("Upload a plate image to begin.")
        return

    image_bytes = uploaded_file.getvalue()
    decoded = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        st.error("The uploaded file could not be read as an image.")
        return

    original_rgb = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
    st.markdown('<div class="section-title">Original Plate</div>', unsafe_allow_html=True)
    st.image(original_rgb, use_container_width=True)

    if not analyze_clicked and "last_result" not in st.session_state:
        st.info("Adjust the thresholds if needed, then click `Analyze plate`.")
        return

    if analyze_clicked or st.session_state.get("last_uploaded_name") != uploaded_file.name:
        clear_manual_count_state(uploaded_file.name)
        settings = ColonySettings(
            plate_radius_fraction=plate_radius_fraction,
            lightness_threshold=lightness_threshold,
            warm_tone_threshold=warm_tone_threshold,
            blob_min_threshold=blob_min_threshold,
            min_contour_area=min_contour_area,
            area_multiplier=area_multiplier,
            min_review_circularity=min_review_circularity,
            min_review_solidity=min_review_solidity,
            max_review_aspect_ratio=max_review_aspect_ratio,
        )
        with st.spinner("Analyzing colony image..."):
            result = analyze_image_array(
                image=decoded,
                image_name=uploaded_file.name,
                settings=settings,
            )
        st.session_state["last_result"] = result
        st.session_state["last_uploaded_name"] = uploaded_file.name
    else:
        result = st.session_state["last_result"]

    review_rows = review_regions_to_rows(result)
    review_rows_with_manual, manual_review_total = build_review_rows_with_manual_counts(
        result,
        review_rows,
    )
    final_count = result.auto_count + manual_review_total

    st.markdown('<div class="section-title">Summary</div>', unsafe_allow_html=True)
    metric_cols = st.columns(5)
    with metric_cols[0]:
        render_metric_card(
            "Auto Count",
            str(result.auto_count),
            "Clear, separate colonies",
        )
    with metric_cols[1]:
        render_metric_card(
            "Review Queue",
            str(result.review_count),
            "Suspicious contours awaiting a person",
        )
    with metric_cols[2]:
        render_metric_card(
            "Final Count",
            str(final_count),
            "Auto-count plus manual review entries",
        )
    with metric_cols[3]:
        render_metric_card(
            "Raw Detections",
            str(result.raw_circle_count),
            "Bright circular colony candidates",
        )
    with metric_cols[4]:
        render_metric_card(
            "Area Threshold",
            f"{result.review_area_threshold:.0f} px",
            f"Median contour area: {result.median_contour_area:.0f} px",
        )

    image_tab, mask_tab, review_tab, download_tab = st.tabs(
        ["Annotated Result", "Contour Mask", "Review Queue", "Downloads"]
    )

    with image_tab:
        st.image(
            cv2.cvtColor(result.annotated_image, cv2.COLOR_BGR2RGB),
            use_container_width=True,
        )

    with mask_tab:
        st.image(result.mask_image, use_container_width=True, clamp=True)

    with review_tab:
        if review_rows:
            st.dataframe(review_rows_with_manual, use_container_width=True, hide_index=True)
            st.markdown(
                '<div class="section-title">Review Region Cards</div>',
                unsafe_allow_html=True,
            )
            st.caption(
                "Open a highlighted region, inspect the crop, and enter how many "
                "colonies are actually present there. The final count updates "
                "automatically."
            )

            for region in result.review_regions:
                suggested_count = max(1, len(region.circle_ids))
                input_key = manual_count_key(result.image_path.name, region.review_id)
                current_count = int(st.session_state.get(input_key, suggested_count))
                expander_title = f"R{region.review_id} · current count: {current_count}"

                with st.expander(expander_title):
                    card_cols = st.columns([1.05, 1.35])
                    with card_cols[0]:
                        st.image(
                            crop_review_region(decoded, region.bbox),
                            caption=f"Review crop for R{region.review_id}",
                            use_container_width=True,
                        )
                    with card_cols[1]:
                        st.number_input(
                            f"Colonies in R{region.review_id}",
                            min_value=0,
                            step=1,
                            value=suggested_count,
                            key=input_key,
                        )

                        metric_cols = st.columns(3)
                        with metric_cols[0]:
                            st.metric("Suggested", suggested_count)
                        with metric_cols[1]:
                            st.metric("Area", f"{region.area:.0f} px")
                        with metric_cols[2]:
                            st.metric("Aspect", f"{region.aspect_ratio:.2f}")

                        if region.circle_ids:
                            st.caption(
                                "Detected centers: "
                                + ", ".join(str(circle_id) for circle_id in region.circle_ids)
                            )
                        else:
                            st.caption("Detected centers: none")

                        st.caption("Reasons: " + " | ".join(region.reasons))

            st.success(f"Final combined count: {final_count}")
        else:
            st.success("No suspicious regions were flagged for review.")
            st.info(f"Final combined count: {final_count}")

    with download_tab:
        annotated_png = encode_png(result.annotated_image)
        mask_png = encode_png(result.mask_image)
        csv_bytes = review_csv_bytes(review_rows_with_manual)
        txt_bytes = notes_bytes(result, manual_review_total, final_count)
        download_cols = st.columns(4)
        with download_cols[0]:
            st.download_button(
                "Download annotated PNG",
                data=annotated_png,
                file_name=f"{result.image_path.stem}_annotated.png",
                mime="image/png",
                use_container_width=True,
            )
        with download_cols[1]:
            st.download_button(
                "Download mask PNG",
                data=mask_png,
                file_name=f"{result.image_path.stem}_mask.png",
                mime="image/png",
                use_container_width=True,
            )
        with download_cols[2]:
            st.download_button(
                "Download review CSV",
                data=csv_bytes,
                file_name=f"{result.image_path.stem}_review.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with download_cols[3]:
            st.download_button(
                "Download notes TXT",
                data=txt_bytes,
                file_name=f"{result.image_path.stem}_notes.txt",
                mime="text/plain",
                use_container_width=True,
            )


if __name__ == "__main__":
    main()
