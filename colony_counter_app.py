from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QDoubleSpinBox,
    QSplitter,
    QStatusBar,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from colony_counter_core import ColonySettings, AnalysisResult, analyze_image, save_analysis_outputs


def bgr_to_pixmap(image) -> QPixmap:
    if len(image.shape) == 2:
        height, width = image.shape
        bytes_per_line = width
        qimage = QImage(
            image.data,
            width,
            height,
            bytes_per_line,
            QImage.Format_Grayscale8,
        )
        return QPixmap.fromImage(qimage.copy())

    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    height, width, channels = rgb_image.shape
    bytes_per_line = channels * width
    qimage = QImage(
        rgb_image.data,
        width,
        height,
        bytes_per_line,
        QImage.Format_RGB888,
    )
    return QPixmap.fromImage(qimage.copy())


class ImagePreview(QScrollArea):
    def __init__(self, title: str) -> None:
        super().__init__()
        self.setWidgetResizable(True)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(8, 8, 8, 8)

        heading = QLabel(title)
        heading.setStyleSheet("font-weight: 700; font-size: 14px;")
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setMinimumSize(480, 480)
        self.image_label.setStyleSheet(
            "background: #111827; color: #d1d5db; border: 1px solid #374151;"
        )

        layout.addWidget(heading)
        layout.addWidget(self.image_label, 1)
        self.setWidget(container)

    def set_pixmap(self, pixmap: QPixmap | None) -> None:
        if pixmap is None:
            self.image_label.setText("No image loaded")
            self.image_label.setPixmap(QPixmap())
            return

        scaled = pixmap.scaled(
            900,
            900,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.image_label.setPixmap(scaled)


class MainWindow(QMainWindow):
    def __init__(self, startup_image: str | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Colony Counter Review App")
        self.resize(1500, 980)

        self.current_image_path: Path | None = None
        self.current_result: AnalysisResult | None = None

        root = QWidget()
        root_layout = QHBoxLayout(root)
        root_layout.setContentsMargins(10, 10, 10, 10)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_controls_panel())
        splitter.addWidget(self._build_preview_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([420, 1040])
        root_layout.addWidget(splitter)

        self.setCentralWidget(root)
        self.setStatusBar(QStatusBar())
        self.statusBar().showMessage("Open a plate image to begin.")

        if startup_image:
            self.current_image_path = Path(startup_image)
            self.image_path_edit.setText(str(self.current_image_path))
            self.run_analysis()

    def _build_controls_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)

        intro = QTextEdit()
        intro.setReadOnly(True)
        intro.setFixedHeight(110)
        intro.setPlainText(
            "This app intentionally auto-counts only clear, separate colonies. "
            "If a contour is unusually large, irregular, or likely overlapping, "
            "it is flagged for human review instead of being counted automatically."
        )
        layout.addWidget(intro)

        file_box = QGroupBox("Image")
        file_layout = QVBoxLayout(file_box)
        self.image_path_edit = QLineEdit()
        self.image_path_edit.setReadOnly(True)
        browse_button = QPushButton("Choose Image...")
        browse_button.clicked.connect(self.choose_image)
        file_layout.addWidget(self.image_path_edit)
        file_layout.addWidget(browse_button)
        layout.addWidget(file_box)

        settings_box = QGroupBox("Detection Settings")
        settings_form = QFormLayout(settings_box)

        self.plate_radius_spin = QDoubleSpinBox()
        self.plate_radius_spin.setRange(0.30, 0.49)
        self.plate_radius_spin.setSingleStep(0.005)
        self.plate_radius_spin.setDecimals(3)
        self.plate_radius_spin.setValue(0.43)

        self.lightness_spin = QSpinBox()
        self.lightness_spin.setRange(0, 255)
        self.lightness_spin.setValue(180)

        self.warm_tone_spin = QSpinBox()
        self.warm_tone_spin.setRange(0, 255)
        self.warm_tone_spin.setValue(145)

        self.blob_threshold_spin = QSpinBox()
        self.blob_threshold_spin.setRange(80, 220)
        self.blob_threshold_spin.setValue(140)

        self.area_multiplier_spin = QDoubleSpinBox()
        self.area_multiplier_spin.setRange(1.0, 5.0)
        self.area_multiplier_spin.setSingleStep(0.1)
        self.area_multiplier_spin.setValue(1.8)

        self.circularity_spin = QDoubleSpinBox()
        self.circularity_spin.setRange(0.0, 1.0)
        self.circularity_spin.setSingleStep(0.05)
        self.circularity_spin.setValue(0.55)

        self.solidity_spin = QDoubleSpinBox()
        self.solidity_spin.setRange(0.0, 1.0)
        self.solidity_spin.setSingleStep(0.05)
        self.solidity_spin.setValue(0.85)

        self.aspect_ratio_spin = QDoubleSpinBox()
        self.aspect_ratio_spin.setRange(1.0, 6.0)
        self.aspect_ratio_spin.setSingleStep(0.1)
        self.aspect_ratio_spin.setValue(1.7)

        settings_form.addRow("Plate radius fraction", self.plate_radius_spin)
        settings_form.addRow("Lightness threshold", self.lightness_spin)
        settings_form.addRow("Warm-tone threshold", self.warm_tone_spin)
        settings_form.addRow("Blob brightness cutoff", self.blob_threshold_spin)
        settings_form.addRow("Area multiplier", self.area_multiplier_spin)
        settings_form.addRow("Review circularity", self.circularity_spin)
        settings_form.addRow("Review solidity", self.solidity_spin)
        settings_form.addRow("Review aspect ratio", self.aspect_ratio_spin)
        layout.addWidget(settings_box)

        action_row = QHBoxLayout()
        self.run_button = QPushButton("Run Analysis")
        self.run_button.clicked.connect(self.run_analysis)
        self.save_button = QPushButton("Save Outputs...")
        self.save_button.clicked.connect(self.save_outputs)
        self.save_button.setEnabled(False)
        action_row.addWidget(self.run_button)
        action_row.addWidget(self.save_button)
        layout.addLayout(action_row)

        summary_box = QGroupBox("Results")
        summary_layout = QVBoxLayout(summary_box)
        self.summary_label = QLabel("No analysis yet.")
        self.summary_label.setWordWrap(True)
        self.summary_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        summary_layout.addWidget(self.summary_label)
        layout.addWidget(summary_box)

        review_box = QGroupBox("Review Queue")
        review_layout = QVBoxLayout(review_box)
        self.review_table = QTableWidget(0, 7)
        self.review_table.setHorizontalHeaderLabels(
            [
                "Review ID",
                "Circle IDs",
                "Area",
                "Circularity",
                "Solidity",
                "Aspect Ratio",
                "Reasons",
            ]
        )
        self.review_table.horizontalHeader().setStretchLastSection(True)
        review_layout.addWidget(self.review_table)
        layout.addWidget(review_box, 1)

        return panel

    def _build_preview_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setSpacing(12)

        self.annotated_preview = ImagePreview("Annotated Result")
        self.mask_preview = ImagePreview("Contour Mask")

        layout.addWidget(self.annotated_preview, 1)
        layout.addWidget(self.mask_preview, 1)
        return panel

    def build_settings(self) -> ColonySettings:
        return ColonySettings(
            plate_radius_fraction=self.plate_radius_spin.value(),
            lightness_threshold=self.lightness_spin.value(),
            warm_tone_threshold=self.warm_tone_spin.value(),
            blob_min_threshold=self.blob_threshold_spin.value(),
            area_multiplier=self.area_multiplier_spin.value(),
            min_review_circularity=self.circularity_spin.value(),
            min_review_solidity=self.solidity_spin.value(),
            max_review_aspect_ratio=self.aspect_ratio_spin.value(),
        )

    def choose_image(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Choose Plate Image",
            str(Path.home()),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff)",
        )
        if not file_path:
            return

        self.current_image_path = Path(file_path)
        self.image_path_edit.setText(file_path)
        self.run_analysis()

    def run_analysis(self) -> None:
        if not self.current_image_path:
            QMessageBox.information(self, "Choose Image", "Select a plate image first.")
            return

        try:
            self.statusBar().showMessage("Running colony analysis...")
            QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
            self.current_result = analyze_image(
                self.current_image_path,
                self.build_settings(),
            )
            self.update_results_ui()
            self.save_button.setEnabled(True)
            self.statusBar().showMessage("Analysis complete.")
        except Exception as exc:
            QMessageBox.critical(self, "Analysis Failed", str(exc))
            self.statusBar().showMessage("Analysis failed.")
        finally:
            QApplication.restoreOverrideCursor()

    def update_results_ui(self) -> None:
        if not self.current_result:
            return

        result = self.current_result
        self.annotated_preview.set_pixmap(bgr_to_pixmap(result.annotated_image))
        self.mask_preview.set_pixmap(bgr_to_pixmap(result.mask_image))

        self.summary_label.setText(
            "\n".join(
                [
                    f"Auto-counted colonies: {result.auto_count}",
                    f"Flagged for human review: {result.review_count}",
                    f"Raw clear-circle detections: {result.raw_circle_count}",
                    f"Ignored edge artifacts: {result.ignored_artifact_count}",
                    f"Median contour area: {result.median_contour_area:.1f} px",
                    f"Review area threshold: {result.review_area_threshold:.1f} px",
                ]
            )
        )

        self.review_table.setRowCount(len(result.review_regions))
        for row, region in enumerate(result.review_regions):
            values = [
                f"R{region.review_id}",
                ", ".join(str(circle_id) for circle_id in region.circle_ids) or "-",
                f"{region.area:.1f}",
                f"{region.circularity:.2f}",
                f"{region.solidity:.2f}",
                f"{region.aspect_ratio:.2f}",
                " | ".join(region.reasons),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.review_table.setItem(row, column, item)

        self.review_table.resizeColumnsToContents()

    def save_outputs(self) -> None:
        if not self.current_result:
            return

        target_dir = QFileDialog.getExistingDirectory(
            self,
            "Choose Output Folder",
            str(Path(self.current_result.image_path).parent),
        )
        if not target_dir:
            return

        try:
            save_analysis_outputs(self.current_result, target_dir)
            self.statusBar().showMessage(f"Saved outputs to {target_dir}")
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", str(exc))


def main() -> int:
    parser = argparse.ArgumentParser(description="PySide6 colony counter app")
    parser.add_argument(
        "image",
        nargs="?",
        default=None,
        help="Optional image path to load on startup.",
    )
    args = parser.parse_args()

    app = QApplication(sys.argv)
    window = MainWindow(startup_image=args.image)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
