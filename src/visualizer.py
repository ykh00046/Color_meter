"""
Inspection Visualizer

Provides visualization functions for inspection results including zone overlay,
delta-E heatmap, comparison charts, and batch processing dashboard.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg


@dataclass
class VisualizerConfig:
    """Visualizer configuration"""

    # Zone overlay
    zone_line_thickness: int = 2
    zone_color_ok: Tuple[int, int, int] = (0, 255, 0)  # BGR: Green
    zone_color_ng: Tuple[int, int, int] = (0, 0, 255)  # BGR: Red
    zone_label_font_scale: float = 0.6
    zone_label_thickness: int = 2
    show_zone_labels: bool = True
    show_lens_circle: bool = True
    show_center_mark: bool = True

    # Heatmap
    heatmap_colormap: str = "RdYlGn_r"  # Red=high, Green=low
    heatmap_resolution: int = 360  # Angular resolution
    show_colorbar: bool = True
    delta_e_range: Tuple[float, float] = (0.0, 10.0)

    # Comparison chart
    comparison_figure_size: Tuple[int, int] = (12, 6)
    comparison_dpi: int = 100
    show_threshold_line: bool = True
    show_pass_fail_zones: bool = True

    # Dashboard
    dashboard_figure_size: Tuple[int, int] = (14, 10)
    dashboard_dpi: int = 100

    # Output
    output_format: str = "png"  # "png", "pdf", "both"
    output_quality: int = 95  # PNG compression (0-9, lower=better quality)


class VisualizationError(Exception):
    """Base exception for visualization errors"""

    pass


class InspectionVisualizer:
    """
    Inspection result visualizer

    Provides methods to visualize inspection results including zone overlay,
    delta-E heatmap, comparison charts, and batch processing dashboard.
    """

    def __init__(self, config: Optional[VisualizerConfig] = None):
        """
        Initialize visualizer

        Args:
            config: Visualizer configuration
        """
        self.config = config or VisualizerConfig()

    def visualize_zone_overlay(
        self,
        image: np.ndarray,
        lens_detection: Any,  # LensDetection
        zones: List[Any],  # List[Zone]
        inspection_result: Any,  # InspectionResult
        show_result: bool = True,
    ) -> np.ndarray:
        """
        Visualize zone overlay on image

        Args:
            image: Input image (BGR, np.ndarray)
            lens_detection: Lens detection result
            zones: List of zones
            inspection_result: Inspection result
            show_result: Show overall judgment on image

        Returns:
            Overlaid image (BGR, np.ndarray)
        """
        # Copy image to avoid modifying original
        overlay = image.copy()

        # Draw lens circle
        if self.config.show_lens_circle:
            cv2.circle(
                overlay,
                (int(lens_detection.center_x), int(lens_detection.center_y)),
                int(lens_detection.radius),
                (128, 128, 128),  # Gray
                1,
                cv2.LINE_AA,
            )

        # Draw center mark
        if self.config.show_center_mark:
            center = (int(lens_detection.center_x), int(lens_detection.center_y))
            cv2.drawMarker(overlay, center, (0, 0, 255), cv2.MARKER_CROSS, 10, 2)  # Red

        # Draw zones
        for zone, zone_result in zip(zones, inspection_result.zone_results):
            color = self.config.zone_color_ok if zone_result.is_ok else self.config.zone_color_ng
            thickness = self.config.zone_line_thickness if zone_result.is_ok else self.config.zone_line_thickness + 1

            # Draw zone boundaries (arcs)
            self._draw_zone_boundary(overlay, lens_detection, zone, color, thickness)

            # Draw zone label
            if self.config.show_zone_labels:
                self._draw_zone_label(overlay, lens_detection, zone, zone_result, color)

        # Draw overall judgment
        if show_result:
            self._draw_judgment_banner(overlay, inspection_result)

        return overlay

    def _draw_zone_boundary(
        self, image: np.ndarray, lens_detection: Any, zone: Any, color: Tuple[int, int, int], thickness: int
    ):
        """Draw zone boundary on image"""
        center_x = int(lens_detection.center_x)
        center_y = int(lens_detection.center_y)
        radius = int(lens_detection.radius)

        # Inner circle
        r_inner = int(zone.r_start * radius)
        if r_inner > 0:
            cv2.circle(image, (center_x, center_y), r_inner, color, thickness, cv2.LINE_AA)

        # Outer circle
        r_outer = int(zone.r_end * radius)
        cv2.circle(image, (center_x, center_y), r_outer, color, thickness, cv2.LINE_AA)

    def _draw_zone_label(
        self, image: np.ndarray, lens_detection: Any, zone: Any, zone_result: Any, color: Tuple[int, int, int]
    ):
        """Draw zone label on image"""
        center_x = int(lens_detection.center_x)
        center_y = int(lens_detection.center_y)
        radius = int(lens_detection.radius)

        # Label position: middle of zone
        r_mid = (zone.r_start + zone.r_end) / 2 * radius
        label_x = int(center_x + r_mid * 0.7)  # Offset to avoid center
        label_y = int(center_y)

        # Label text
        status = "OK" if zone_result.is_ok else "NG"
        label_text = f"{zone_result.zone_name}: dE={zone_result.delta_e:.2f} [{status}]"

        # Text size
        font = cv2.FONT_HERSHEY_SIMPLEX
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, font, self.config.zone_label_font_scale, self.config.zone_label_thickness
        )

        # Draw background rectangle
        bg_color = (255, 255, 255) if zone_result.is_ok else (200, 200, 255)
        cv2.rectangle(
            image, (label_x - 5, label_y - text_h - 5), (label_x + text_w + 5, label_y + baseline + 5), bg_color, -1
        )

        # Draw text
        text_color = (0, 0, 0)  # Black
        cv2.putText(
            image,
            label_text,
            (label_x, label_y),
            font,
            self.config.zone_label_font_scale,
            text_color,
            self.config.zone_label_thickness,
            cv2.LINE_AA,
        )

    def _draw_judgment_banner(self, image: np.ndarray, inspection_result: Any):
        """Draw overall judgment banner"""
        h, w = image.shape[:2]

        # Banner text
        judgment = inspection_result.judgment
        delta_e = inspection_result.overall_delta_e
        banner_text = f"[{judgment}] dE={delta_e:.2f}"

        # Banner position
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        (text_w, text_h), baseline = cv2.getTextSize(banner_text, font, font_scale, thickness)

        # Background
        bg_color = (0, 200, 0) if judgment == "OK" else (0, 0, 200)  # Green or Red
        cv2.rectangle(image, (10, 10), (30 + text_w, 30 + text_h), bg_color, -1)

        # Text
        cv2.putText(
            image, banner_text, (20, 20 + text_h), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA  # White
        )

    def visualize_comparison(self, zones: List[Any], inspection_result: Any) -> plt.Figure:
        """
        Visualize comparison chart (measured vs target LAB, delta-E vs threshold)

        Args:
            zones: List of zones
            inspection_result: Inspection result

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(1, 2, figsize=self.config.comparison_figure_size, dpi=self.config.comparison_dpi)

        # Extract data
        zone_names = [zr.zone_name for zr in inspection_result.zone_results]
        measured_L = [zr.measured_lab[0] for zr in inspection_result.zone_results]
        measured_a = [zr.measured_lab[1] for zr in inspection_result.zone_results]
        measured_b = [zr.measured_lab[2] for zr in inspection_result.zone_results]
        target_L = [zr.target_lab[0] for zr in inspection_result.zone_results]
        target_a = [zr.target_lab[1] for zr in inspection_result.zone_results]
        target_b = [zr.target_lab[2] for zr in inspection_result.zone_results]
        delta_es = [zr.delta_e for zr in inspection_result.zone_results]
        thresholds = [zr.threshold for zr in inspection_result.zone_results]

        # Plot 1: LAB comparison (grouped bar chart)
        ax = axes[0]
        x = np.arange(len(zone_names))
        width = 0.15

        ax.bar(x - width * 2, measured_L, width, label="Measured L*", color="steelblue", alpha=0.8)
        ax.bar(x - width, target_L, width, label="Target L*", color="lightblue", alpha=0.6)
        ax.bar(x, measured_a, width, label="Measured a*", color="indianred", alpha=0.8)
        ax.bar(x + width, target_a, width, label="Target a*", color="lightcoral", alpha=0.6)
        ax.bar(x + width * 2, measured_b, width, label="Measured b*", color="gold", alpha=0.8)
        ax.bar(x + width * 3, target_b, width, label="Target b*", color="khaki", alpha=0.6)

        ax.set_xlabel("Zone")
        ax.set_ylabel("LAB Value")
        ax.set_title("Measured vs Target LAB Values")
        ax.set_xticks(x)
        ax.set_xticklabels(zone_names)
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, axis="y", alpha=0.3)

        # Plot 2: Delta-E vs Threshold
        ax = axes[1]
        ax.plot(zone_names, delta_es, "o-", label="Measured ΔE", color="steelblue", linewidth=2, markersize=8)
        ax.plot(zone_names, thresholds, "--", label="Threshold", color="red", linewidth=2)

        # Fill pass/fail zones
        if self.config.show_pass_fail_zones:
            ax.fill_between(range(len(zone_names)), 0, thresholds, alpha=0.2, color="green", label="Pass Zone")
            ax.fill_between(
                range(len(zone_names)),
                thresholds,
                [max(delta_es) * 1.2] * len(zone_names),
                alpha=0.2,
                color="red",
                label="Fail Zone",
            )

        ax.set_xlabel("Zone")
        ax.set_ylabel("ΔE")
        ax.set_title("ΔE vs Threshold")
        ax.set_xticks(range(len(zone_names)))
        ax.set_xticklabels(zone_names)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def visualize_dashboard(self, results: List[Any]) -> plt.Figure:  # List[InspectionResult]
        """
        Visualize batch processing dashboard

        Args:
            results: List of inspection results

        Returns:
            matplotlib Figure
        """
        fig, axes = plt.subplots(2, 2, figsize=self.config.dashboard_figure_size, dpi=self.config.dashboard_dpi)

        # Extract data
        judgments = [r.judgment for r in results]
        delta_es = [r.overall_delta_e for r in results]
        skus = [r.sku for r in results]

        # Plot 1: Judgment distribution (pie chart)
        ax = axes[0, 0]
        judgment_counts = pd.Series(judgments).value_counts()
        colors = ["green" if j == "OK" else "red" for j in judgment_counts.index]
        ax.pie(judgment_counts.values, labels=judgment_counts.index, autopct="%1.1f%%", colors=colors, startangle=90)
        ax.set_title(f"Judgment Distribution\n(Total: {len(results)})")

        # Plot 2: Delta-E distribution by SKU (box plot)
        ax = axes[0, 1]
        df = pd.DataFrame({"SKU": skus, "ΔE": delta_es})
        df.boxplot(column="ΔE", by="SKU", ax=ax)
        ax.set_title("ΔE Distribution by SKU")
        ax.set_xlabel("SKU")
        ax.set_ylabel("ΔE")
        plt.sca(ax)
        plt.xticks(rotation=45)

        # Plot 3: Zone NG frequency (heatmap)
        ax = axes[1, 0]
        zone_ng_data = []
        for result in results:
            for zr in result.zone_results:
                if not zr.is_ok:
                    zone_ng_data.append({"SKU": result.sku, "Zone": zr.zone_name})

        if zone_ng_data:
            df_ng = pd.DataFrame(zone_ng_data)
            ng_freq = df_ng.groupby(["SKU", "Zone"]).size().unstack(fill_value=0)
            im = ax.imshow(ng_freq.values, cmap="Reds", aspect="auto")
            ax.set_xticks(range(len(ng_freq.columns)))
            ax.set_xticklabels(ng_freq.columns)
            ax.set_yticks(range(len(ng_freq.index)))
            ax.set_yticklabels(ng_freq.index)
            ax.set_title("Zone NG Frequency")
            ax.set_xlabel("Zone")
            ax.set_ylabel("SKU")
            plt.colorbar(im, ax=ax)
        else:
            ax.text(0.5, 0.5, "No NG zones", ha="center", va="center", transform=ax.transAxes)
            ax.set_title("Zone NG Frequency")

        # Plot 4: Processing summary (bar chart)
        ax = axes[1, 1]
        summary_data = df.groupby("SKU").agg({"ΔE": ["mean", "min", "max", "std"]}).round(2)
        summary_data.columns = ["Mean ΔE", "Min ΔE", "Max ΔE", "Std ΔE"]
        summary_data["Mean ΔE"].plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title("Average ΔE by SKU")
        ax.set_xlabel("SKU")
        ax.set_ylabel("Mean ΔE")
        ax.grid(True, axis="y", alpha=0.3)
        plt.sca(ax)
        plt.xticks(rotation=45)

        plt.tight_layout()
        return fig

    def save_visualization(self, image: Union[np.ndarray, plt.Figure], output_path: Path, format: Optional[str] = None):
        """
        Save visualization to file

        Args:
            image: Image or Figure to save
            output_path: Output file path
            format: Output format ("png", "pdf", None=auto-detect)
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Auto-detect format
        if format is None:
            format = output_path.suffix[1:] if output_path.suffix else self.config.output_format

        if isinstance(image, np.ndarray):
            # Save OpenCV image
            if format.lower() == "png":
                cv2.imwrite(
                    str(output_path), image, [cv2.IMWRITE_PNG_COMPRESSION, 9 - self.config.output_quality // 10]
                )
            elif format.lower() in ["jpg", "jpeg"]:
                cv2.imwrite(str(output_path), image, [cv2.IMWRITE_JPEG_QUALITY, self.config.output_quality])
            else:
                cv2.imwrite(str(output_path), image)

        elif isinstance(image, plt.Figure):
            # Save matplotlib figure
            image.savefig(output_path, format=format, dpi=self.config.comparison_dpi, bbox_inches="tight")
            plt.close(image)

        else:
            raise VisualizationError(f"Unsupported image type: {type(image)}")

    def figure_to_array(self, fig: plt.Figure) -> np.ndarray:
        """
        Convert matplotlib figure to numpy array (BGR)

        Args:
            fig: matplotlib Figure

        Returns:
            Image array (BGR, np.ndarray)
        """
        canvas = FigureCanvasAgg(fig)
        canvas.draw()
        buf = canvas.buffer_rgba()
        image = np.asarray(buf)

        # RGBA to BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)

        plt.close(fig)
        return image
