"""
Telemetry JSON Exporter

모든 파이프라인 정보를 AI 분석 가능한 완전한 JSON으로 출력.
"""

import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from src.schemas.inspection import InspectionResult, Zone


class TelemetryExporter:
    """
    완전한 텔레메트리 JSON 생성기

    AI 학습/분석을 위해 파이프라인의 모든 정보를 JSON으로 출력.
    """

    def __init__(
        self,
        include_images: bool = True,
        include_radial_profile: bool = True,
        include_processing_times: bool = True,
        include_config_snapshot: bool = True,
    ):
        """
        Args:
            include_images: 이미지 Base64 포함 여부
            include_radial_profile: Radial profile 원본 포함 여부
            include_processing_times: 단계별 처리 시간 포함 여부
            include_config_snapshot: 설정값 스냅샷 포함 여부
        """
        self.include_images = include_images
        self.include_radial_profile = include_radial_profile
        self.include_processing_times = include_processing_times
        self.include_config_snapshot = include_config_snapshot

    def export_full_telemetry(
        self,
        inspection_result: InspectionResult,
        image: Optional[np.ndarray] = None,
        radial_profile: Optional[Any] = None,
        lens_detection: Optional[Any] = None,
        zones: Optional[List[Zone]] = None,
        ring_sector_cells: Optional[List[Any]] = None,
        uniformity_analysis: Optional[Dict[str, Any]] = None,
        boundary_detection: Optional[Dict[str, Any]] = None,
        background_mask_stats: Optional[Dict[str, Any]] = None,
        processing_times: Optional[Dict[str, float]] = None,
        config_snapshot: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        완전한 텔레메트리 JSON 생성

        Args:
            inspection_result: 최종 검사 결과
            image: 원본 이미지
            radial_profile: Radial profile 원본
            lens_detection: 렌즈 검출 정보
            zones: Zone 분할 결과
            ring_sector_cells: Ring×Sector 셀 리스트
            uniformity_analysis: 균일성 분석 결과
            boundary_detection: 경계 검출 상세 정보
            background_mask_stats: 배경 마스크 통계
            processing_times: 단계별 처리 시간 (ms)
            config_snapshot: 설정값 스냅샷
            metadata: 추가 메타데이터

        Returns:
            완전한 텔레메트리 JSON dict
        """
        telemetry = {
            "version": "1.0.0",
            "timestamp": datetime.now().isoformat(),
            "metadata": self._export_metadata(metadata or {}),
            # 1. 기본 판정 정보
            "inspection": {
                "sku": inspection_result.sku,
                "judgment": inspection_result.judgment,
                "overall_delta_e": float(inspection_result.overall_delta_e),
                "confidence": float(inspection_result.confidence),
                "ng_reasons": inspection_result.ng_reasons,
                "zone_count": len(inspection_result.zone_results),
            },
            # 2. Zone별 평가 결과
            "zone_results": [
                {
                    "zone_name": zr.zone_name,
                    "measured_lab": {
                        "L": float(zr.measured_lab[0]),
                        "a": float(zr.measured_lab[1]),
                        "b": float(zr.measured_lab[2]),
                    },
                    "target_lab": {
                        "L": float(zr.target_lab[0]),
                        "a": float(zr.target_lab[1]),
                        "b": float(zr.target_lab[2]),
                    },
                    "delta_e": float(zr.delta_e),
                    "threshold": float(zr.threshold),
                    "is_ok": zr.is_ok,
                }
                for zr in inspection_result.zone_results
            ],
            # 3. 렌즈 검출 정보 (상세)
            "lens_detection": self._export_lens_detection(lens_detection) if lens_detection else None,
            # 4. Radial Profile 원본 (AI 학습용 핵심!)
            "radial_profile": (
                self._export_radial_profile(radial_profile) if radial_profile and self.include_radial_profile else None
            ),
            # 5. Zone 분할 결과 (상세)
            "zones": self._export_zones(zones) if zones else None,
            # 6. Boundary Detection 상세
            "boundary_detection": boundary_detection,
            # 7. Background Mask 통계
            "background_mask": background_mask_stats,
            # 8. Ring×Sector 2D 분석 (36개 셀)
            "ring_sector_cells": self._export_ring_sector_cells(ring_sector_cells) if ring_sector_cells else None,
            # 9. 균일성 분석
            "uniformity_analysis": uniformity_analysis,
            # 10. 이미지 (Base64)
            "images": self._export_images(image) if self.include_images and image is not None else None,
            # 11. 처리 시간 (단계별)
            "processing_times": processing_times if self.include_processing_times else None,
            # 12. 설정값 스냅샷 (재현성)
            "config_snapshot": config_snapshot if self.include_config_snapshot else None,
        }

        return telemetry

    def _export_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """메타데이터 추출"""
        return {
            "image_filename": metadata.get("image_filename", "unknown"),
            "image_width": metadata.get("image_width", 0),
            "image_height": metadata.get("image_height", 0),
            "image_format": metadata.get("image_format", "unknown"),
            "sku_code": metadata.get("sku_code", ""),
            "inspection_id": metadata.get("inspection_id", ""),
        }

    def _export_lens_detection(self, lens: Any) -> Dict[str, Any]:
        """렌즈 검출 정보 (상세)"""
        return {
            "center_x": float(lens.center_x),
            "center_y": float(lens.center_y),
            "radius": float(lens.radius),
            "confidence": float(lens.confidence),
            "method": lens.method,
            "roi": lens.roi.tolist() if isinstance(lens.roi, np.ndarray) else lens.roi,
        }

    def _export_radial_profile(self, profile: Any) -> Dict[str, Any]:
        """
        Radial Profile 원본 (AI 학습용 핵심!)

        이 데이터로 가능한 AI 분석:
        - 색상 변화 패턴 학습
        - 불량 패턴 인식
        - 자동 threshold 학습
        - 이상치 탐지
        """
        return {
            "r_normalized": profile.r_normalized.tolist(),
            "L": profile.L.tolist(),
            "a": profile.a.tolist(),
            "b": profile.b.tolist(),
            "std_L": profile.std_L.tolist(),
            "std_a": profile.std_a.tolist(),
            "std_b": profile.std_b.tolist(),
            "pixel_count": profile.pixel_count.tolist(),
            "length": len(profile.r_normalized),
            # 통계
            "statistics": {
                "L_mean": float(np.mean(profile.L)),
                "L_std": float(np.std(profile.L)),
                "L_min": float(np.min(profile.L)),
                "L_max": float(np.max(profile.L)),
                "a_mean": float(np.mean(profile.a)),
                "a_std": float(np.std(profile.a)),
                "b_mean": float(np.mean(profile.b)),
                "b_std": float(np.std(profile.b)),
            },
        }

    def _export_zones(self, zones: List[Zone]) -> List[Dict[str, Any]]:
        """Zone 분할 결과 (상세)"""
        return [
            {
                "name": zone.name,
                "r_start": float(zone.r_start),
                "r_end": float(zone.r_end),
                "mean_L": float(zone.mean_L),
                "mean_a": float(zone.mean_a),
                "mean_b": float(zone.mean_b),
                "std_L": float(zone.std_L),
                "std_a": float(zone.std_a),
                "std_b": float(zone.std_b),
                "zone_type": zone.zone_type,
            }
            for zone in zones
        ]

    def _export_ring_sector_cells(self, cells: List[Any]) -> List[Dict[str, Any]]:
        """Ring×Sector 36개 셀"""
        return [
            {
                "ring_index": cell.ring_index,
                "sector_index": cell.sector_index,
                "r_start": float(cell.r_start),
                "r_end": float(cell.r_end),
                "angle_start": float(cell.angle_start),
                "angle_end": float(cell.angle_end),
                "mean_L": float(cell.mean_L),
                "mean_a": float(cell.mean_a),
                "mean_b": float(cell.mean_b),
                "std_L": float(cell.std_L),
                "std_a": float(cell.std_a),
                "std_b": float(cell.std_b),
                "pixel_count": int(cell.pixel_count),
            }
            for cell in cells
        ]

    def _export_images(self, image: np.ndarray) -> Dict[str, Any]:
        """이미지 Base64 인코딩"""
        # JPEG로 인코딩
        success, buffer = cv2.imencode(".jpg", image)
        if not success:
            return None

        # Base64 인코딩
        jpg_base64 = base64.b64encode(buffer).decode("utf-8")

        return {
            "original": {
                "format": "jpeg",
                "encoding": "base64",
                "data": jpg_base64,
                "width": image.shape[1],
                "height": image.shape[0],
                "channels": image.shape[2] if len(image.shape) == 3 else 1,
            }
        }

    def save_json(self, telemetry: Dict[str, Any], output_path: Path):
        """JSON 파일 저장"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(telemetry, f, indent=2, ensure_ascii=False)

    def save_compact_json(self, telemetry: Dict[str, Any], output_path: Path):
        """Compact JSON 저장 (공백 최소화)"""
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(telemetry, f, ensure_ascii=False, separators=(",", ":"))
