#!/usr/bin/env python3
"""
Alignment Algorithm Benchmark Script

알고리즘 벤치마크: DTW vs FastDTW vs Cross-Correlation
- 성능 측정 (처리 시간)
- 정확도 측정 (정렬 품질)
- 목표: avg < 1.0s, p99 < 3.0s
"""

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


@dataclass
class BenchmarkResult:
    """벤치마크 결과"""

    algorithm: str
    n_samples: int
    avg_time_ms: float
    median_time_ms: float
    p95_time_ms: float
    p99_time_ms: float
    max_time_ms: float
    avg_correlation: float
    avg_rmse: float
    success_rate: float


class SyntheticProfileGenerator:
    """합성 Radial Profile 생성기"""

    def __init__(self, n_points: int = 500, random_seed: int = 42):
        self.n_points = n_points
        np.random.seed(random_seed)

    def generate_std_profile(self) -> np.ndarray:
        """STD 프로파일 생성 (3-zone 구조)"""
        r = np.linspace(0, 1, self.n_points)

        # Zone A (inner): L=85, smooth
        zone_A = 85 + 3 * np.sin(r * 10 * np.pi)

        # Zone B (middle): L=65, with transition
        zone_B = 65 + 2 * np.sin(r * 15 * np.pi)

        # Zone C (outer): L=75, smooth
        zone_C = 75 + 1.5 * np.sin(r * 8 * np.pi)

        # Combine zones with smooth transitions
        profile = np.zeros(self.n_points)
        for i, r_val in enumerate(r):
            if r_val < 0.3:
                profile[i] = zone_A[i]
            elif r_val < 0.7:
                # Smooth transition from A to B
                alpha = (r_val - 0.3) / 0.4
                profile[i] = (1 - alpha) * zone_A[i] + alpha * zone_B[i]
            else:
                # Smooth transition from B to C
                alpha = (r_val - 0.7) / 0.3
                profile[i] = (1 - alpha) * zone_B[i] + alpha * zone_C[i]

        # Add noise
        profile += np.random.normal(0, 0.5, self.n_points)

        return profile

    def apply_shift(self, profile: np.ndarray, shift_percent: float) -> np.ndarray:
        """프로파일에 shift 적용 (±10% 범위)"""
        shift_pixels = int(self.n_points * shift_percent / 100)
        shifted = np.roll(profile, shift_pixels)
        return shifted

    def apply_noise(self, profile: np.ndarray, noise_level: float = 1.0) -> np.ndarray:
        """노이즈 추가"""
        noisy = profile + np.random.normal(0, noise_level, len(profile))
        return noisy

    def apply_stretch(self, profile: np.ndarray, stretch_factor: float = 1.05) -> np.ndarray:
        """프로파일 늘리기/줄이기 (±5% 범위)"""
        stretched_length = int(len(profile) * stretch_factor)
        stretched = np.interp(np.linspace(0, len(profile) - 1, stretched_length), np.arange(len(profile)), profile)
        # Resize back to original length
        resized = np.interp(np.linspace(0, len(stretched) - 1, len(profile)), np.arange(len(stretched)), stretched)
        return resized

    def generate_test_profiles(self, std_profile: np.ndarray, n_samples: int = 100) -> List[np.ndarray]:
        """테스트 프로파일 생성 (다양한 변형)"""
        test_profiles = []

        for i in range(n_samples):
            test = std_profile.copy()

            # Random transformations
            # 1. Shift (-5% to +5%)
            shift = np.random.uniform(-5, 5)
            test = self.apply_shift(test, shift)

            # 2. Noise (0.5 to 2.0)
            noise_level = np.random.uniform(0.5, 2.0)
            test = self.apply_noise(test, noise_level)

            # 3. Stretch (0.98 to 1.02)
            stretch = np.random.uniform(0.98, 1.02)
            test = self.apply_stretch(test, stretch)

            test_profiles.append(test)

        return test_profiles


class AlignmentAlgorithms:
    """정렬 알고리즘 모음"""

    @staticmethod
    def dtw_full(test: np.ndarray, std: np.ndarray) -> Tuple[float, np.ndarray]:
        """Full DTW (O(n²))"""
        try:
            from dtaidistance import dtw

            start = time.perf_counter()
            distance = dtw.distance(test, std)
            elapsed = (time.perf_counter() - start) * 1000
            return elapsed, None
        except ImportError:
            print("Warning: dtaidistance not installed. Skipping DTW.")
            return -1.0, None

    @staticmethod
    def dtw_fast(test: np.ndarray, std: np.ndarray, window: int = 50) -> Tuple[float, np.ndarray]:
        """FastDTW with window constraint (O(n))"""
        try:
            from dtaidistance import dtw

            start = time.perf_counter()
            distance = dtw.distance_fast(test, std, window=window)
            elapsed = (time.perf_counter() - start) * 1000
            return elapsed, None
        except ImportError:
            print("Warning: dtaidistance not installed. Skipping FastDTW.")
            return -1.0, None

    @staticmethod
    def cross_correlation(test: np.ndarray, std: np.ndarray) -> Tuple[float, np.ndarray]:
        """Cross-Correlation"""
        from scipy.signal import correlate

        start = time.perf_counter()

        # Normalize
        test_norm = (test - np.mean(test)) / np.std(test)
        std_norm = (std - np.mean(std)) / np.std(std)

        # Compute correlation
        corr = correlate(test_norm, std_norm, mode="same")
        max_corr_idx = np.argmax(corr)
        shift = max_corr_idx - len(std) // 2

        # Apply shift
        aligned = np.roll(test, -shift)

        elapsed = (time.perf_counter() - start) * 1000
        return elapsed, aligned

    @staticmethod
    def circular_shift(test: np.ndarray, std: np.ndarray, max_shift: int = 50) -> Tuple[float, np.ndarray]:
        """Circular Shift (Brute Force)"""
        start = time.perf_counter()

        best_corr = -1
        best_shift = 0

        for shift in range(-max_shift, max_shift + 1):
            shifted = np.roll(test, shift)
            corr = np.corrcoef(shifted, std)[0, 1]
            if corr > best_corr:
                best_corr = corr
                best_shift = shift

        aligned = np.roll(test, best_shift)

        elapsed = (time.perf_counter() - start) * 1000
        return elapsed, aligned


class AlignmentBenchmark:
    """알고리즘 벤치마크 실행"""

    def __init__(self, n_samples: int = 100, n_points: int = 500):
        self.n_samples = n_samples
        self.n_points = n_points
        self.generator = SyntheticProfileGenerator(n_points=n_points)
        self.algorithms = AlignmentAlgorithms()

    def generate_test_data(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        """테스트 데이터 생성"""
        print(f"Generating test data: {self.n_samples} samples, {self.n_points} points each...")
        std_profile = self.generator.generate_std_profile()
        test_profiles = self.generator.generate_test_profiles(std_profile, self.n_samples)
        print(f"Test data generated: {len(test_profiles)} profiles")
        return std_profile, test_profiles

    def benchmark_algorithm(
        self, algorithm_name: str, algorithm_func, std_profile: np.ndarray, test_profiles: List[np.ndarray]
    ) -> BenchmarkResult:
        """단일 알고리즘 벤치마크"""
        print(f"\n{'='*60}")
        print(f"Benchmarking: {algorithm_name}")
        print(f"{'='*60}")

        times = []
        correlations = []
        rmses = []
        successes = 0

        for i, test_profile in enumerate(test_profiles):
            if (i + 1) % 20 == 0:
                print(f"Progress: {i+1}/{self.n_samples} samples...")

            try:
                # Run algorithm
                elapsed_ms, aligned = algorithm_func(test_profile, std_profile)

                if elapsed_ms < 0:
                    # Algorithm not available
                    return None

                times.append(elapsed_ms)

                # Evaluate alignment quality
                if aligned is not None:
                    corr = np.corrcoef(aligned, std_profile)[0, 1]
                    rmse = np.sqrt(np.mean((aligned - std_profile) ** 2))
                else:
                    # DTW doesn't return aligned profile, just use original
                    corr = np.corrcoef(test_profile, std_profile)[0, 1]
                    rmse = np.sqrt(np.mean((test_profile - std_profile) ** 2))

                correlations.append(corr)
                rmses.append(rmse)

                if corr > 0.85:  # Success threshold
                    successes += 1

            except Exception as e:
                print(f"Error on sample {i}: {e}")
                continue

        if not times:
            return None

        # Calculate statistics
        times = np.array(times)
        correlations = np.array(correlations)
        rmses = np.array(rmses)

        result = BenchmarkResult(
            algorithm=algorithm_name,
            n_samples=len(times),
            avg_time_ms=np.mean(times),
            median_time_ms=np.median(times),
            p95_time_ms=np.percentile(times, 95),
            p99_time_ms=np.percentile(times, 99),
            max_time_ms=np.max(times),
            avg_correlation=np.mean(correlations),
            avg_rmse=np.mean(rmses),
            success_rate=successes / len(times) * 100,
        )

        # Print results
        print(f"\n{algorithm_name} Results:")
        print(f"  Time (ms):")
        print(f"    Avg:    {result.avg_time_ms:>8.2f}")
        print(f"    Median: {result.median_time_ms:>8.2f}")
        print(f"    P95:    {result.p95_time_ms:>8.2f}")
        print(f"    P99:    {result.p99_time_ms:>8.2f}")
        print(f"    Max:    {result.max_time_ms:>8.2f}")
        print(f"  Quality:")
        print(f"    Avg Correlation: {result.avg_correlation:.4f}")
        print(f"    Avg RMSE:        {result.avg_rmse:.4f}")
        print(f"    Success Rate:    {result.success_rate:.1f}%")

        # Check against target
        target_avg = 1000  # 1.0s
        target_p99 = 3000  # 3.0s
        avg_ok = "[OK]" if result.avg_time_ms < target_avg else "[FAIL]"
        p99_ok = "[OK]" if result.p99_time_ms < target_p99 else "[FAIL]"

        print(f"  Target Check:")
        print(f"    Avg < 1.0s:  {avg_ok} ({result.avg_time_ms:.2f} ms)")
        print(f"    P99 < 3.0s:  {p99_ok} ({result.p99_time_ms:.2f} ms)")

        return result

    def run_all_benchmarks(self) -> List[BenchmarkResult]:
        """모든 알고리즘 벤치마크"""
        std_profile, test_profiles = self.generate_test_data()

        algorithms = [
            ("Cross-Correlation", lambda t, s: self.algorithms.cross_correlation(t, s)),
            ("Circular Shift (±50px)", lambda t, s: self.algorithms.circular_shift(t, s, max_shift=50)),
            ("FastDTW (window=50)", lambda t, s: self.algorithms.dtw_fast(t, s, window=50)),
            ("FastDTW (window=20)", lambda t, s: self.algorithms.dtw_fast(t, s, window=20)),
            ("Full DTW", lambda t, s: self.algorithms.dtw_full(t, s)),
        ]

        results = []
        for name, func in algorithms:
            result = self.benchmark_algorithm(name, func, std_profile, test_profiles)
            if result is not None:
                results.append(result)

        return results

    def generate_comparison_report(self, results: List[BenchmarkResult], output_path: Path):
        """비교 보고서 생성"""
        print(f"\n{'='*60}")
        print("COMPARISON REPORT")
        print(f"{'='*60}\n")

        # Summary table
        print(f"{'Algorithm':<30} {'Avg (ms)':>10} {'P99 (ms)':>10} {'Corr':>8} {'Success %':>10}")
        print("-" * 70)
        for result in results:
            print(
                f"{result.algorithm:<30} {result.avg_time_ms:>10.2f} {result.p99_time_ms:>10.2f} "
                f"{result.avg_correlation:>8.4f} {result.success_rate:>10.1f}"
            )

        # Save to JSON
        report_data = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "n_samples": results[0].n_samples if results else 0,
            "results": [
                {
                    "algorithm": r.algorithm,
                    "avg_time_ms": r.avg_time_ms,
                    "median_time_ms": r.median_time_ms,
                    "p95_time_ms": r.p95_time_ms,
                    "p99_time_ms": r.p99_time_ms,
                    "max_time_ms": r.max_time_ms,
                    "avg_correlation": r.avg_correlation,
                    "avg_rmse": r.avg_rmse,
                    "success_rate": r.success_rate,
                }
                for r in results
            ],
        }

        json_path = output_path.parent / (output_path.stem + ".json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)
        print(f"\nReport saved to: {json_path}")

        # Visualizations
        self.plot_comparison(results, output_path)

    def plot_comparison(self, results: List[BenchmarkResult], output_path: Path):
        """비교 차트 생성"""
        if not results:
            return

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("Alignment Algorithm Benchmark Comparison", fontsize=16, fontweight="bold")

        algorithms = [r.algorithm for r in results]
        avg_times = [r.avg_time_ms for r in results]
        p99_times = [r.p99_time_ms for r in results]
        correlations = [r.avg_correlation for r in results]
        success_rates = [r.success_rate for r in results]

        # Plot 1: Average Time
        ax1 = axes[0, 0]
        bars1 = ax1.bar(algorithms, avg_times, color="steelblue")
        ax1.axhline(y=1000, color="red", linestyle="--", label="Target (1000ms)")
        ax1.set_ylabel("Time (ms)")
        ax1.set_title("Average Processing Time")
        ax1.legend()
        ax1.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars1, avg_times):
            ax1.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.1f}", ha="center", va="bottom", fontsize=9
            )

        # Plot 2: P99 Time
        ax2 = axes[0, 1]
        bars2 = ax2.bar(algorithms, p99_times, color="coral")
        ax2.axhline(y=3000, color="red", linestyle="--", label="Target (3000ms)")
        ax2.set_ylabel("Time (ms)")
        ax2.set_title("P99 Processing Time")
        ax2.legend()
        ax2.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars2, p99_times):
            ax2.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.1f}", ha="center", va="bottom", fontsize=9
            )

        # Plot 3: Correlation
        ax3 = axes[1, 0]
        bars3 = ax3.bar(algorithms, correlations, color="seagreen")
        ax3.axhline(y=0.85, color="red", linestyle="--", label="Threshold (0.85)")
        ax3.set_ylabel("Correlation")
        ax3.set_title("Average Correlation (Alignment Quality)")
        ax3.set_ylim(0, 1.0)
        ax3.legend()
        ax3.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars3, correlations):
            ax3.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.3f}", ha="center", va="bottom", fontsize=9
            )

        # Plot 4: Success Rate
        ax4 = axes[1, 1]
        bars4 = ax4.bar(algorithms, success_rates, color="mediumpurple")
        ax4.axhline(y=95, color="red", linestyle="--", label="Target (95%)")
        ax4.set_ylabel("Success Rate (%)")
        ax4.set_title("Success Rate (Corr > 0.85)")
        ax4.set_ylim(0, 100)
        ax4.legend()
        ax4.tick_params(axis="x", rotation=45)
        for bar, val in zip(bars4, success_rates):
            ax4.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{val:.1f}%", ha="center", va="bottom", fontsize=9
            )

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Comparison chart saved to: {output_path}")


def main():
    """메인 실행"""
    print("=" * 60)
    print("ALIGNMENT ALGORITHM BENCHMARK")
    print("=" * 60)
    print("Comparing: DTW, FastDTW, Cross-Correlation, Circular Shift")
    print("Target: avg < 1.0s (1000ms), p99 < 3.0s (3000ms)")
    print("=" * 60)

    # Configuration
    n_samples = 100  # Number of test profiles
    n_points = 500  # Points per profile (radial profile length)

    # Run benchmark
    benchmark = AlignmentBenchmark(n_samples=n_samples, n_points=n_points)
    results = benchmark.run_all_benchmarks()

    # Generate report
    output_dir = project_root / "results"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "alignment_benchmark.png"

    benchmark.generate_comparison_report(results, output_path)

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)

    # Recommendation
    if results:
        best_speed = min(results, key=lambda r: r.avg_time_ms)
        best_quality = max(results, key=lambda r: r.avg_correlation)
        best_balanced = min(results, key=lambda r: r.avg_time_ms / r.avg_correlation)

        print("\nRecommendations:")
        print(f"  Fastest:       {best_speed.algorithm} ({best_speed.avg_time_ms:.2f} ms)")
        print(f"  Best Quality:  {best_quality.algorithm} (corr={best_quality.avg_correlation:.4f})")
        print(f"  Best Balanced: {best_balanced.algorithm}")

        # Check if any meets both targets
        passed = [r for r in results if r.avg_time_ms < 1000 and r.p99_time_ms < 3000]
        if passed:
            print(f"\n[PASS] {len(passed)} algorithm(s) meet performance targets:")
            for r in passed:
                print(f"    - {r.algorithm}")
        else:
            print("\n[FAIL] No algorithm meets both performance targets (avg < 1s, p99 < 3s)")
            print("  Consider:")
            print("    1. Downsample radial profile (500 -> 250 points)")
            print("    2. Use FastDTW with smaller window")
            print("    3. Use Cross-Correlation (fastest, good quality)")


if __name__ == "__main__":
    main()
