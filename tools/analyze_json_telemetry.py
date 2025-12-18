"""
JSON 텔레메트리 분석 예제 스크립트

현재 results/web/ 폴더의 모든 result.json을 분석하여 AI 학습용 데이터셋 생성.
"""

import glob
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def collect_all_results(results_dir="results/web"):
    """모든 검사 결과 수집"""
    json_files = glob.glob(f"{results_dir}/*/result.json")

    ok_samples = []
    ng_samples = []

    for json_file in json_files:
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            if data["judgment"] == "OK":
                ok_samples.append(data)
            else:
                ng_samples.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    print(f"\n=== Data Collection Summary ===")
    print(f"OK samples: {len(ok_samples)}")
    print(f"NG samples: {len(ng_samples)}")
    print(f"Total: {len(ok_samples) + len(ng_samples)}")

    return ok_samples, ng_samples


def extract_zone_statistics(samples):
    """Zone별 통계 추출"""
    records = []

    for sample in samples:
        sku = sample.get("sku", "unknown")
        judgment = sample.get("judgment", "unknown")
        overall_de = sample.get("overall_delta_e", 0)

        for zr in sample.get("zone_results", []):
            records.append(
                {
                    "sku": sku,
                    "judgment": judgment,
                    "overall_delta_e": overall_de,
                    "zone_name": zr["zone_name"],
                    "measured_L": zr["measured_lab"][0],
                    "measured_a": zr["measured_lab"][1],
                    "measured_b": zr["measured_lab"][2],
                    "target_L": zr["target_lab"][0] if zr.get("target_lab") else None,
                    "target_a": zr["target_lab"][1] if zr.get("target_lab") else None,
                    "target_b": zr["target_lab"][2] if zr.get("target_lab") else None,
                    "delta_e": zr["delta_e"],
                    "threshold": zr["threshold"],
                    "is_ok": zr["is_ok"],
                }
            )

    df = pd.DataFrame(records)
    print(f"\n=== Zone Statistics ===")
    print(f"Total zone measurements: {len(df)}")
    print(f"\nZone distribution:")
    print(df["zone_name"].value_counts())
    print(f"\nΔE statistics:")
    print(df["delta_e"].describe())

    return df


def extract_ring_sector_statistics(samples):
    """Ring×Sector 36개 셀 통계 추출"""
    records = []

    for sample in samples:
        sku = sample.get("sku", "unknown")
        judgment = sample.get("judgment", "unknown")

        for cell in sample.get("ring_sector_cells", []):
            records.append(
                {
                    "sku": sku,
                    "judgment": judgment,
                    "ring_index": cell["ring_index"],
                    "sector_index": cell["sector_index"],
                    "mean_L": cell["mean_L"],
                    "mean_a": cell["mean_a"],
                    "mean_b": cell["mean_b"],
                    "std_L": cell["std_L"],
                    "std_a": cell["std_a"],
                    "std_b": cell["std_b"],
                    "pixel_count": cell["pixel_count"],
                }
            )

    df = pd.DataFrame(records)
    print(f"\n=== Ring×Sector Statistics ===")
    print(f"Total cells: {len(df)}")
    print(f"\nRing distribution:")
    print(df["ring_index"].value_counts().sort_index())
    print(f"\nMean L* by Ring:")
    print(df.groupby("ring_index")["mean_L"].mean())

    return df


def extract_uniformity_statistics(samples):
    """균일성 통계 추출"""
    records = []

    for sample in samples:
        sku = sample.get("sku", "unknown")
        judgment = sample.get("judgment", "unknown")

        uniformity = sample.get("uniformity_analysis")
        if not uniformity:
            continue

        record = {
            "sku": sku,
            "judgment": judgment,
            "is_uniform": uniformity.get("is_uniform", False),
            "max_delta_e": uniformity.get("max_delta_e", 0),
            "mean_delta_e": uniformity.get("mean_delta_e", 0),
            "confidence": uniformity.get("confidence", 0),
        }

        # Ring별 균일성
        for ring_idx, ring_stats in uniformity.get("ring_uniformity", {}).items():
            record[f"ring{ring_idx}_mean_de"] = ring_stats.get("mean_de", 0)
            record[f"ring{ring_idx}_is_uniform"] = ring_stats.get("is_uniform", False)

        records.append(record)

    df = pd.DataFrame(records)
    print(f"\n=== Uniformity Statistics ===")
    print(f"Total samples: {len(df)}")
    print(f"\nUniform distribution:")
    print(df["is_uniform"].value_counts())
    print(f"\nMax ΔE statistics:")
    print(df["max_delta_e"].describe())

    return df


def visualize_zone_distribution(df_zones, output_dir="results/analysis"):
    """Zone별 ΔE 분포 시각화"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # 1. ΔE 분포 (Zone별)
    plt.figure(figsize=(12, 6))
    sns.boxplot(data=df_zones, x="zone_name", y="delta_e", hue="judgment")
    plt.title("ΔE Distribution by Zone and Judgment", fontsize=14)
    plt.ylabel("ΔE")
    plt.xlabel("Zone")
    plt.legend(title="Judgment")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/zone_de_distribution.png", dpi=150)
    print(f"\nSaved: {output_dir}/zone_de_distribution.png")

    # 2. Lab 공간 분포 (a* vs b*)
    plt.figure(figsize=(10, 8))
    for zone in df_zones["zone_name"].unique():
        zone_df = df_zones[df_zones["zone_name"] == zone]
        plt.scatter(zone_df["measured_a"], zone_df["measured_b"], label=f"Zone {zone}", alpha=0.6, s=50)

    plt.xlabel("a* (green ← → red)")
    plt.ylabel("b* (blue ← → yellow)")
    plt.title("Lab Color Space Distribution (a* vs b*)", fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/lab_space_distribution.png", dpi=150)
    print(f"Saved: {output_dir}/lab_space_distribution.png")


def visualize_ring_uniformity(df_ring_sector, output_dir="results/analysis"):
    """Ring별 균일성 시각화"""
    # Ring별 평균 L*
    plt.figure(figsize=(10, 6))
    ring_mean_L = df_ring_sector.groupby(["ring_index", "judgment"])["mean_L"].mean().unstack()
    ring_mean_L.plot(kind="bar", ax=plt.gca())
    plt.title("Mean L* by Ring and Judgment", fontsize=14)
    plt.xlabel("Ring Index")
    plt.ylabel("Mean L*")
    plt.legend(title="Judgment")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ring_mean_L.png", dpi=150)
    print(f"Saved: {output_dir}/ring_mean_L.png")

    # Ring별 표준편차
    plt.figure(figsize=(10, 6))
    ring_std_L = df_ring_sector.groupby(["ring_index", "judgment"])["std_L"].mean().unstack()
    ring_std_L.plot(kind="bar", ax=plt.gca())
    plt.title("Mean Std L* by Ring and Judgment", fontsize=14)
    plt.xlabel("Ring Index")
    plt.ylabel("Mean Std L*")
    plt.legend(title="Judgment")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ring_std_L.png", dpi=150)
    print(f"Saved: {output_dir}/ring_std_L.png")


def create_ml_dataset(df_zones, df_ring_sector, df_uniformity, output_dir="results/analysis"):
    """AI 학습용 데이터셋 생성"""
    # SKU별로 그룹화
    dataset = []

    for sku in df_uniformity["sku"].unique():
        sku_uniformity = df_uniformity[df_uniformity["sku"] == sku]

        for idx, row in sku_uniformity.iterrows():
            features = {
                "sku": sku,
                "judgment_label": 1 if row["judgment"] == "OK" else 0,
                # 균일성 특징
                "max_delta_e": row["max_delta_e"],
                "mean_delta_e": row["mean_delta_e"],
                "is_uniform": 1 if row["is_uniform"] else 0,
                # Ring별 특징
                "ring0_mean_de": row.get("ring0_mean_de", 0),
                "ring1_mean_de": row.get("ring1_mean_de", 0),
                "ring2_mean_de": row.get("ring2_mean_de", 0),
            }

            dataset.append(features)

    ml_df = pd.DataFrame(dataset)

    # CSV 저장
    output_path = f"{output_dir}/ml_dataset.csv"
    ml_df.to_csv(output_path, index=False)
    print(f"\nML Dataset saved: {output_path}")
    print(f"Shape: {ml_df.shape}")
    print(f"\nFeature columns:")
    for col in ml_df.columns:
        print(f"  - {col}")

    return ml_df


def simple_ml_model(ml_df):
    """간단한 ML 모델 학습 예제"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix
    from sklearn.model_selection import train_test_split

    # 특징과 라벨 분리
    X = ml_df.drop(["sku", "judgment_label"], axis=1)
    y = ml_df["judgment_label"]

    # Train/Test 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Random Forest 학습
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # 평가
    y_pred = model.predict(X_test)

    print(f"\n=== ML Model Performance ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"\nAccuracy: {model.score(X_test, y_test):.2%}")

    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["NG", "OK"]))

    print(f"\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Feature Importance
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({"feature": X.columns, "importance": importances}).sort_values(
        "importance", ascending=False
    )

    print(f"\nTop 5 Important Features:")
    print(feature_importance_df.head())

    return model, feature_importance_df


def main():
    print("=" * 60)
    print("JSON Telemetry Analysis")
    print("=" * 60)

    # 1. 데이터 수집
    ok_samples, ng_samples = collect_all_results()

    if len(ok_samples) + len(ng_samples) == 0:
        print("\n[ERROR] No result.json files found in results/web/")
        print("Please run inspections first to generate data.")
        return

    all_samples = ok_samples + ng_samples

    # 2. 통계 추출
    df_zones = extract_zone_statistics(all_samples)
    df_ring_sector = extract_ring_sector_statistics(all_samples)
    df_uniformity = extract_uniformity_statistics(all_samples)

    # 3. 시각화
    print(f"\n=== Generating Visualizations ===")
    visualize_zone_distribution(df_zones)
    visualize_ring_uniformity(df_ring_sector)

    # 4. ML 데이터셋 생성
    print(f"\n=== Creating ML Dataset ===")
    ml_df = create_ml_dataset(df_zones, df_ring_sector, df_uniformity)

    # 5. 간단한 ML 모델 학습 (샘플 충분 시)
    if len(ml_df) >= 20:
        print(f"\n=== Training Simple ML Model ===")
        model, feature_importance = simple_ml_model(ml_df)
    else:
        print(f"\n[INFO] Not enough samples ({len(ml_df)}) for ML training (need ≥20)")

    print(f"\n=== Analysis Complete ===")
    print(f"Results saved in: results/analysis/")


if __name__ == "__main__":
    main()
