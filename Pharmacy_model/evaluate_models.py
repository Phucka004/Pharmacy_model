from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List

from src.baseline_ml.inference import MedicalMLPredictor
from src.rag_system.inference import _RUNTIME as rag_runtime

DATA_PATH = Path("data/silver/synthetic_medical_qa.json")


@dataclass
class TestSample:
    question: str
    true_category: str


def load_test_set(path: Path) -> List[TestSample]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    test_set: List[TestSample] = []
    for category, item in payload.items():
        questions = item.get("questions", [])
        if not questions:
            continue
        test_set.append(TestSample(question=str(questions[-1]), true_category=category))
    return test_set


def ms(seconds: float) -> float:
    return seconds * 1000.0


def main() -> None:
    test_set = load_test_set(DATA_PATH)
    baseline = MedicalMLPredictor(
        model_path="models/medical_classifier.pkl",
        qa_path="data/silver/synthetic_medical_qa.json",
        products_path="data/silver/products_kb.csv",
    )

    baseline_correct = 0
    rag_correct = 0
    baseline_latencies: List[float] = []
    rag_stage1_latencies: List[float] = []
    rag_stage2_latencies: List[float] = []
    rag_total_latencies: List[float] = []

    print("=" * 110)
    print("BENCHMARK: ML BASELINE vs RAG ADVANCED")
    print("=" * 110)

    for idx, sample in enumerate(test_set, start=1):
        start = time.perf_counter()
        baseline_result = baseline.predict(sample.question)
        baseline_time = time.perf_counter() - start
        baseline_latencies.append(ms(baseline_time))
        if baseline_result.category == sample.true_category:
            baseline_correct += 1

        stage1_start = time.perf_counter()
        rag_category, display_name, is_out_of_domain, _ = rag_runtime.diagnosis_phase(sample.question)
        stage1_time = time.perf_counter() - stage1_start

        stage2_start = time.perf_counter()
        products = [] if is_out_of_domain else rag_runtime.product_retrieval_phase(rag_category or "", display_name or "")
        stage2_time = time.perf_counter() - stage2_start

        rag_stage1_latencies.append(ms(stage1_time))
        rag_stage2_latencies.append(ms(stage2_time))
        rag_total_latencies.append(ms(stage1_time + stage2_time))

        if not is_out_of_domain and rag_category == sample.true_category:
            rag_correct += 1

        print(
            f"[{idx:03d}/{len(test_set):03d}] "
            f"BASELINE={baseline_result.category} ({baseline_latencies[-1]:.2f} ms) | "
            f"RAG={rag_category or 'None'} (S1={rag_stage1_latencies[-1]:.2f} ms, S2={rag_stage2_latencies[-1]:.2f} ms, Total={rag_total_latencies[-1]:.2f} ms) | "
            f"TRUE={sample.true_category}"
        )

    total = len(test_set) or 1
    baseline_acc = baseline_correct / total * 100.0
    rag_acc = rag_correct / total * 100.0
    baseline_avg = sum(baseline_latencies) / total
    rag_avg_total = sum(rag_total_latencies) / total
    rag_avg_stage1 = sum(rag_stage1_latencies) / total
    rag_avg_stage2 = sum(rag_stage2_latencies) / total
    rag_stage1_share = (rag_avg_stage1 / rag_avg_total * 100.0) if rag_avg_total else 0.0
    rag_stage2_share = (rag_avg_stage2 / rag_avg_total * 100.0) if rag_avg_total else 0.0

    print("\n" + "=" * 110)
    print("BENCHMARK REPORT")
    print("=" * 110)
    print(f"{'Model':<20}{'Accuracy (%)':>18}{'Avg Latency (ms)':>22}")
    print("-" * 60)
    print(f"{'ML Baseline':<20}{baseline_acc:>18.2f}{baseline_avg:>22.2f}")
    print(f"{'RAG Advanced':<20}{rag_acc:>18.2f}{rag_avg_total:>22.2f}")
    print("=" * 110)

    print("\n" + "=" * 110)
    print("RAG BREAKDOWN")
    print("=" * 110)
    print(f"{'Metric':<28}{'Stage-1 Vector':>20}{'Stage-2 Pandas':>20}{'Total':>16}")
    print("-" * 84)
    print(f"{'Average Latency (ms)':<28}{rag_avg_stage1:>20.2f}{rag_avg_stage2:>20.2f}{rag_avg_total:>16.2f}")
    print(f"{'Latency Share (%)':<28}{rag_stage1_share:>20.2f}{rag_stage2_share:>20.2f}{100.00:>16.2f}")
    print("=" * 110)


if __name__ == "__main__":
    main()
