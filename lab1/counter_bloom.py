import hashlib
import random
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Set


class CountingBloomFilter:
    def __init__(self, size: int, hash_count: int):
        self.size = size
        self.hash_count = hash_count
        self.counter_array = [0] * size

    def _hashes(self, item: str) -> List[int]:
        return [
            int(hashlib.sha256((item + str(i)).encode()).hexdigest(), 16) % self.size
            for i in range(self.hash_count)
        ]

    def add(self, item: str):
        for idx in self._hashes(item):
            self.counter_array[idx] += 1

    def remove(self, item: str):
        for idx in self._hashes(item):
            if self.counter_array[idx] > 0:
                self.counter_array[idx] -= 1

    def __contains__(self, item: str) -> bool:
        return all(self.counter_array[idx] > 0 for idx in self._hashes(item))

    def union(self, other: 'CountingBloomFilter') -> 'CountingBloomFilter':
        if self.size != other.size or self.hash_count != other.hash_count:
            raise ValueError("Фильтры должны быть одинаковыми")
        result = CountingBloomFilter(self.size, self.hash_count)
        result.counter_array = [max(a, b) for a, b in zip(self.counter_array, other.counter_array)]
        return result

    def intersection(self, other: 'CountingBloomFilter') -> 'CountingBloomFilter':
        if self.size != other.size or self.hash_count != other.hash_count:
            raise ValueError("Фильтры должны быть одинаковыми")
        result = CountingBloomFilter(self.size, self.hash_count)
        result.counter_array = [min(a, b) for a, b in zip(self.counter_array, other.counter_array)]
        return result


def false_positive_rate(filter: CountingBloomFilter, inserted: Set[str], test_set: Set[str]) -> float:
    false_positives = sum(1 for item in test_set if item in filter and item not in inserted)
    return false_positives / len(test_set)


def experiment_counting_bloom(n=1000, test_size=1000, m_values=None, k_values=None):
    if m_values is None:
        m_values = [1000, 2000, 4000, 8000, 16000]
    if k_values is None:
        k_values = [2, 4, 6, 8, 10]

    results = []

    for m in m_values:
        for k in k_values:
            cbf = CountingBloomFilter(m, k)
            inserted = {f"item_{i}" for i in range(n)}
            for item in inserted:
                cbf.add(item)

            # удалим часть элементов (20%)
            for item in list(inserted)[:int(0.2 * n)]:
                cbf.remove(item)

            test_set = {f"test_{i}" for i in range(test_size)}
            fpr = false_positive_rate(cbf, inserted, test_set)

            results.append({"m": m, "k": k, "false_positive_rate": fpr})
            print(f"m={m}, k={k} -> FPR={fpr:.4f}")

    return pd.DataFrame(results)


def plot_fpr_counting(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    for k in sorted(df["k"].unique()):
        subset = df[df["k"] == k]
        plt.plot(subset["m"], subset["false_positive_rate"], label=f"k={k}")
    plt.xlabel("Размер массива m")
    plt.ylabel("Ложноположительный процент")
    plt.title("FPR в счётном фильтре Блума при разных m и k")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 🚀 Запуск
if __name__ == "__main__":
    df = experiment_counting_bloom()
    plot_fpr_counting(df)
