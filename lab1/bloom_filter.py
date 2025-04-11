import hashlib
import random
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Set

class BloomFilter:
    def __init__(self, size: int, hash_count: int):
        self.size = size
        self.hash_count = hash_count
        self.bit_array = [0] * size

    def _custom_hashes(self, item: str) -> List[int]:
        # Собственные хеш-функции: просто основанные на md5 + разные модификаторы
        return [
            int(hashlib.md5((item + str(i)).encode()).hexdigest(), 16) % self.size
            for i in range(self.hash_count)
        ]

    def add(self, item: str):
        for hash_val in self._custom_hashes(item):
            self.bit_array[hash_val] = 1

    def __contains__(self, item: str) -> bool:
        return all(self.bit_array[hash_val] for hash_val in self._custom_hashes(item))

    def union(self, other: 'BloomFilter') -> 'BloomFilter':
        if self.size != other.size or self.hash_count != other.hash_count:
            raise ValueError("Фильтры должны быть одинаковыми по размеру и числу хешей")
        result = BloomFilter(self.size, self.hash_count)
        result.bit_array = [a | b for a, b in zip(self.bit_array, other.bit_array)]
        return result

    def intersection(self, other: 'BloomFilter') -> 'BloomFilter':
        if self.size != other.size or self.hash_count != other.hash_count:
            raise ValueError("Фильтры должны быть одинаковыми по размеру и числу хешей")
        result = BloomFilter(self.size, self.hash_count)
        result.bit_array = [a & b for a, b in zip(self.bit_array, other.bit_array)]
        return result


def false_positive_rate(bloom: BloomFilter, inserted: Set[str], test_set: Set[str]) -> float:
    false_positives = sum(1 for item in test_set if item in bloom and item not in inserted)
    return false_positives / len(test_set)


def run_experiment(n=1000, test_size=1000, m_values=None, k_values=None):
    if m_values is None:
        m_values = [1000, 2000, 4000, 8000, 16000, 32000]
    if k_values is None:
        k_values = [2, 4, 6, 8, 10]

    results = []

    for m in m_values:
        for k in k_values:
            bloom = BloomFilter(size=m, hash_count=k)
            inserted = {f"item_{i}" for i in range(n)}
            for item in inserted:
                bloom.add(item)

            test_set = {f"test_{i}" for i in range(test_size)}
            fpr = false_positive_rate(bloom, inserted, test_set)
            results.append({"m": m, "k": k, "false_positive_rate": fpr})
            print(f"m={m}, k={k} => FPR={fpr:.4f}")

    df = pd.DataFrame(results)
    return df


def plot_fpr(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    for k in sorted(df["k"].unique()):
        subset = df[df["k"] == k]
        plt.plot(subset["m"], subset["false_positive_rate"], label=f"k={k}")
    plt.xlabel("Размер массива (m)")
    plt.ylabel("Процент ложноположительных")
    plt.title("Зависимость ложноположительных срабатываний от m и k")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# 🚀 Запуск эксперимента
if __name__ == "__main__":
    df = run_experiment()
    plot_fpr(df)
