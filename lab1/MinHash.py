import random
import hashlib
import matplotlib.pyplot as plt
import pandas as pd
from statistics import mean


class MinHash:
    def __init__(self, num_hashes=100):
        self.num_hashes = num_hashes
        self.max_hash = (1 << 32) - 1
        self.hash_funcs = self._generate_hash_functions()

    def _generate_hash_functions(self):
        hash_funcs = []
        for _ in range(self.num_hashes):
            a = random.randint(1, self.max_hash)
            b = random.randint(0, self.max_hash)
            hash_funcs.append(lambda x, a=a, b=b: (a * x + b) % self.max_hash)
        return hash_funcs

    def _hash_token(self, token):
        return int(hashlib.md5(token.encode()).hexdigest(), 16)

    def compute_signature(self, tokens):
        signature = []
        for func in self.hash_funcs:
            min_hash = min(func(self._hash_token(t)) for t in tokens)
            signature.append(min_hash)
        return signature

    def jaccard_sim(self, sig1, sig2):
        return sum(1 for a, b in zip(sig1, sig2) if a == b) / len(sig1)


def generate_sets(base_size=100, overlap_ratio=0.5):
    overlap_size = int(base_size * overlap_ratio)
    unique_size = base_size - overlap_size
    base = [f"item_{i}" for i in range(overlap_size)]
    set1 = set(base + [f"item1_{i}" for i in range(unique_size)])
    set2 = set(base + [f"item2_{i}" for i in range(unique_size)])
    return set1, set2


def experiment_varying_hashes():
    overlaps = [0.1, 0.3, 0.5, 0.7, 0.9]
    hash_counts = [10, 20, 50, 100, 150, 200]
    results = []

    for overlap in overlaps:
        for num_hashes in hash_counts:
            errors = []
            for _ in range(10):
                set1, set2 = generate_sets(100, overlap)
                true_jaccard = len(set1 & set2) / len(set1 | set2)
                mh = MinHash(num_hashes=num_hashes)
                sig1 = mh.compute_signature(set1)
                sig2 = mh.compute_signature(set2)
                estimated = mh.jaccard_sim(sig1, sig2)
                errors.append(abs(estimated - true_jaccard))
            results.append({
                "overlap": overlap,
                "num_hashes": num_hashes,
                "avg_error": mean(errors)
            })

    return pd.DataFrame(results)


def plot_results(df):
    plt.figure(figsize=(10, 6))
    for overlap in sorted(df["overlap"].unique()):
        subset = df[df["overlap"] == overlap]
        plt.plot(subset["num_hashes"], subset["avg_error"], marker='o', label=f"Пересечение {int(overlap * 100)}%")
    plt.title("Ошибка MinHash в зависимости от количества хэш-функций")
    plt.xlabel("Число хэш-функций (num_hashes)")
    plt.ylabel("Средняя ошибка")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = experiment_varying_hashes()
    print(df)
    plot_results(df)
