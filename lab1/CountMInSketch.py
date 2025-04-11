import hashlib
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class CountMinSketch:
    def __init__(self, width, depth):
        self.width = width
        self.depth = depth
        self.table = np.zeros((depth, width), dtype=int)
        self.hash_seeds = [random.randint(0, 1 << 32) for _ in range(depth)]

    def _hash(self, x, seed):
        h = hashlib.blake2b(str(x).encode(), digest_size=4, key=seed.to_bytes(4, 'little'))
        return int.from_bytes(h.digest(), 'little') % self.width

    def add(self, x, count=1):
        for i in range(self.depth):
            idx = self._hash(x, self.hash_seeds[i])
            self.table[i][idx] += count

    def estimate(self, x):
        return min(self.table[i][self._hash(x, self.hash_seeds[i])] for i in range(self.depth))


# Генерация данных и оценка FP
def run_fp_test(width, depth, stream_size=1000, test_size=500):
    cms = CountMinSketch(width, depth)
    inserted = set()

    for _ in range(stream_size):
        item = random.randint(1, 1_000_000)
        cms.add(item)
        inserted.add(item)

    false_positives = 0
    for _ in range(test_size):
        item = random.randint(1, 1_000_000)
        if item not in inserted and cms.estimate(item) > 0:
            false_positives += 1

    fp_rate = false_positives / test_size
    return fp_rate


# Проведение экспериментов
def experiment():
    widths = [100, 300, 500, 800, 1000]
    depths = [2, 4, 6, 8]
    results = []

    for w in widths:
        for d in depths:
            rates = [run_fp_test(w, d) for _ in range(5)]
            avg_fp = sum(rates) / len(rates)
            results.append({
                'width': w,
                'depth': d,
                'fp_rate': avg_fp
            })

    return pd.DataFrame(results)


# Построение графика
def plot_fp(df):
    plt.figure(figsize=(10, 6))
    for d in sorted(df['depth'].unique()):
        subset = df[df['depth'] == d]
        plt.plot(subset['width'], subset['fp_rate'], marker='o', label=f'depth={d}')

    plt.title("Зависимость ложноположительных срабатываний от параметров Count-Min Sketch")
    plt.xlabel("Ширина таблицы (width)")
    plt.ylabel("False Positive Rate")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = experiment()
    print(df)
    plot_fp(df)
