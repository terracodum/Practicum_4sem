import hashlib
import matplotlib.pyplot as plt
import pandas as pd


class Slot:
    def __init__(self, remainder=None):
        self.remainder = remainder
        self.occupied = False
        self.continuation = False
        self.shifted = False

    def is_empty(self):
        return self.remainder is None


class QuotientFilter:
    def __init__(self, q: int, r: int):
        self.q = q  # количество бит на индекс
        self.r = r  # количество бит на остаток
        self.size = 1 << q
        self.table = [Slot() for _ in range(self.size)]

    def _hash(self, value: str):
        full_hash = int(hashlib.sha256(value.encode()).hexdigest(), 16)
        quotient = (full_hash >> self.r) & ((1 << self.q) - 1)
        remainder = full_hash & ((1 << self.r) - 1)
        return quotient, remainder

    def insert(self, value: str):
        q, r = self._hash(value)
        i = q
        steps = 0
        while not self.table[i].is_empty() and steps < self.size:
            i = (i + 1) % self.size
            steps += 1
        if steps < self.size:
            self.table[i] = Slot(r)
            self.table[i].occupied = True
        else:
            print("⚠️ Фильтр переполнен, вставка пропущена.")

    def query(self, value: str) -> bool:
        q, r = self._hash(value)
        i = q
        checked = 0
        while not self.table[i].is_empty() and checked < self.size:
            if self.table[i].remainder == r:
                return True
            i = (i + 1) % self.size
            checked += 1
        return False


def run_false_positive_test_qf(q=10, r=4, inserted=None, test_size=1000):
    qf = QuotientFilter(q, r)
    capacity = 1 << q
    inserted = inserted if inserted is not None else int(0.75 * capacity)  # до 75% загрузки

    inserted_elements = [f"item_{i}" for i in range(inserted)]
    test_elements = [f"test_{i}" for i in range(test_size)]

    for item in inserted_elements:
        qf.insert(item)

    false_positives = sum(qf.query(item) for item in test_elements)
    fpr = false_positives / len(test_elements)
    return {"q": q, "r": r, "false_positive_rate": fpr}


def experiment_qf():
    results = []
    for q in range(6, 11):     # Размер таблицы: от 2^6 до 2^10
        for r in [3, 4, 5, 6]:  # Размер остатка
            res = run_false_positive_test_qf(q=q, r=r)
            print(f"q={res['q']}, r={res['r']}, FPR={res['false_positive_rate']:.4%}")
            results.append(res)
    return pd.DataFrame(results)


def plot_qf_results(df: pd.DataFrame):
    pivot = df.pivot(index='q', columns='r', values='false_positive_rate')
    pivot.plot(marker='o')
    plt.title("Ложноположительные срабатывания Quotient Filter")
    plt.xlabel("q (log₂ размера таблицы)")
    plt.ylabel("False Positive Rate")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = experiment_qf()
    print("\n📋 Таблица результатов:\n", df)
    plot_qf_results(df)