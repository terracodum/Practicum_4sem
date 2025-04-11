import hashlib
import math
import random
import matplotlib.pyplot as plt
import pandas as pd


class HyperLogLog:
    def __init__(self, b: int = 10):
        self.b = b  # log2 количества регистров
        self.m = 1 << b  # количество регистров
        self.registers = [0] * self.m
        self.alpha = self._get_alpha(self.m)

    def _get_alpha(self, m: int) -> float:
        if m == 16:
            return 0.673
        elif m == 32:
            return 0.697
        elif m == 64:
            return 0.709
        else:
            return 0.7213 / (1 + 1.079 / m)

    def _hash(self, value: str) -> int:
        return int(hashlib.sha256(value.encode()).hexdigest(), 16)

    def add(self, value: str):
        x = self._hash(value)
        j = x & (self.m - 1)  # индекс регистра (последние b бит)
        w = x >> self.b       # остальная часть хэша
        self.registers[j] = max(self.registers[j], self._rho(w))

    def _rho(self, w: int) -> int:
        # позиция первого единичного бита (от начала)
        bin_w = bin(w)[2:]
        return len(bin_w) - len(bin_w.lstrip("0")) + 1 if w > 0 else self.b + 1

    def count(self) -> float:
        Z = 1.0 / sum([2.0 ** -reg for reg in self.registers])
        raw_estimate = self.alpha * self.m * self.m * Z

        if raw_estimate <= 2.5 * self.m:
            V = self.registers.count(0)
            if V > 0:
                return self.m * math.log(self.m / V)
            return raw_estimate

        elif raw_estimate <= (1 / 30.0) * (1 << 32):
            return raw_estimate

        elif raw_estimate < (1 << 32):
            try:
                return -(1 << 32) * math.log(1 - raw_estimate / (1 << 32))
            except ValueError:
                return raw_estimate
        else:
            return raw_estimate


def experiment_hyperloglog(true_count=10000, test_size=10, b_values=[4, 5, 6, 7, 8, 9, 10, 15]):
    results = []

    for b in b_values:
        hll = HyperLogLog(b=b)
        elements = [f"item_{i}" for i in range(true_count)]
        for item in elements:
            hll.add(item)

        estimate = hll.count()
        error = abs(estimate - true_count) / true_count
        print(f"b={b}, registers={2**b}, estimate={estimate:.0f}, error={error:.4%}")
        results.append({"b": b, "registers": 2 ** b, "estimated": estimate, "error": error})

    return pd.DataFrame(results)


def plot_hll_errors(df: pd.DataFrame):
    plt.figure(figsize=(10, 6))
    plt.plot(df["registers"], df["error"], marker='o')
    plt.xlabel("Количество регистров (m = 2^b)")
    plt.ylabel("Относительная ошибка")
    plt.title("Зависимость ошибки от количества регистров (HyperLogLog)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


# 🚀 Запуск
if __name__ == "__main__":
    df = experiment_hyperloglog()
    plot_hll_errors(df)
