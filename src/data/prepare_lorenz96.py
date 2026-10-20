import os
import numpy as np

D, F = (40, 8.0)
DT, SUBS = (0.01, 5)
TRANSIENT = 3000
N_TRAIN, N_VAL, N_TEST, T_SAVE = (128, 16, 16, 1500)


def rhs(x):
    return (np.roll(x, -1, -1) - np.roll(x, 2, -1)) * np.roll(x, 1, -1) - x + F


def rk4(x, dt):
    k1 = rhs(x)
    k2 = rhs(x + 0.5 * dt * k1)
    k3 = rhs(x + 0.5 * dt * k2)
    k4 = rhs(x + dt * k3)
    return x + dt / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate(n, seed):
    rng = np.random.RandomState(seed)
    x = F + rng.randn(n, D)
    for _ in range(TRANSIENT):
        x = rk4(x, DT)
    out = np.empty((n, T_SAVE, D), np.float32)
    for t in range(T_SAVE):
        for _ in range(SUBS):
            x = rk4(x, DT)
        out[:, t] = x
    return out.reshape(n, T_SAVE, 1, 8, 5)


def main():
    os.makedirs("data/lorenz96", exist_ok=True)
    for split, n, seed in (
        ("train", N_TRAIN, 0),
        ("val", N_VAL, 1),
        ("test", N_TEST, 2),
    ):
        a = simulate(n, seed)
        np.save(f"data/lorenz96/{split}.npy", a)
        print(split, a.shape, "std", round(float(a.std()), 3), flush=True)


if __name__ == "__main__":
    main()
