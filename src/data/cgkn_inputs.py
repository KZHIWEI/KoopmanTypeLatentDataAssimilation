import json, os
import numpy as np, h5py

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.environ.get("CGKN_PREP_OUT", "checkpoint/cgkn_inputs")


if __name__ == "__main__":
    os.chdir(REPO)

    def stats_from_iter(it, C, ch_axis):
        n = 0
        s = np.zeros(C, np.float64)
        ss = np.zeros(C, np.float64)
        for chunk in it:
            a = np.asarray(chunk, dtype=np.float64)
            a = np.moveaxis(a, ch_axis, -1).reshape(-1, C)
            n += a.shape[0]
            s += a.sum(0)
            ss += (a * a).sum(0)
        m = s / n
        v = ss / n - m * m
        return m.astype(np.float32), np.sqrt(np.maximum(v, 0)).astype(np.float32)

    def npy_chunks(path, tsel, chan_axis=2, tr=None):
        a = np.load(path, mmap_mode="r")
        trs = range(a.shape[0]) if tr is None else tr
        for i in trs:
            yield a[i, tsel]

    def h5_chunks(path, idx, block=200):
        with h5py.File(path, "r") as f:
            d = f["data"]
            for i in range(0, len(idx), block):
                yield d[idx[i : i + block].tolist()]

    res = {"cgkn_norm": {}, "bench_norm": {}}

    res["cgkn_norm"]["cylinder"] = [
        x.tolist()
        for x in stats_from_iter(
            npy_chunks(
                "data/cylinder/train.npy",
                slice(None, None, 5),
                tr=range(4),
            ),
            2,
            1,
        )
    ]
    res["cgkn_norm"]["shallow_water"] = [
        x.tolist()
        for x in stats_from_iter(
            npy_chunks(
                "data/shallow_water/train.npy", slice(None, None, 2), tr=range(4)
            ),
            3,
            1,
        )
    ]
    idx = np.arange(2000) * 4
    res["cgkn_norm"]["era5"] = [
        x.tolist() for x in stats_from_iter(h5_chunks("data/era5/train.h5", idx), 5, 3)
    ]
    res["cgkn_norm"]["era5_high"] = [
        x.tolist()
        for x in stats_from_iter(
            h5_chunks("data/era5_high/train.h5", idx, block=100), 5, 3
        )
    ]
    print("cgkn norms done", flush=True)

    def bench_npy(path, C):
        a = np.load(path, mmap_mode="r")
        step = max(1, a.shape[1] // 200)
        return [
            x.tolist()
            for x in stats_from_iter(
                npy_chunks(path, slice(None, None, step), tr=None), C, 1
            )
        ]

    res["bench_norm"]["lorenz96"] = bench_npy("data/lorenz96/train.npy", 1)
    res["bench_norm"]["cylinder"] = bench_npy("data/cylinder/train.npy", 2)
    res["bench_norm"]["shallow_water"] = bench_npy("data/shallow_water/train.npy", 3)
    for ds, p, T in (
        ("era5", "data/era5/train.h5", 54056),
        ("era5_high", "data/era5_high/train.h5", 54056),
    ):
        step = max(1, T // 200)
        idx = np.arange(0, T, step)
        res["bench_norm"][ds] = [
            x.tolist() for x in stats_from_iter(h5_chunks(p, idx, block=64), 5, 3)
        ]
    print("bench norms done", flush=True)

    for d in (res["cgkn_norm"], res["bench_norm"]):
        for k in d:
            d[k][1] = [v + 1e-6 for v in d[k][1]]

    with open(f"{OUT}/normalization.json", "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps(res, indent=1))
