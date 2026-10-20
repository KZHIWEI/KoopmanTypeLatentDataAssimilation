import argparse, hashlib, json, os, shutil, sys

REPO_ID = "KZHIwEI/KoopmanDA"
REPO_TYPE = "dataset"
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
MANIFEST = os.path.join(HERE, "manifest.json")


def sha256(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def state(rec, dest):
    p = os.path.join(dest, rec["path"])
    if not os.path.isfile(p):
        return "missing"
    if os.path.getsize(p) != rec["bytes"]:
        return "wrong size"
    return "ok" if sha256(p) == rec["sha256"] else "wrong hash"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=ROOT)
    ap.add_argument("--only", default="")
    ap.add_argument("--kind", choices=("all", "data", "ckpt"), default="all")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    files = json.load(open(MANIFEST))["files"]
    if a.only:
        keep = {x.strip() for x in a.only.split(",")}
        shared = {"cgkn_inputs", "pretrained_ae"}
        files = [
            r
            for r in files
            if r["kind"] in shared
            or r["dataset"] in keep
            or (r["kind"] == "ablation" and r["model"] in keep)
        ]
        if not files:
            sys.exit(a.only)
    if a.kind != "all":
        files = [r for r in files if (r["kind"] == "data") == (a.kind == "data")]
    print(
        f"  {len(files)} files, {sum((r['bytes'] for r in files)) / 1000000000.0:.2f} GB -> {a.dest}"
    )
    todo, bad = ([], [])
    for r in files:
        s = state(r, a.dest)
        if s == "ok" and (not a.force):
            continue
        (bad if s != "missing" else todo).append((s, r))
    todo += bad
    if a.check:
        for s, r in bad:
            print(f"  {s:11s} {r['path']}")
        n_missing = sum((1 for (s, _) in todo if s == "missing"))
        print(
            f"  verified {len(files) - len(todo)}/{len(files)}; {n_missing} missing, {len(bad)} corrupt"
        )
        sys.exit(1 if todo else 0)
    if not todo:
        print(f"  {len(files)} ok")
        return
    from huggingface_hub import hf_hub_download

    stage = os.path.join(a.dest, ".hf_download")
    for i, (s, r) in enumerate(todo, 1):
        print(f"  [{i}/{len(todo)}] {s:11s} {r['path']}", flush=True)
        got = hf_hub_download(
            repo_id=r.get("repo", REPO_ID),
            repo_type=r.get("repo_type", REPO_TYPE),
            filename=r.get("hub_path", r["path"]),
            local_dir=stage,
        )
        want = os.path.join(a.dest, r["path"])
        if os.path.abspath(got) != os.path.abspath(want):
            os.makedirs(os.path.dirname(want), exist_ok=True)
            os.replace(got, want)
            got = want
        if sha256(got) != r["sha256"]:
            sys.exit(r["path"])
    shutil.rmtree(stage, ignore_errors=True)
    print(f"  {len(todo)} downloaded")


if __name__ == "__main__":
    main()
