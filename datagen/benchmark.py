"""A/B circuit benchmark: freeze a test set, run models, score, compare.

    python -m datagen.benchmark build   --out data/bench --seed 7
    python -m datagen.benchmark calc    --bench data/bench --out data/bench/preds_calculator.jsonl
    python -m datagen.benchmark run     --bench data/bench --out data/bench/preds_base.jsonl \\
            --base-url https://openrouter.ai/api/v1 --model ricursiveai/ripple-65b-preview --openrouter
    python -m datagen.benchmark score   --bench data/bench --preds data/bench/preds_base.jsonl
    python -m datagen.benchmark compare --bench data/bench --a preds_base.jsonl --b preds_lora.jsonl

The test set has four categories: unseen component values (20), out-of-range values (15), unseen
circuit configurations (15) and design or fault correction (10). Training and validation sets come
only from the three training families inside the training ranges, grouped by circuit so no netlist
appears in two splits.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import pathlib
import random
import sys

from . import calculator, design, families, modelrun, scoring
from .measure import DatasetError
from .pipeline import make_example, split_groups

SIZES = {"unseen_values": 20, "out_of_range": 15, "unseen_config": 15, "design": 10}


def _write(path: pathlib.Path, rows: list[dict]) -> str:
    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in rows)
    path.write_text(text)
    return hashlib.sha256(text.encode()).hexdigest()


def _read(path: pathlib.Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _generate(rng, fns: dict, ranges, n: int, rejects: collections.Counter, seen: set) -> list[dict]:
    out, attempts = [], 0
    names = list(fns)
    while len(out) < n:
        attempts += 1
        if attempts > n * 50:
            raise RuntimeError(f"could not generate {n} examples; rejections: {dict(rejects)}")
        g = fns[names[attempts % len(names)]](rng, ranges)
        try:
            ex = make_example(g.netlist, g.spec, family=g.family, source="analytic", analytic=g.analytic,
                              rationale=g.rationale)
        except DatasetError as e:
            rejects[str(e).split(":")[0]] += 1
            continue
        if ex["id"] in seen:
            continue
        seen.add(ex["id"])
        out.append(ex)
    return out


def build(out: pathlib.Path, seed: int, n_train: int, n_val: int) -> dict:
    rng = random.Random(seed)
    rejects: collections.Counter = collections.Counter()
    seen: set = set()
    out.mkdir(parents=True, exist_ok=True)

    pool = _generate(rng, families.TRAIN_FAMILIES, families.TRAIN, n_train + n_val + SIZES["unseen_values"] + 60,
                     rejects, seen)
    parts = split_groups(pool, {"unseen_values": SIZES["unseen_values"], "val": n_val, "train": n_train},
                         "netlist", rng)
    train, val = parts["train"][:n_train], parts["val"][:n_val]
    unseen = parts["unseen_values"][:SIZES["unseen_values"]]
    ood = _generate(rng, families.TRAIN_FAMILIES, families.OOD, SIZES["out_of_range"], rejects, seen)
    configs = []
    for name, fn in families.CONFIG_FAMILIES.items():
        configs += _generate(rng, {name: fn}, families.TRAIN, SIZES["unseen_config"] // len(families.CONFIG_FAMILIES),
                             rejects, seen)

    designs = []
    while len(designs) < SIZES["design"]:
        for p in design.design_problems(rng, SIZES["design"]):
            # every design problem must be solvable with a standard value, or it measures nothing
            snapped = calculator.predict(p)
            if design.evaluate(p, design.parse_value(snapped, p["unit"]))["success"] and p["id"] not in seen:
                seen.add(p["id"])
                designs.append(p)
            if len(designs) == SIZES["design"]:
                break

    for ex, cat in [(e, "unseen_values") for e in unseen] + [(e, "out_of_range") for e in ood] + \
            [(e, "unseen_config") for e in configs]:
        ex["category"] = cat
    test = unseen + ood + configs + designs

    # integrity: nothing in the test set may have been trained on
    train_nets = {e["netlist"] for e in train + val}
    assert not {e["netlist"] for e in test if "netlist" in e} & train_nets, "a test netlist appears in training"
    train_topos = {e["topology"] for e in train + val}
    assert not {e["topology"] for e in configs} & train_topos, "an unseen configuration shares a training topology"
    assert not {e["netlist"] for e in val} & {e["netlist"] for e in train}, "a netlist is in both train and val"
    # A family that fails every time is not an error to the generator: the others fill the quota. A
    # title heuristic once rejected every series RLC example and the build still "succeeded", so
    # every family must be well represented or the build stops.
    share = collections.Counter(e["family"] for e in train)
    for fam in families.TRAIN_FAMILIES:
        assert share[fam] >= 0.15 * len(train), f"training family {fam} is underrepresented: {dict(share)}"
    cshare = collections.Counter(e["family"] for e in configs)
    assert set(cshare) == set(families.CONFIG_FAMILIES), f"an unseen configuration family is missing: {dict(cshare)}"

    for e in train + val:
        e["messages"] = [{"role": "system", "content": modelrun.SYSTEM},
                         {"role": "user", "content": e["prompt"]},
                         {"role": "assistant", "content": e["completion"]}]
    shas = {"train": _write(out / "train.jsonl", train), "val": _write(out / "val.jsonl", val),
            "test": _write(out / "test.jsonl", test)}
    count = lambda rows, key: dict(collections.Counter(r.get(key) for r in rows))  # noqa: E731
    manifest = {
        "created": datetime.datetime.now().astimezone().isoformat(timespec="seconds"), "seed": seed,
        "sha256": shas, "sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "test_by_category": count(test, "category"), "test_by_family": count(test, "family"),
        "train_by_family": count(train, "family"),
        "train_by_kind": dict(collections.Counter(e["measurement"]["kind"] for e in train)),
        "rejected_during_generation": dict(rejects),
        "ground_truth": "ngspice, required to agree with an independent MNA solver and the closed-form formula",
        "system_prompt": modelrun.SYSTEM, "decoding": modelrun.DECODING,
        "limits": ["Train and test share the three training families' templates for unseen_values; that category "
                   "tests interpolation and format, not structural generalisation.",
                   "unseen_config uses three topologies never seen in training (topology fingerprints disjoint).",
                   "design problems are a task type absent from training."],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def _predict_file(problems: list[dict], rows: list[dict], out: pathlib.Path, meta: dict) -> None:
    _write(out, rows)
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))


def _scored(problems: list[dict], preds_path: pathlib.Path) -> list[dict]:
    """Score every problem; a missing reply counts as a transport error, never as a wrong answer."""
    preds = {r["id"]: r for r in _read(preds_path)}
    out = []
    for p in problems:
        row = preds.get(p["id"], {"error": "no reply recorded"})
        rec = scoring.score(p, row.get("response", ""), truncated=row.get("finish_reason") == "length",
                            transport_error=bool(row.get("error")))
        rec["completion_tokens"] = (row.get("usage") or {}).get("completion_tokens")
        out.append(rec)
    return out


def _table(summary: dict, label: str) -> str:
    lines = [f"{label}", f"{'category':16} {'n':>3} {'xport':>5} {'trunc':>5} {'valid':>6} {'strict':>7} {'correct':>8} "
             f"{'median rel err':>15} {'design ok':>9} {'med tokens':>10}"]
    rows = list(summary["by_category"].items()) + [("OVERALL", summary["overall"])]
    for cat, s in rows:
        med = f"{100 * s['median_rel_error']:.2f}%" if s["median_rel_error"] is not None else "-"
        dz = f"{s['design_success']}/{s['design_n']}" if "design_n" in s else "-"
        tk = f"{s['median_completion_tokens']:.0f}" if s.get("median_completion_tokens") else "-"
        lines.append(f"{cat:16} {s['n']:>3} {s['transport_errors']:>5} {s['truncated']:>5} {s['valid']:>6} "
                     f"{s['strict_format']:>7} {s['correct_within_tolerance']:>4}/{s['n']:<3} {med:>15} {dz:>9} {tk:>10}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m datagen.benchmark")
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", type=pathlib.Path, required=True)
    b.add_argument("--seed", type=int, default=7)
    b.add_argument("--train", type=int, default=300)
    b.add_argument("--val", type=int, default=60)
    c = sub.add_parser("calc")
    c.add_argument("--bench", type=pathlib.Path, required=True)
    c.add_argument("--out", type=pathlib.Path, required=True)
    r = sub.add_parser("run")
    r.add_argument("--bench", type=pathlib.Path, required=True)
    r.add_argument("--out", type=pathlib.Path, required=True)
    r.add_argument("--base-url", required=True)
    r.add_argument("--model", required=True)
    r.add_argument("--key-file", type=pathlib.Path, default=pathlib.Path.home() / ".openrouter_key")
    r.add_argument("--openrouter", action="store_true", help="pin OpenRouter providers (GMICloud first)")
    r.add_argument("--workers", type=int, default=6)
    s = sub.add_parser("score")
    s.add_argument("--bench", type=pathlib.Path, required=True)
    s.add_argument("--preds", type=pathlib.Path, required=True)
    m = sub.add_parser("compare")
    m.add_argument("--bench", type=pathlib.Path, required=True)
    m.add_argument("--a", type=pathlib.Path, required=True)
    m.add_argument("--b", type=pathlib.Path, required=True)
    args = ap.parse_args(argv)

    if args.cmd == "build":
        man = build(args.out, args.seed, args.train, args.val)
        print(json.dumps({k: man[k] for k in ("sizes", "test_by_category", "rejected_during_generation", "sha256")},
                         indent=1))
        return 0
    problems = _read(args.bench / "test.jsonl")
    if args.cmd == "calc":
        rows = [{"id": p["id"], "response": calculator.predict(p)} for p in problems]
        _predict_file(problems, rows, args.out, {"model": "textbook calculator", "tools": "none"})
        print(f"wrote {len(rows)} predictions to {args.out}")
        return 0
    if args.cmd == "run":
        key = args.key_file.read_text().strip()
        rows = modelrun.run(problems, base_url=args.base_url, model=args.model, key=key, openrouter=args.openrouter,
                            workers=args.workers)
        _predict_file(problems, rows, args.out, {"model": args.model, "base_url": args.base_url,
                                                 "system_prompt": modelrun.SYSTEM, "decoding": modelrun.DECODING,
                                                 "tools": "none", "openrouter_provider_pin": args.openrouter})
        errors = sum(1 for r in rows if r.get("error"))
        print(f"wrote {len(rows)} responses to {args.out} ({errors} transport errors)")
        return 0
    if args.cmd == "score":
        recs = _scored(problems, args.preds)
        out = args.preds.with_suffix(".scored.jsonl")
        _write(out, recs)
        summary = scoring.summarise(recs)
        args.preds.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
        print(_table(summary, f"== {args.preds.name} =="))
        return 0
    if args.cmd == "compare":
        print(json.dumps(scoring.compare(_scored(problems, args.a), _scored(problems, args.b)), indent=1))
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
