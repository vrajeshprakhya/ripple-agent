"""Turning a circuit plus a measurement spec into one verified example, and splitting without leaks."""
from __future__ import annotations

import hashlib
import json

from . import measure, mna
from .measure import DatasetError

PROMPT = """You are given a SPICE netlist.

```
{netlist}
```

Question: {question}
Reply with the final answer only, as a number with its SI unit."""


def signature(elements: list[mna.Element]) -> str:
    """A topology fingerprint that ignores component values, names and node names.

    Weisfeiler-Lehman style: nodes and elements are labelled, then re-labelled three times from
    their neighbours' labels, and the sorted labels are hashed. Two circuits share a signature
    when they have the same structure, so splitting by signature keeps a topology out of the test
    set if any circuit of that shape was used for training.
    """
    def h(*parts) -> str:
        return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:10]

    node_label = {n: ("gnd" if n == "0" else "n") for e in elements for n in (e.n1, e.n2)}
    elem_label: dict[str, str] = {}
    for _ in range(3):
        elem_label = {}
        for e in elements:
            a, b = node_label[e.n1], node_label[e.n2]
            pair = (a, b) if e.kind in "vi" else tuple(sorted((a, b)))  # sources are polarised, passives are not
            elem_label[e.name] = h(e.kind, "ac" if e.ac else "dc", *pair)
        incident: dict[str, list[str]] = {n: [] for n in node_label}
        for e in elements:
            incident[e.n1].append(elem_label[e.name] + ("+" if e.kind in "vi" else ""))
            incident[e.n2].append(elem_label[e.name] + ("-" if e.kind in "vi" else ""))
        node_label = {n: h(node_label[n], *sorted(incident[n])) for n in node_label}
    return h(*sorted(node_label.values()), *sorted(elem_label.values()))


def make_example(netlist: str, spec: dict, *, family: str, source: str, analytic: float | None = None,
                 rationale: str | None = None) -> dict:
    """Verify a circuit and a measurement with two solvers and return the example, or raise DatasetError."""
    netlist = "\n".join(line.rstrip() for line in netlist.strip().splitlines()) + "\n"
    try:
        elements = mna.parse(netlist)
    except mna.MnaError as e:
        raise DatasetError(f"unsupported netlist: {e}") from e
    if len(elements) > 20:
        raise DatasetError("more than 20 elements")
    spec = measure.validate_spec(spec, elements)
    kind = spec["kind"]

    v_ng = measure.ngspice_value(netlist, spec)
    v_mna = measure.mna_value(elements, spec)
    if not measure.agree(kind, v_ng, v_mna):
        raise DatasetError(f"solvers disagree: ngspice {v_ng:.6g} vs independent {v_mna:.6g}")
    if analytic is not None and not measure.agree(kind, v_ng, analytic):
        raise DatasetError(f"analytic formula disagrees with the solvers: {analytic:.6g} vs {v_ng:.6g}")
    if not measure.sane(kind, v_ng):
        raise DatasetError(f"answer {v_ng:.6g} is outside the sane range for {kind}")

    unit = measure.UNITS[kind]
    question = measure.render_question(spec)
    text = measure.format_quantity(v_ng, unit)
    ident = hashlib.sha1((netlist + json.dumps(spec, sort_keys=True)).encode()).hexdigest()[:12]
    ex = {
        "id": ident, "task": "predict", "family": family, "source": source, "topology": signature(elements),
        "netlist": netlist, "measurement": spec, "question": question,
        "prompt": PROMPT.format(netlist=netlist.strip(), question=question),
        "completion": text,
        "answer": {"value": v_ng, "unit": unit, "text": text},
        "verification": {"ngspice": v_ng, "independent_solver": v_mna,
                         **({"analytic": analytic} if analytic is not None else {})},
    }
    if rationale:
        ex["rationale"] = rationale
    return ex


def split_groups(examples: list[dict], sizes: dict[str, int], key: str, rng) -> dict[str, list[dict]]:
    """Assign whole groups (same `key`) to splits so no group straddles two of them.

    Fills the splits in order of how far each is from its target; a group is never cut, so a split
    can overshoot by one group. Groups left over once every split is full are dropped.
    """
    groups: dict[str, list[dict]] = {}
    for ex in examples:
        groups.setdefault(ex[key], []).append(ex)
    order = list(groups)
    rng.shuffle(order)
    out: dict[str, list[dict]] = {name: [] for name in sizes}
    for g in order:
        pending = [n for n in sizes if len(out[n]) < sizes[n]]
        if not pending:
            break
        out[min(pending, key=lambda n: len(out[n]) / sizes[n])].extend(groups[g])
    return out
