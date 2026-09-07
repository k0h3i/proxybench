"""Compare reference-selected fields and count answered scalar slots."""

import json

from proxybench.normalization.values import ENUMS, text
from proxybench.schemas.records import FIELD_TYPES


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def project(field, kind):
    result = {k: field[k] for k in ("availability", "origin")}
    value = field["value"]
    if field["availability"] != "PRESENT":
        result["value"] = value
    elif isinstance(kind, str):
        result["value"] = value
        if kind in ENUMS and value == "OTHER":
            result["raw_text"] = text(field["raw_text"])
    elif isinstance(kind, dict):
        result["value"] = {k: project(value[k], t) for k, t in kind.items()}
    else:
        member = kind[0]
        children = [({k: project(v[k], t) for k, t in member.items()} if isinstance(member, dict)
                     else project(v, member)) for v in value]
        result["value"] = sorted(children, key=canonical)
    return result


def primary_paths(record):
    paths = []

    def visit(field, kind, path):
        paths.append(path)
        if field["availability"] != "PRESENT" or isinstance(kind, str):
            return
        if isinstance(kind, dict):
            for key, child in kind.items():
                visit(field["value"][key], child, path + "/value/" + key)
        else:
            for i, value in enumerate(field["value"]):
                if isinstance(kind[0], dict):
                    for key, child in kind[0].items():
                        visit(value[key], child, f"{path}/value/{i}/{key}")
                else:
                    visit(value, kind[0], f"{path}/value/{i}")

    for key, kind in FIELD_TYPES.items():
        if key == "participation" and record["fields"][key]["origin"] == "DERIVED":
            continue
        visit(record["fields"][key], kind, "/fields/" + key)
    return paths


def compare_fields(reference, prediction):
    return {key: project(reference["fields"][key], kind) == project(prediction["fields"][key], kind)
            for key, kind in FIELD_TYPES.items()
            if not (key == "participation" and reference["fields"][key]["origin"] == "DERIVED")}


def maximum_assignment(weights):
    """Return maximum total weight with a polynomial rectangular assignment."""
    if not weights or not weights[0]:
        return 0
    # Hungarian algorithm with zero-weight padding for unmatched reference slots.
    n, m = len(weights), max(len(weights), len(weights[0]))
    costs = [[-v for v in row] + [0] * (m - len(row)) for row in weights]
    u, v, p, way = [0] * (n + 1), [0] * (m + 1), [0] * (m + 1), [0] * (m + 1)
    for i in range(1, n + 1):
        p[0], j0 = i, 0
        minimum, used = [float("inf")] * (m + 1), [False] * (m + 1)
        while True:
            used[j0] = True
            i0, delta, j1 = p[j0], float("inf"), 0
            for j in range(1, m + 1):
                if not used[j]:
                    current = costs[i0 - 1][j - 1] - u[i0] - v[j]
                    if current < minimum[j]:
                        minimum[j], way[j] = current, j0
                    if minimum[j] < delta:
                        delta, j1 = minimum[j], j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minimum[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return sum(-costs[p[j] - 1][j - 1] for j in range(1, m + 1) if p[j])


def leaf_count(field, kind):
    if field["availability"] != "PRESENT" or field["origin"] != "EXTRACTED":
        return 0
    if isinstance(kind, str):
        return 1
    if isinstance(kind, dict):
        return sum(leaf_count(field["value"][k], t) for k, t in kind.items())
    return sum(member_count(v, kind[0]) for v in field["value"])


def member_count(value, kind):
    if isinstance(kind, dict):
        return sum(leaf_count(value[k], t) for k, t in kind.items())
    return leaf_count(value, kind)


def answered(reference, prediction, kind):
    if reference["availability"] != "PRESENT" or reference["origin"] != "EXTRACTED":
        return 0
    if prediction["availability"] != "PRESENT":
        return 0
    if isinstance(kind, str):
        return 1
    if isinstance(kind, dict):
        return sum(answered(reference["value"][k], prediction["value"][k], t) for k, t in kind.items())
    member = kind[0]
    def member_projection(value):
        return ({k: project(value[k], t) for k, t in member.items()} if isinstance(member, dict)
                else project(value, member))
    ordered = sorted(reference["value"], key=lambda value: canonical(member_projection(value)))
    weights = [[(sum(answered(r[k], p[k], t) for k, t in member.items()) if isinstance(member, dict)
                 else answered(r, p, member)) for p in prediction["value"]] for r in ordered]
    return maximum_assignment(weights)


def coverage(reference, prediction=None):
    total = sum(leaf_count(reference["fields"][k], t) for k, t in FIELD_TYPES.items())
    count = sum(answered(reference["fields"][k], prediction["fields"][k], t)
                for k, t in FIELD_TYPES.items()) if prediction else 0
    return {"answered": count, "recoverable": total, "rate": count / total if total else None}
