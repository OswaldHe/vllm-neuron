# SPDX-License-Identifier: Apache-2.0
"""Are the ten Block classes each running as one NKI kernel?

The port has 43 blocks but only ten distinct *classes*: two blocks need two
kernels only if something they compute differs, and the per-layer axes in
`config.py` collapse to ten combinations. This derives those classes from the
config rather than hard-coding them, so the check cannot drift from the model,
then asks whether each one has a kernel claiming the whole class.

**Soft until iteration 15, hard from there.** The optimization loop ramps the
target one class an iteration; failing early iterations for kernels that are not
written yet would reject every iteration in the run. So this exits 0 while
reporting progress, and only fails once the run is in its final two iterations
and the remaining classes are the only thing left to do.

How a class is claimed. `model.py` must carry

    FUSED_BLOCK_TYPES: dict[int, str] = {1: "block_type1", ...}

mapping a class id to the module under `kernels/` that runs that whole class.
The mapping is a *claim*, and this script checks that the claim is wired up —
the module exists, is non-trivial, and `model.py` imports it — not that the
arithmetic is right. `verifier` and the `full-model-faithful` judge are what
establish the forward path still computes DeepSeek-V4.1-Flash; a registry entry
pointing at an empty file passes here and fails there.

No device, no checkpoint. Reads source and `.autohelix/history.jsonl`.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
from pathlib import Path

#: From this iteration on, an unfused class fails the constraint rather than
#: just being reported. Matches the `iteration_constraints:` ramp, whose last
#: slot is `from: 15 / to: 16`.
HARD_FROM_ITERATION = 15

#: How many classes the ramp asks for at each iteration. Index is the iteration;
#: anything past the end means all of them. Iterations 0-1 ask for none.
RAMP = [0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

HERE = Path(__file__).resolve()


def _model_dir() -> Path:
    """The port to inspect: `--model-dir` if given, else this file's own.

    The gate runs the worktree's copy of this script so it inspects the
    iteration's `model.py`. A worktree created before this file was committed
    does not have it, which made the constraint fail with "can't open file" —
    so the caller may instead run the main checkout's copy and point it at the
    worktree. Without the override that fallback would survey the wrong tree.
    """
    argv = sys.argv[1:]
    if "--model-dir" in argv:
        return Path(argv[argv.index("--model-dir") + 1]).resolve()
    return HERE.parent.parent


MODEL_DIR = _model_dir()
MODEL_PY = MODEL_DIR / "model.py"
KERNELS = MODEL_DIR / "kernels"


def _config():
    """`config.py` loaded straight off disk.

    By path rather than `import vllm_neuron...`, which would pull in the vLLM
    platform plugin and the Neuron runtime for a file that is only dataclasses.
    """
    name = "_ds41_config_for_fusion_check"
    spec = importlib.util.spec_from_file_location(name, MODEL_DIR / "config.py")
    module = importlib.util.module_from_spec(spec)
    # Registered before exec: `@dataclass` resolves annotations through
    # `sys.modules[cls.__module__]`, and a module absent from there makes
    # `dataclasses._is_type` raise on `None.__dict__`.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.DeepseekV41Config()


def block_classes(cfg) -> list[dict]:
    """The distinct Block classes, largest first, ties by lowest layer id.

    Ordered the same way the loop's ramp is ordered, so class 1 here is the
    class iteration 2 was asked for.
    """
    total = cfg.num_hidden_layers + cfg.num_nextn_predict_layers
    groups: dict[tuple, list[int]] = {}
    for lid in range(total):
        key = (
            "DSparkBlock" if lid >= cfg.num_hidden_layers else "Block",
            cfg.compress_ratios[lid],
            lid in cfg.engram_layer_ids,
            lid in cfg.kv_source_layer_ids,
            lid in cfg.index_source_layer_ids,
            lid == cfg.candidate_source_layer_id,
            lid in cfg.dspark_target_layer_ids,
        )
        groups.setdefault(key, []).append(lid)

    ordered = sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[1][0]))
    out = []
    for index, (key, layers) in enumerate(ordered, start=1):
        flags = dict(
            zip(
                ("engram", "kv_source", "index_source", "candidate_source", "dspark_tap"),
                key[2:],
            )
        )
        out.append({
            "type": index,
            "layers": layers,
            "klass": key[0],
            "compress_ratio": key[1],
            "flags": sorted(k for k, v in flags.items() if v),
        })
    return out


def registry() -> dict[int, str]:
    """`FUSED_BLOCK_TYPES` out of `model.py`, parsed rather than imported.

    Static so that a model.py which cannot import — mid-edit, or needing a
    device — is reported as "no classes fused" instead of crashing the gate.
    """
    try:
        tree = ast.parse(MODEL_PY.read_text())
    except (OSError, SyntaxError) as exc:
        print(f"could not parse model.py: {exc}", file=sys.stderr)
        return {}

    for node in tree.body:
        targets = (
            [node.target] if isinstance(node, ast.AnnAssign) else
            getattr(node, "targets", [])
        )
        if not any(isinstance(t, ast.Name) and t.id == "FUSED_BLOCK_TYPES" for t in targets):
            continue
        try:
            value = ast.literal_eval(node.value)
        except (ValueError, SyntaxError):
            print("FUSED_BLOCK_TYPES is not a literal dict", file=sys.stderr)
            return {}
        if not isinstance(value, dict):
            return {}
        return {int(k): str(v) for k, v in value.items()}
    return {}


def imported_kernels() -> set[str]:
    """Kernel module names `model.py` actually imports.

    From the import statements rather than by searching the text: the registry
    literal contains every module name it claims, so a substring search over
    model.py is satisfied by the claim itself and checks nothing.
    """
    try:
        tree = ast.parse(MODEL_PY.read_text())
    except (OSError, SyntaxError):
        return set()

    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module.endswith("kernels"):
                # from .kernels import block_type1
                found.update(a.name for a in node.names)
            elif ".kernels." in f".{module}.":
                # from .kernels.block_type1 import kernel
                found.add(module.rsplit(".", 1)[-1])
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if ".kernels." in f".{alias.name}.":
                    found.add(alias.name.rsplit(".", 1)[-1])
    return found


def wired(module_name: str, imported: set[str]) -> str | None:
    """Why `module_name` does not count, or None if it does."""
    path = KERNELS / f"{module_name}.py"
    if not path.exists():
        return f"kernels/{module_name}.py does not exist"
    # A stub registered to satisfy the check. 40 lines is well under any real
    # fused block kernel; `legacy/moe_short.py` is hundreds.
    lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
    if len(lines) < 40:
        return f"kernels/{module_name}.py is {len(lines)} lines — a stub, not a block kernel"
    if module_name not in imported:
        return f"model.py never imports kernels/{module_name}"
    return None


def current_iteration() -> int | None:
    """The iteration being gated, from the history the worktree carries.

    History holds one entry an iteration including rejected ones, so the next
    iteration is one past the highest recorded. None when it cannot be read —
    and None is permissive: an unreadable history is an infrastructure problem
    and must not reject an iteration's work.
    """
    for base in (Path.cwd(), MODEL_DIR.parents[2]):
        path = base / ".autohelix" / "history.jsonl"
        if not path.exists():
            continue
        seen = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                seen.append(int(json.loads(line)["iteration"]))
            except (ValueError, KeyError, TypeError):
                continue
        if seen:
            return max(seen) + 1
    return None


def main() -> int:
    """Exit 0 for every iteration before `HARD_FROM_ITERATION`, whatever happens.

    The iteration number is read first and the rest runs inside `survey`, so a
    bug in this file — or a `config.py` an iteration renamed a field in — cannot
    reject work before the final two. A checker that discards an iteration's four
    hours because it crashed is worse than no checker.
    """
    iteration = current_iteration()
    hard = iteration is not None and iteration >= HARD_FROM_ITERATION
    try:
        return survey(iteration, hard)
    except Exception as exc:  # noqa: BLE001 - see the docstring
        print(f"check could not run: {type(exc).__name__}: {exc}", file=sys.stderr)
        if hard:
            print(f"FAIL: iteration {iteration} is gated and this check could not "
                  f"establish that the Block classes are fused. Fix the checker or "
                  f"the config it reads.")
            return 1
        print(f"PASS (advisory until iteration {HARD_FROM_ITERATION}; the check itself "
              f"errored, which is worth fixing before then)")
        return 0


def survey(iteration: int | None, hard: bool) -> int:
    cfg = _config()
    classes = block_classes(cfg)
    claimed = registry()
    imported = imported_kernels()

    fused: list[int] = []
    print(f"{len(classes)} distinct Block class(es) over "
          f"{cfg.num_hidden_layers + cfg.num_nextn_predict_layers} blocks\n")
    for entry in classes:
        tid, layers = entry["type"], entry["layers"]
        shape = f"compress_ratio {entry['compress_ratio']}"
        if entry["flags"]:
            shape += " + " + " + ".join(entry["flags"])
        if entry["klass"] != "Block":
            shape = f"{entry['klass']}, {shape}"
        if len(layers) == 1:
            span = str(layers[0])
        elif layers == list(range(layers[0], layers[-1] + 1)):
            span = f"{layers[0]}-{layers[-1]}"
        else:
            span = ",".join(map(str, layers))

        module = claimed.get(tid)
        if module is None:
            status = "not fused"
        elif (why := wired(module, imported)) is not None:
            status = f"CLAIMED BUT NOT WIRED: {why}"
        else:
            status = f"fused -> kernels/{module}"
            fused.append(tid)
        print(f"  type {tid:<2} {len(layers):2d} block(s)  layers {span:<22} "
              f"{shape:<58} {status}")

    print(f"\n##autohelix[fused_block_types={len(fused)}]")

    required = (
        0 if iteration is None
        else RAMP[iteration] if iteration < len(RAMP) else len(classes)
    )
    missing = [c["type"] for c in classes if c["type"] not in fused][:required]

    where = "unknown iteration" if iteration is None else f"iteration {iteration}"
    print(f"\n{len(fused)} of {len(classes)} fused; {where} asks for {required}")

    if not hard:
        if missing:
            print(f"behind the ramp: type(s) {missing} not fused yet — reported, not gated "
                  f"(hard from iteration {HARD_FROM_ITERATION})")
        print(f"PASS (advisory until iteration {HARD_FROM_ITERATION})")
        return 0

    unfused = [c["type"] for c in classes if c["type"] not in fused]
    if unfused:
        print(f"FAIL: iteration {iteration} is in the final two and type(s) {unfused} are "
              f"still not fused. Each must run as one NKI kernel registered in "
              f"FUSED_BLOCK_TYPES.")
        return 1
    print("PASS: every Block class runs as one NKI kernel")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
