"""Circuit tracker based on the execution of the analyzed code with the real Qiskit.

Drop-in replacement of OperationCircuitTracker.analyze_quantum_file: it returns
{circuit_name: [operation dicts]} in the same format, so the IM, IdQ, IQ, ROC and
LC detectors are unchanged. The code is executed in a separate process (see
_real_circuit_runner.py) with a timeout, and the result is cached per file so
that the five detectors do not execute the same file five times.
"""
import json
import os
import subprocess
import sys
import threading

from smells.utils import mock_missing

# third-party modules missing in the analysis environment are mocked in child processes
mock_missing.enable_for_children()

_RUNNER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_real_circuit_runner.py")
_MARK = "@@QSPIRE_JSON@@"
_cache = {}
_CACHE_MAX = 256
_lock = threading.Lock()
# Q-Spire analyses all functions of a file in parallel threads; limit the number of
# concurrent runner processes (each imports Qiskit) to avoid exhausting memory.
_slots = threading.BoundedSemaphore(int(os.environ.get("QSPIRE_TRACKER_JOBS", str(os.cpu_count() or 2))))
LAST_ERRORS = {}
_inflight = {}


def _parses(p):
    """False if the script has a syntax error (e.g., `yield` inlined in the main block),
    which only AutoFix repairs."""
    import ast
    try:
        with open(p, encoding="utf-8") as f:
            ast.parse(f.read())
        return True
    except Exception:
        return False


def analyze_quantum_file(input_file, output_file=None, debug=False, timeout=None):
    path = os.path.abspath(input_file)
    # Static mode keeps a copy of each generated script before AutoFix (<script>.prefix):
    # run that copy, since statements that fail are skipped here and AutoFix may have
    # deleted circuit-building code. Rows then refer to the script before AutoFix.
    if os.path.exists(path + ".prefix") and _parses(path + ".prefix"):
        path = path + ".prefix"
    try:
        key = (path, os.path.getmtime(path), os.path.getsize(path))
    except OSError:
        return {}
    # The detectors of a function run in parallel threads and ask for the same file at the
    # same time: compute it once and let the other callers wait for the result.
    while True:
        with _lock:
            if key in _cache:
                return _cache[key]
            ev = _inflight.get(key)
            if ev is None:
                ev = _inflight[key] = threading.Event()
                break
        ev.wait()
    try:
        return _compute(path, key, output_file, debug, timeout)
    finally:
        with _lock:
            _inflight.pop(key, None)
        ev.set()


def _compute(path, key, output_file, debug, timeout):
    timeout = timeout or int(os.environ.get("QSPIRE_TRACKER_TIMEOUT", "180"))
    per_stmt = os.environ.get("QSPIRE_STATEMENT_TIMEOUT", "30")
    circuits, errors = {}, []
    try:
        with _slots:
            proc = subprocess.run([sys.executable, _RUNNER, path, per_stmt], capture_output=True, text=True,
                                  timeout=timeout, cwd=os.path.dirname(path))
        for line in proc.stdout.splitlines()[::-1]:
            if line.startswith(_MARK):
                data = json.loads(line[len(_MARK):])
                circuits, errors = data.get("circuits", {}), data.get("errors", [])
                break
        else:
            errors = [f"runner exit {proc.returncode}: {proc.stderr[-300:]}"]
    except subprocess.TimeoutExpired:
        errors = ["runner timeout"]
    except Exception as e:  # noqa
        errors = [f"runner failure: {type(e).__name__}: {e}"]
    if debug or os.environ.get("QSPIRE_TRACKER_DEBUG"):
        sizes = {k: len(v) for k, v in circuits.items()}
        print(f"[RealCircuitTracker] {path}: circuits {sizes}; errors: {errors[:5]}", file=sys.stderr)
    with _lock:
        if len(_cache) >= _CACHE_MAX:
            _cache.pop(next(iter(_cache)))
        _cache[key] = circuits
        LAST_ERRORS[path] = errors
    if output_file:
        with open(output_file, "w") as f:
            json.dump(circuits, f, indent=2)
    return circuits
