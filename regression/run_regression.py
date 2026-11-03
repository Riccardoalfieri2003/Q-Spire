"""Run Q-Spire static mode on the regression cases with the original (legacy) and the fixed version."""
import csv, json, os, subprocess, sys, glob, concurrent.futures as cf
QS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "qspire")
# baseline: the original Q-Spire (commit ce8c41b) checked out as a git worktree
ORIG = os.environ.get("QSPIRE_ORIG", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "qspire_orig", "qspire"))
QS = os.environ.get("QSPIRE_FIXED", QS)
ROOT = {"legacy": ORIG, "real": QS}
VERSIONS = ("real",) if os.environ.get("REG_ONLY_REAL") else ("legacy", "real")
PY = sys.argv[1] if len(sys.argv) > 1 else sys.executable
CASES = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "cases", "*.py")))
EXP = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "expected.json")))
SMELLS = {"IM", "IdQ", "IQ", "ROC", "LC", "CG", "LPQ", "NC"}

def run(case, tracker):
    name = os.path.basename(case)[:-3]
    out = os.path.join(os.environ.get("REG_OUT", "/tmp/qsreg"), tracker, name)
    subprocess.run(["rm", "-rf", out])
    env = dict(os.environ)
    code = f"import sys; sys.argv=['qspire','-static',{case!r},{out!r}]\nfrom util.CLIModule import qspire; qspire()"
    subprocess.run([PY, "-c", code], cwd=os.path.abspath(ROOT[tracker]), env=env, capture_output=True, timeout=600)
    rows = []
    for f in glob.glob(out + "/.*.csv") + glob.glob(out + "/*.csv"):
        rows += [(r["type"], r.get("qubit", "") or "") for r in csv.DictReader(open(f))]
    got = sorted({(t, q.replace(".0", "")) for t, q in rows if t in SMELLS})
    counts = {}
    for t, _ in rows:
        if t in SMELLS:
            counts[t] = counts.get(t, 0) + 1
    print(name, tracker, "instances:", counts, flush=True)
    return name, tracker, got

jobs = [(c, t) for c in CASES for t in VERSIONS]
res = {}
with cf.ThreadPoolExecutor(1) as ex:
    for name, tracker, got in ex.map(lambda a: run(*a), jobs):
        res[(name, tracker)] = got
ok = {t: 0 for t in VERSIONS}
for c in CASES:
    name = os.path.basename(c)[:-3]
    exp = sorted(tuple(x) for x in EXP[name])
    line = [name]
    for t in VERSIONS:
        got = res[(name, t)]
        good = got == exp
        ok[t] += good
        line.append(f"{t}: {'OK ' if good else 'BAD'} {got}")
    print(" | ".join(line), "| expected", exp)
print("passed:", ok, "of", len(CASES))
