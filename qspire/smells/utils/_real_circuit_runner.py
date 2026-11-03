"""Subprocess runner for RealCircuitTracker.

Executes a Q-Spire generated executable (or any Python file) with the *real*
Qiskit QuantumCircuit class and returns, as JSON on stdout, the operations of
every circuit created while the analyzed code runs.

Why: the previous tracker rebuilt circuits by scanning every source line of the
generated script with regular expressions, grouping lines by variable name. The
generated script contains the analyzed body twice (as a function and inlined in
the main block) and also the bodies of all other functions of the file, so
circuits with the same variable name were merged and operations were counted
twice. Executing the code and reading QuantumCircuit.data avoids both problems,
handles loops, registers, parameters, and control flow, and keeps one circuit
per QuantumCircuit object.

Usage: python _real_circuit_runner.py <file> [per_statement_timeout_s]
Output: a line starting with '@@QSPIRE_JSON@@' followed by the JSON result.
"""
import ast
import builtins
import json
import os
import re
import signal
import sys

MARK = "@@QSPIRE_JSON@@"
_THIS = os.path.abspath(__file__)


class _StatementTimeout(BaseException):
    pass


def _alarm_handler(signum, frame):
    raise _StatementTimeout()


def _max_int(rep):
    """Largest integer index in a qubit/clbit specifier, or None."""
    try:
        import numpy as np  # noqa
        int_types = (int, np.integer)
    except Exception:
        int_types = (int,)
    if isinstance(rep, bool):
        return None
    if isinstance(rep, int_types):
        return int(rep)
    if isinstance(rep, range):
        return max(rep) if len(rep) else None
    if isinstance(rep, (list, tuple)):
        vals = [_max_int(r) for r in rep]
        vals = [v for v in vals if v is not None]
        return max(vals) if vals else None
    return None


def _compat_aliases(QuantumCircuit):
    """Re-add QuantumCircuit methods removed in Qiskit 1.0/2.0 as thin aliases."""
    import math
    from qiskit.circuit import Instruction, InstructionSet

    def alias(name, fn):
        if not hasattr(QuantumCircuit, name):
            setattr(QuantumCircuit, name, fn)

    alias("cnot", lambda self, c, t, *a, **k: self.cx(c, t))
    alias("toffoli", lambda self, c1, c2, t, *a, **k: self.ccx(c1, c2, t))
    alias("fredkin", lambda self, c, t1, t2, *a, **k: self.cswap(c, t1, t2))
    alias("i", lambda self, q: self.id(q))
    # the u1/u2/u3 gate classes still exist: keep their names (ROC compares gate names)
    from qiskit.circuit.library import U1Gate, U2Gate, U3Gate, CU1Gate, CU3Gate
    alias("u1", lambda self, lam, q: self.append(U1Gate(lam), [q]))
    alias("u2", lambda self, phi, lam, q: self.append(U2Gate(phi, lam), [q]))
    alias("u3", lambda self, theta, phi, lam, q: self.append(U3Gate(theta, phi, lam), [q]))
    alias("cu1", lambda self, lam, c, t, *a, **k: self.append(CU1Gate(lam), [c, t]))
    alias("cu3", lambda self, theta, phi, lam, c, t, *a, **k: self.append(CU3Gate(theta, phi, lam), [c, t]))
    alias("mct", lambda self, controls, target, *a, **k: self.mcx(controls, target))
    alias("bind_parameters", lambda self, values: self.assign_parameters(values))
    alias("snapshot", lambda self, *a, **k: None)
    def _fit(self, other):
        # old `+` matched registers by name; here circuits are joined qubit by qubit, so
        # grow the left circuit when the right one is wider
        from qiskit.circuit import QuantumRegister, ClassicalRegister
        if other.num_qubits > self.num_qubits:
            self.add_register(QuantumRegister(other.num_qubits - self.num_qubits, f"_qsadd{len(self.qregs)}"))
        if other.num_clbits > self.num_clbits:
            self.add_register(ClassicalRegister(other.num_clbits - self.num_clbits, f"_qsaddc{len(self.cregs)}"))

    def _iadd(self, other):
        _fit(self, other)
        self.compose(other, inplace=True)
        return self

    def _add(self, other):
        return _iadd(self.copy(), other)

    alias("combine", _add)
    alias("extend", _iadd)
    # `qc + other` and `qc += other` were removed in Qiskit 1.0
    if not hasattr(QuantumCircuit, "__add__"):
        QuantumCircuit.__add__ = _add
    if not hasattr(QuantumCircuit, "__iadd__"):
        QuantumCircuit.__iadd__ = _iadd
    # c_if was removed in Qiskit 2.0: keep the operation, drop the classical condition
    for cls in (InstructionSet, Instruction):
        if not hasattr(cls, "c_if"):
            cls.c_if = lambda self, *a, **k: self


def main():
    script = os.path.abspath(sys.argv[1])
    per_stmt_timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (6 * 1024 ** 3, 6 * 1024 ** 3))
    except Exception:
        pass

    from qiskit.circuit import QuantumCircuit, QuantumRegister, ClassicalRegister, Gate, Instruction
    from qiskit.circuit.exceptions import CircuitError

    src = open(script, encoding="utf-8").read()
    src_lines = src.split("\n")
    # Keep the real QuantumCircuit: neutralize the mock alias of the generated scripts.
    src_exec = re.sub(r"(?m)^QuantumCircuit[ \t]*=[ \t]*MockQuantumCircuit[ \t]*$",
                      "pass  # qspire-fix: real QuantumCircuit", src)
    # When AutoFix replaced a failing constructor call with a mock instance, use an empty
    # real circuit instead: it grows on first use, so the following operations are kept.
    src_exec = re.sub(r"(?m)^([ \t]*)([\w.]+)[ \t]*=[ \t]*MockQuantumCircuit\(\)[ \t]*# mock instance of QuantumCircuit[ \t]*$",
                      r"\1\2 = QuantumCircuit()  # qspire-fix: real circuit", src_exec)
    try:
        tree = ast.parse(src_exec)
    except SyntaxError as e:
        print(MARK + json.dumps({"circuits": {}, "errors": [f"SyntaxError: {e}"]}))
        return

    def is_main_if(node):
        if not isinstance(node, ast.If):
            return False
        t = node.test
        return (isinstance(t, ast.Compare) and isinstance(t.left, ast.Name) and t.left.id == "__name__"
                and len(t.comparators) == 1 and isinstance(t.comparators[0], ast.Constant)
                and t.comparators[0].value == "__main__")

    pre, main_stmts = [], None
    for node in tree.body:
        if main_stmts is None and is_main_if(node):
            main_stmts = list(node.body)
        else:
            pre.append(node)
    if main_stmts is None:  # plain script: the whole module is the analyzed code
        main_stmts, pre = pre, [n for n in pre if isinstance(n, (ast.Import, ast.ImportFrom))]
        main_stmts = [n for n in main_stmts if not isinstance(n, (ast.Import, ast.ImportFrom))]

    state = {"main": False}
    registry = []          # (circuit, creation line)
    lines_of = {}          # id(circuit) -> {instruction index: line}

    def script_line(depth=2):
        f = sys._getframe(depth)
        while f is not None:
            if os.path.abspath(f.f_code.co_filename) == script:
                return f.f_lineno
            f = f.f_back
        return None

    # --- compatibility with APIs removed in Qiskit 1.0/2.0 ----------------------------
    # The analyzed projects target several Qiskit versions; without these aliases a call
    # such as qc.cnot(0, 1) would stop the statement (and the rest of a loop) here.
    _compat_aliases(QuantumCircuit)

    # --- register circuits created by the analyzed code -----------------------------
    orig_init = QuantumCircuit.__init__
    try:
        import numpy as np  # noqa
        _int_types = (int, np.integer)
    except Exception:
        _int_types = (int,)
    from qiskit.circuit import Register, Bit

    def _usable(x):
        if isinstance(x, bool):
            return False
        if isinstance(x, _int_types) or isinstance(x, Register):
            return True
        return isinstance(x, (list, tuple)) and all(isinstance(b, Bit) for b in x)

    def patched_init(self, *a, **k):
        try:
            orig_init(self, *a, **k)
        except Exception:
            # Sizes that are unknown in the generated script (mocked self attributes,
            # unresolved helpers) become 0; registers then grow on first use (see below).
            a2 = [x if _usable(x) else 0 for x in a]
            k2 = {kk: vv for kk, vv in k.items()
                  if (kk == "name" and isinstance(vv, str)) or (kk == "global_phase" and isinstance(vv, (int, float)))}
            try:
                orig_init(self, *a2, **k2)
            except Exception:
                orig_init(self)
        if state["main"]:
            f = sys._getframe(1)
            if os.path.abspath(f.f_code.co_filename) == script:
                registry.append((self, f.f_lineno))
    QuantumCircuit.__init__ = patched_init

    # --- grow circuits when synthetic sizes are too small -----------------------------
    from qiskit.circuit import Qubit, Clbit

    def _foreign(self, rep, kind):
        """Registers and bits in rep that the circuit does not contain."""
        bit_cls = Qubit if kind == "q" else Clbit
        reg_cls = QuantumRegister if kind == "q" else ClassicalRegister
        regs, bits = [], []
        stack = [rep]
        while stack:
            x = stack.pop()
            if isinstance(x, reg_cls):
                if x not in (self.qregs if kind == "q" else self.cregs):
                    regs.append(x)
            elif isinstance(x, bit_cls):
                try:
                    self.find_bit(x)
                except Exception:
                    bits.append(x)
            elif isinstance(x, (list, tuple)):
                stack.extend(x)
        return regs, bits

    def make_conv(orig, kind):
        def conv(self, rep):
            try:
                return orig(self, rep)
            except CircuitError:
                idx = _max_int(rep)
                size = self.num_qubits if kind == "q" else self.num_clbits
                if idx is not None and size <= idx < 4096:
                    reg_cls = QuantumRegister if kind == "q" else ClassicalRegister
                    self.add_register(reg_cls(idx + 1 - size, f"_qsx{kind}{len(self.qregs) + len(self.cregs)}"))
                    return orig(self, rep)
                # Bits or registers created outside the circuit (e.g., the circuit was built
                # with an unknown size): add them, so the operation is kept.
                regs, bits = _foreign(self, rep, kind)
                if regs or bits:
                    for r in regs:
                        try:
                            self.add_register(r)
                        except Exception:
                            bits.extend(b for b in r if b not in bits)
                    new = []
                    for b in bits:
                        try:
                            self.find_bit(b)
                        except Exception:
                            new.append(b)
                    if new:
                        self.add_bits(new)
                    return orig(self, rep)
                raise
        return conv
    QuantumCircuit._qbit_argument_conversion = make_conv(QuantumCircuit._qbit_argument_conversion, "q")
    QuantumCircuit._cbit_argument_conversion = make_conv(QuantumCircuit._cbit_argument_conversion, "c")

    # --- remember the source line of each appended instruction ------------------------
    def wrap_append(meth):
        def w(self, *a, **k):
            try:
                n0 = len(self._data)
            except Exception:
                n0 = None
            r = meth(self, *a, **k)
            try:
                n1 = len(self._data)
                line = script_line(2)
                if n0 is not None and line is not None and n1 > n0:
                    d = lines_of.setdefault(id(self), {})
                    for i in range(n0, n1):
                        d[i] = line
            except Exception:
                pass
            return r
        return w
    QuantumCircuit.append = wrap_append(QuantumCircuit.append)
    QuantumCircuit._append_standard_gate = wrap_append(QuantumCircuit._append_standard_gate)
    QuantumCircuit._append = wrap_append(QuantumCircuit._append)  # e.g. barrier(), compose()

    # --- circuits returned by combining circuits in the analyzed code -------------------
    # qc3 = qc1.compose(qc2), qc1 + qc2, qc1.tensor(qc2) build a new circuit in the function
    def wrap_new(meth):
        def w(self, *a, **k):
            r = meth(self, *a, **k)
            try:
                if state["main"] and isinstance(r, QuantumCircuit) and r is not self:
                    # only direct calls from the analyzed code (skipping this runner's
                    # own helpers, e.g. the `+` alias), not calls inside library code
                    f = sys._getframe(1)
                    while f is not None and os.path.abspath(f.f_code.co_filename) == _THIS:
                        f = f.f_back
                    if (f is not None and os.path.abspath(f.f_code.co_filename) == script
                            and all(r is not c for c, _ in registry)):
                        registry.append((r, f.f_lineno))
            except Exception:
                pass
            return r
        return w
    QuantumCircuit.compose = wrap_new(QuantumCircuit.compose)
    QuantumCircuit.tensor = wrap_new(QuantumCircuit.tensor)
    if hasattr(QuantumCircuit, "__add__"):
        QuantumCircuit.__add__ = wrap_new(QuantumCircuit.__add__)

    # --- execute ----------------------------------------------------------------------
    ns = {"__name__": "__main__", "__file__": script, "__builtins__": builtins}
    errors = []
    try:
        signal.signal(signal.SIGALRM, _alarm_handler)
        use_alarm = True
    except Exception:
        use_alarm = False

    def run(node):
        code = compile(ast.Module(body=[node], type_ignores=[]), script, "exec")
        if use_alarm:
            signal.alarm(per_stmt_timeout)
        try:
            exec(code, ns)
        finally:
            if use_alarm:
                signal.alarm(0)

    for node in pre:
        try:
            run(node)
        except BaseException as e:  # noqa
            errors.append(f"pre l.{getattr(node, 'lineno', '?')}: {type(e).__name__}")
    state["main"] = True
    for node in main_stmts:
        try:
            run(node)
        except SystemExit:
            break  # a `return` of the analyzed function (rewritten as sys.exit by Q-Spire)
        except _StatementTimeout:
            errors.append(f"main l.{node.lineno}: timeout")
        except BaseException as e:  # noqa
            errors.append(f"main l.{node.lineno}: {type(e).__name__}: {str(e)[:120]}")
    state["main"] = False

    # --- export circuits --------------------------------------------------------------
    def expandable(op):
        return type(op) in (Gate, Instruction) and getattr(op, "definition", None) is not None

    def ops_of(circ, inst_iter, qmap, cmap, line_for, depth=0):
        out = []
        for i, inst in inst_iter:
            op = inst.operation
            qs = [qmap(q) for q in inst.qubits]
            cs = [cmap(c) for c in inst.clbits]
            if depth < 10 and expandable(op):
                d = op.definition
                sub_q = {id(b): qs[j] for j, b in enumerate(d.qubits)}
                sub_c = {id(b): cs[j] for j, b in enumerate(d.clbits)}
                out.extend(ops_of(d, enumerate(d.data), lambda b: sub_q[id(b)], lambda b: sub_c[id(b)],
                                  lambda _i, _l=line_for(i): _l, depth + 1))
                continue
            line = line_for(i)
            text = src_lines[line - 1] if line and 0 < line <= len(src_lines) else ""
            out.append({
                "operation_name": op.name,
                "qubits_affected": qs,
                "clbits_affected": cs,
                "row": line,
                "column_start": len(text) - len(text.lstrip()),
                "column_end": len(text.rstrip()),
                "source_line": text.strip(),
            })
        return out

    names = {}
    for k, v in ns.items():
        if isinstance(v, QuantumCircuit) and id(v) not in names:
            names[id(v)] = k
    result, seen = {}, set()
    for circ, cline in registry:
        if id(circ) in seen:
            continue
        seen.add(id(circ))
        try:
            lmap = lines_of.get(id(circ), {})
            ops = ops_of(circ, enumerate(circ.data), lambda b: circ.find_bit(b).index,
                         lambda b: circ.find_bit(b).index, lambda i: lmap.get(i, cline))
        except Exception as e:  # noqa
            errors.append(f"export: {type(e).__name__}")
            continue
        if not ops:
            continue
        name = names.get(id(circ))
        if not name:
            m = re.match(r"\s*(?:self\.)?(\w+)\s*(?::[^=]+)?=", src_lines[cline - 1]) if 0 < cline <= len(src_lines) else None
            name = m.group(1) if m else "circuit"
        key, n = name, 2
        while key in result:
            key, n = f"{name}#{n}", n + 1
        result[key] = ops

    sys.stdout.flush()
    print(MARK + json.dumps({"circuits": result, "errors": errors}))


if __name__ == "__main__":
    main()
