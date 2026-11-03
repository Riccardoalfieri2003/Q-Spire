# Fixes to Q-Spire static mode (branch `fix/circuit-scope`)

This branch fixes problems found while validating Q-Spire's static-mode detections by hand on a sample of open-source repositories. It starts from commit `ce8c41b`. Every change is marked in the code with `qspire-fix`.

## 1. Problems

| # | Problem | Effect | Regression cases |
|---|---|---|---|
| 1 | The operation tracker (`OperationCircuitTracker.analyze_quantum_file`) rebuilds circuits from the source lines of the generated script, grouped by variable name. The generated script holds the analysed body twice (function copy and `__main__` block) and the bodies of all other functions of the file. | Operations are counted twice. Circuits with the same name in different functions (`qc`, `expected`) are merged into circuits that do not exist. This leads to false ROC, LC, IM, IdQ and IQ. | r01, r02, r04 |
| 2 | `return` is rewritten as `sys.exit(0)`, which also ends the detector thread. | Smells of functions with `return` are lost, or come only from the merged circuits of point 1. | r03 |
| 3 | The regex parsing of gate arguments mistakes non-qubit arguments for qubits and drops register-relative or variable qubits. Examples: the parameter index in `rx(params[2], 0)`, the duration in `delay(200, 0)`, `qr2[1]`, `range(n)`. | Wrong qubits for IdQ, IQ and IM. Operations without qubits all fall in one layer, which inflates the parallelism used by LC. | r05, r06, r08, r11, r25 |
| 4 | Control flow is flattened. The mock circuit has no `__enter__`/`__exit__`, so AutoFix deletes `with qc.if_test(...)` blocks. | Mutually exclusive branches are treated as consecutive operations, or are deleted. | r09, r10 |
| 5 | Sub-circuits appended with `append`/`compose` are not expanded. | Operations inside user-defined gates are missed. | r13 |
| 6 | LPQ flags any call named `transpile`. | Calls to other libraries (qBraid) and to a project's own wrappers are flagged. Calls that pass `initial_layout` through `**kwargs` are also flagged. | r18, r19, r20 |
| 7 | CG flags any `.unitary(...)` call or name `unitary`. | Cirq's `circuit.unitary()`, which returns a matrix, and local functions named `unitary` are flagged. | r15, r16, r17 |
| 8 | NC counts `PassManager.run` as an execution, and ignores parameters bound through primitive PUBs (`sampler.run([(qc, values)])`). | False NC. | r21, r22, r23 |
| 9 | The mock `QuantumCircuit` constructor fails with registers, unknown sizes (`self.n`) or extra arguments. AutoFix then replaces the whole constructor call with an unrelated mock. | Circuits built with registers or with sizes from test attributes are lost. | r24 |
| 10 | AutoFix stops at the first missing third-party module ("Required library ... is not installed"). | Results depend on the packages installed on the machine. Without the project's dependencies, almost no circuit smell and no NC is detected. CG and LPQ are not affected, since they are AST-based. | — |
| 11 | All functions of a file are analysed at once, one thread per function, and every detector re-runs the same script. AutoFix and NC runs have no time limit. | Memory peaks on files with many test functions, and runs can hang on long-running code. | — |
| 12 | When later code fails on the mock circuit (e.g., the circuit is passed to `circuit_to_dag` or a pass manager), AutoFix deletes the whole assignment of the circuit and its uses, replacing it with a generic mock. | The circuit disappears from the script that the detectors analyse. | r27 |
| 13 | When AutoFix fails (e.g., a final `assert` that cannot hold with mocks), it stores `None` as mapping. `get_function_smells` then calls `.items()` on `None` inside a bare `except: pass`. | All smells of the function are lost silently. | r28 |
| 14 | The mocked `self` returns `None` for every method, so `with self.subTest(...)` and `with self.assertRaises(...)` fail, and AutoFix deletes the whole block. | Circuits built inside such blocks, common in test files, are lost. | r26 |
| 15 | The ROC detector returns a flat list of rows, while the row mapping expects tuples. | ROC rows were never mapped to the original file: they are lines of the generated script. | — |
| 16 | NC counts `run()` on any object. When the object is a mock (e.g., a pass manager whose passes come from a missing package), a transpilation is counted as an execution. | False NC. | r30 |
| 17 | Code written for Qiskit < 1.0 uses `qc += other` and `qc1 + qc2`, which no longer exist. | The statement fails and the added part is missing (e.g., two barriers end up next to each other, a false ROC). | r32 |
| 18 | `u1`/`u2`/`u3` were mapped to `p`/`u`/`u`. | `u2` followed by `u3` looked like a repetition (false ROC). | r31 |
| 19 | NC counts `run()` of transpiler passes on DAGs, and ignores parameter values passed to `run()` next to the circuits (V1 primitives, gradients, `parameter_binds`). | False NC. | r33, r34 |
| 20 | The line mapping picks the first line with the same text. | When a statement occurs twice in a function, the reported row is the first occurrence. | — |
| 21 | IM counts a barrier after a measurement as an operation. | A measurement followed only by barriers (a terminal measurement) is reported as IM; QSmell skips barriers. | r35 |
| 22 | NC counts `pass_.run(dag)` when the pass comes from a missing package (a mock). | False NC. | r36 |

The regression suite is in `regression/` (35 cases with expected smells; `regression/known_limitations/` holds one case that is still wrong):

- Original version: 10 of 35 in the last run. The original version is not deterministic on a few cases.
- This branch: 35 of 35.

## 2. Changes

- **`smells/utils/RealCircuitTracker.py` and `smells/utils/_real_circuit_runner.py` (new).** These replace the regex tracker and return the same format, so the IM, IdQ, IQ, ROC and LC detectors are unchanged.
  - The generated script runs in a subprocess with the real Qiskit `QuantumCircuit`. Only circuits created by the analysed body (the `__main__` block) are recorded, one per object, with the source line of each instruction.
  - `return`/`sys.exit` ends the body. Each statement has its own timeout, and an error stops only that statement.
  - Registers grow when an index is out of range (sizes in generated scripts are synthetic). Bits and registers created outside the circuit are added.
  - User-defined gates are expanded. Control-flow operations count as one operation.
  - Aliases are provided for APIs removed in Qiskit 1.0/2.0: `cnot`, `toffoli`, `fredkin`, `u1/u2/u3` and `cu1/cu3` (with the original gate classes, so names are kept), `mct`, `bind_parameters`, `c_if`, `combine`, `extend`, `+` and `+=`.
  - Circuits returned by `compose`, `tensor` and `+` called in the analysed code are recorded as new circuits.
  - Instructions added through `append`, `_append` and the standard-gate methods record their source line.
  - Results are cached per script. Concurrent requests share one run, and at most `QSPIRE_TRACKER_JOBS` runs happen at once.
- **`smells/utils/OperationCircuitTracker.py`.** `analyze_quantum_file` delegates to the new tracker. The old code remains as `analyze_quantum_file_legacy` (`QSPIRE_TRACKER=legacy`).
- **`detection/StaticDetection/StaticCircuit.py`.** The mock circuit is now a context manager and is callable. Its constructor accepts registers, unknown sizes and extra arguments.
- **`smells/LPQ/LPQDetector.py`.** Only Qiskit's `transpile` is flagged:
  - a name imported from `qiskit*`, or an alias of it;
  - an attribute of a `qiskit` module;
  - a star import from `qiskit`, unless the file defines its own `transpile`.

  Calls with `**kwargs` are skipped.
- **`smells/LC/LCDetector.py`.** An unused and slow call (`analyze_circuits_backends_runs`, whose result was overwritten) is removed from the static branch.
- **`smells/IM/IMDetector.py`.** Barriers are skipped: a measurement followed only by barriers is terminal.
- **`smells/CG/CGDetector.py`.** A bare `unitary(...)` counts only if it is imported from Qiskit. `.unitary(...)` needs a matrix and qubits.
- **`smells/utils/RunExecuteParametersCalls.py` (NC).**
  - The script is run statement by statement (see AutoFix below), descending into the `__main__` block.
  - `run()` of a transpiler pass, or on a DAG, is not an execution (also when the pass is a mock: receivers named like `pass_`, or a first argument named like `dag`). Parameter values passed to `run()` next to the circuits (`run(circuits, observables, values)`, `parameter_values=`, `parameter_binds=`) count as a binding.
  - `PassManager.run` is not an execution: neither when the object is a pass manager, nor when the receiver is a name bound to `PassManager(...)`/`generate_preset_pass_manager(...)` in the source, a name such as `pm` or `pass_manager`, or a `PassManager(...).run(...)` chain.
  - PUBs with parameter values count as a binding.
  - The in-process execution stops after `QSPIRE_NC_EXEC_TIMEOUT` seconds.
- **`smells/utils/mock_missing.py` and `smells/utils/_mockpath/sitecustomize.py` (new).**
  - A module that cannot be found is replaced by a permissive mock, in line with the mocking approach of static mode.
  - This applies only when the module is imported by the analysed code. Installed libraries, including their optional imports, and Q-Spire itself are never mocked.
  - It is active in AutoFix runs, in the circuit runner and in the NC execution. `QSPIRE_MOCK_MISSING_MODULES=0` restores the old behaviour.
- **`detection/StaticDetection/StaticMappedDetection.py` and `AutoFix_Mapping.py`.**
  - At most `QSPIRE_FUNCTION_JOBS` functions of a file are analysed at once.
  - Each AutoFix run of a script stops after `QSPIRE_AUTOFIX_RUN_TIMEOUT` seconds. A script that runs that long is treated as fixed.
  - A copy of each generated script is kept before AutoFix (`<script>.prefix`). The circuit tracker runs this copy; a statement that fails is skipped there, so no circuit-building code is deleted. The line mapping is computed on the same copy, instead of racing with AutoFix on the same file. The copy is not used if it has a syntax error that only AutoFix repairs.
  - **AutoFix is skipped** when the new tracker is used (`QSPIRE_AUTOFIX=1` restores it). The detectors run the generated script statement by statement and skip what fails. For NC, an undefined name is bound to a mock and the statement is run again, as AutoFix would do, after discarding the calls recorded in the failed attempt. AutoFix took about 80% of the analysis time, since it re-runs the script after every fix, and it could delete circuit-building code. Without it, the analysis is about four times faster.
  - A failed AutoFix no longer drops the smells of the function. The row mapping of each smell is guarded, and errors are printed instead of being swallowed.
  - ROC rows are mapped to the original file.
  - The line mapping keeps the order of the function body, so a statement that occurs twice is mapped to the right occurrence.
- **`detection/StaticDetection/StaticCircuit.py` (mocked `self`).** Methods of the mocked `self` return an object that can be called, iterated and used as a context manager, and that suppresses exceptions on exit, as `subTest` and `assertRaises` do.

## 3. Environment variables

| Variable | Default | Meaning |
|---|---|---|
| `QSPIRE_TRACKER` | (new tracker) | `legacy` uses the old regex tracker |
| `QSPIRE_TRACKER_TIMEOUT` | 180 | seconds per circuit-runner run |
| `QSPIRE_STATEMENT_TIMEOUT` | 30 | seconds per top-level statement in the runner |
| `QSPIRE_TRACKER_JOBS` | CPU count | concurrent circuit-runner processes |
| `QSPIRE_FUNCTION_JOBS` | CPU count | functions of a file analysed at once |
| `QSPIRE_AUTOFIX` | 0 | 1 runs AutoFix as in the original version (with the new tracker it is skipped) |
| `QSPIRE_AUTOFIX_RUN_TIMEOUT` | 60 | seconds per AutoFix run of a generated script |
| `QSPIRE_NC_EXEC_TIMEOUT` | 60 | seconds for the NC execution of a generated script |
| `QSPIRE_MOCK_MISSING_MODULES` | 1 | mock third-party modules that are not installed |
| `QSPIRE_TRACKER_DEBUG` | unset | print the circuits found for each script |

## 4. Known limitations (unchanged)

- **Module-level code.** Code outside functions is not analysed for the circuit smells and NC (r14). The same holds in the original version.
- **Functions with the same name.** If a file has two functions with the same name (e.g., `setUp` or `test_run` in two test classes), only one generated script is kept, as before.
- **Circuits built outside the analysed function.** A circuit built by a helper in the same module is analysed with the caller. If the helper is a method of `self`, it is not followed, since `self` is a mock.
- **Old code on Qiskit 2.x.** Code written for old Qiskit versions runs on Qiskit 2.x. Removed APIs are covered by the aliases above or by the mocks. A statement that still fails is skipped.

## 5. How to run

```bash
# regression suite: original version (git worktree at ce8c41b) vs this branch
git worktree add ../qspire_orig ce8c41b
python3 regression/run_regression.py <python with qiskit 2.2.3 and qiskit-aer>
# only this branch
REG_ONLY_REAL=1 python3 regression/run_regression.py <python>
```

The CLI is unchanged. Note that `generated_executables/` is created in the working directory, so two runs must not share the same working directory.
