"""Mocks for third-party modules that are not installed in the analysis environment.

Static mode runs generated scripts that keep the imports of the analyzed file. Without
the analyzed project's dependencies, the first missing import used to stop AutoFix
("Required library ... is not installed") and the execution-based detectors. With this
hook, a module that no other finder can locate is replaced by a permissive mock module,
in line with the mocking approach of static mode. Installed packages (e.g., Qiskit) are
always loaded normally, since the hook is consulted only after all other finders.

Enabled in child processes through sitecustomize (see _mockpath/) when the environment
variable QSPIRE_MOCK_MISSING_MODULES is 1 (the default set by RealCircuitTracker).
"""
import builtins
import contextlib
import importlib.abc
import importlib.machinery
import sys
import threading
import types


def _is_decorated_target(a, k):
    return len(a) == 1 and not k and (isinstance(a[0], type) or isinstance(a[0], types.FunctionType))


class Mock:
    """Permissive stand-in object: attributes, calls, items, iteration, context manager."""

    def __init__(self, name="mock"):
        object.__setattr__(self, "_qs_name", name)

    def __getattr__(self, k):
        if k.startswith("__") and k.endswith("__"):
            raise AttributeError(k)
        return Mock(f"{self._qs_name}.{k}")

    def __setattr__(self, k, v):
        object.__setattr__(self, k, v)

    def __call__(self, *a, **k):
        if _is_decorated_target(a, k):  # used as a decorator: keep the decorated object
            return a[0]
        return Mock(f"{self._qs_name}()")

    def __getitem__(self, k):
        return Mock(f"{self._qs_name}[]")

    def __setitem__(self, k, v):
        pass

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def __bool__(self):
        return True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def __mro_entries__(self, bases):  # class X(MockBase): ... -> plain class
        return (object,)

    def __repr__(self):
        return f"<qspire mock {self._qs_name}>"

    def __eq__(self, other):
        return self is other

    def __hash__(self):
        return id(self)

    def _op(self, *a):
        return self

    __add__ = __radd__ = __sub__ = __rsub__ = __mul__ = __rmul__ = __truediv__ = __rtruediv__ = _op
    __floordiv__ = __rfloordiv__ = __mod__ = __pow__ = __neg__ = __pos__ = __abs__ = _op
    __matmul__ = __rmatmul__ = __and__ = __or__ = __xor__ = _op


class MockModule(types.ModuleType):
    def __init__(self, name):
        super().__init__(name)
        self.__path__ = []
        self.__all__ = []
        self.__qspire_mock__ = True

    def __getattr__(self, k):
        if k.startswith("__") and k.endswith("__"):
            raise AttributeError(k)
        return Mock(f"{self.__name__}.{k}")

    def __call__(self, *a, **k):  # e.g. `from qiskit import execute` resolved as a submodule
        if _is_decorated_target(a, k):
            return a[0]
        return Mock(f"{self.__name__}()")


class _Loader(importlib.abc.Loader):
    def create_module(self, spec):
        return MockModule(spec.name)

    def exec_module(self, module):
        pass


class _Finder(importlib.abc.MetaPathFinder):
    # never mock these: Q-Spire and the standard tooling it needs
    _never = {"smells", "detection", "util", "qspire", "_pytest", "pkg_resources", "setuptools", "pip"}

    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in self._never or name.split(".")[0] in sys.stdlib_module_names:
            return None
        # Mock only imports made by the analyzed code (generated script or project files),
        # never optional imports of installed libraries (e.g., Qiskit's lazy optional
        # dependencies), whose behavior must not change.
        f = sys._getframe(1)
        while f is not None and (f.f_code.co_filename.startswith("<") or "importlib" in f.f_code.co_filename
                                 or f.f_code.co_filename == __file__):
            f = f.f_back
        fn = f.f_code.co_filename if f is not None else ""
        if not fn or fn.startswith(_LIB_PREFIXES) or "site-packages" in fn or "dist-packages" in fn:
            return None
        if fn.startswith(_QSPIRE_ROOT) and "generated_executables" not in fn:
            return None  # Q-Spire's own code
        return importlib.machinery.ModuleSpec(name, _Loader(), is_package=True)


_LIB_PREFIXES = tuple({sys.prefix, sys.base_prefix, sys.exec_prefix})
import os as _os
_QSPIRE_ROOT = _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))) + _os.sep
_FINDER = _Finder()
_lock = threading.Lock()


def install():
    with _lock:
        if _FINDER not in sys.meta_path:
            sys.meta_path.append(_FINDER)


def uninstall():
    with _lock:
        if _FINDER in sys.meta_path:
            sys.meta_path.remove(_FINDER)


@contextlib.contextmanager
def active():
    installed_here = _FINDER not in sys.meta_path
    install()
    try:
        yield
    finally:
        if installed_here:
            uninstall()


def enable_for_children():
    """Make child Python processes (AutoFix runs, circuit runner) load this hook at startup."""
    import os
    os.environ.setdefault("QSPIRE_MOCK_MISSING_MODULES", "1")
    if os.environ["QSPIRE_MOCK_MISSING_MODULES"] != "1":
        return
    mp = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_mockpath")
    paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if mp not in paths:
        os.environ["PYTHONPATH"] = os.pathsep.join([mp] + [p for p in paths if p])


def enabled():
    import os
    return os.environ.get("QSPIRE_MOCK_MISSING_MODULES", "1") == "1"
