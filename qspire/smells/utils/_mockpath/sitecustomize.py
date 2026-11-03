# Loaded by child Python processes of Q-Spire static mode (see smells/utils/mock_missing.py).
import os as _os
if _os.environ.get("QSPIRE_MOCK_MISSING_MODULES") == "1":
    try:
        import importlib.util as _ilu
        _p = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "mock_missing.py")
        _spec = _ilu.spec_from_file_location("qspire_mock_missing", _p)
        _m = _ilu.module_from_spec(_spec)
        _spec.loader.exec_module(_m)
        _m.install()
    except Exception:
        pass
