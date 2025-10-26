from abc import ABC, abstractmethod

class Detector(ABC):
    registry = {}

    def __new__(cls, smell_cls):
        detector_cls = cls.registry.get(smell_cls)
        if detector_cls is None:
            raise ValueError(f"No detector registered for smell {smell_cls}")
        instance = super().__new__(detector_cls)
        instance.__init__(smell_cls)
        return instance

    def __init__(self, smell_cls):
        self.smell_cls = smell_cls

    @abstractmethod
    def detect(self, file):
        """Detect method to be implemented by each specific smell detector"""
        pass

    @classmethod
    def register(cls, smell_class):
        def decorator(subclass):
            cls.registry[smell_class] = subclass
            return subclass
        return decorator