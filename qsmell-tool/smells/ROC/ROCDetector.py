from smells.utils.OperationCircuitTracker import analyze_quantum_file
from smells.Detector import Detector
from smells.ROC.ROC import ROC

@Detector.register(ROC)
class ROCDetector(Detector):

    def __init__(self, smell_cls):
        super().__init__(smell_cls)
        self.circuit_tracker = None
        self.operation_tracker = None
        self.position_tracker = None
    


    from math import gcd
    from functools import reduce

    def detect(self, file):

        def get_divisors(n, threshold):
            return [i for i in range(1, n + 1) if n % i == 0 and i >= threshold and i<n]

        def extract_pattern(ops):
            return [(op['operation_name'], tuple(op['qubits_affected'])) for op in ops]
    
        smells = []
        circuits = analyze_quantum_file(file)

        threshold = 2  # minimum pattern size to consider

        for circuit_name, operations in circuits.items():
            #print(f"\n🔍 Checking circuit: {circuit_name}")
            total_ops = len(operations)
            #print(f"Total operations: {total_ops}")

            if total_ops < threshold * 2:
                #print("⏭️  Skipped — not enough operations")
                continue

            divisors = get_divisors(total_ops, threshold)
            #print(f"Valid divisors (≥ {threshold}): {divisors}")

            for d in divisors:
                chunk_count = total_ops // d
                #print(f"\n➡️  Trying divider {d} ({chunk_count} chunks of size {d})")

                chunks = [operations[i * d:(i + 1) * d] for i in range(chunk_count)]

                #print("First chunk pattern:")
                base_pattern = extract_pattern(chunks[0])
                #print(base_pattern)

                is_repeated = True
                repeated_rows = []

                for i in range(1, chunk_count):
                    current_chunk = chunks[i]
                    current_pattern = extract_pattern(current_chunk)
                    #print(f"\nComparing chunk {i}:")
                    #print(current_pattern)

                    if current_pattern != base_pattern:
                        #print("❌ Pattern mismatch found — aborting this divider.")
                        is_repeated = False
                        break
                    else:
                        #("✅ Pattern matches.")
                        chunk_rows = tuple(op['row'] for op in current_chunk)
                        repeated_rows.append(chunk_rows)

                if is_repeated:
                    #print("🎯 Repeated pattern detected!")
                    smell = ROC(
                        operations=base_pattern,
                        repetitions=chunk_count - 1,
                        rows=repeated_rows,
                        circuit_name=circuit_name
                    )
                    smells.append(smell)
                    break  # Only detect one ROC per circuit

        return smells