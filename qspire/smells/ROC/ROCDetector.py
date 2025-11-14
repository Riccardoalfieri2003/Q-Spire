from smells.utils.OperationCircuitTracker import analyze_quantum_file
from smells.Detector import Detector
from smells.ROC.ROC import ROC
from smells.utils.config_loader import get_detector_option

def create_circuit_batches(operations):
    """
    Convert a sequential list of quantum operations into parallel execution batches.
    
    Each batch contains operations that can be executed in parallel (don't conflict on qubits).
    The batches represent the actual execution order in the quantum circuit.
    
    Args:
        operations: List of operation dictionaries with 'qubits_affected' field
        
    Returns:
        dict: Dictionary where keys are batch numbers and values are lists of operations
    """
    
    if not operations:
        return {}
    
    batches = {}
    batch_number = 1
    
    # Keep track of the last batch where each qubit was used
    qubit_last_batch = {}
    
    # Process each operation in order
    for operation in operations:
        operation_qubits = operation['qubits_affected']
        
        # Find the minimum batch this operation can be placed in
        # It must be after the last batch that used any of its qubits
        min_batch = 1
        for qubit in operation_qubits:
            if qubit in qubit_last_batch:
                min_batch = max(min_batch, qubit_last_batch[qubit] + 1)
        
        # Try to place the operation in the earliest possible batch
        # where it doesn't conflict with other operations in that batch
        placed = False
        current_batch_num = min_batch
        
        while not placed:
            # Check if this batch exists and if there's a conflict
            if current_batch_num not in batches:
                batches[current_batch_num] = []
            
            # Check for conflicts with existing operations in this batch
            conflict = False
            for existing_op in batches[current_batch_num]:
                existing_qubits = set(existing_op['qubits_affected'])
                if set(operation_qubits).intersection(existing_qubits):
                    conflict = True
                    break
            
            if not conflict:
                # Place the operation in this batch
                batches[current_batch_num].append(operation)
                
                # Update the last batch for all affected qubits
                for qubit in operation_qubits:
                    qubit_last_batch[qubit] = current_batch_num
                
                placed = True
            else:
                # Try the next batch
                current_batch_num += 1
    
    # Clean up empty batches and renumber
    final_batches = {}
    batch_counter = 1
    for batch_num in sorted(batches.keys()):
        if batches[batch_num]:  # Only include non-empty batches
            final_batches[batch_counter] = batches[batch_num]
            batch_counter += 1
    
    return final_batches

def print_circuit_batches(batches):
    """Helper function to print the batches in a readable format"""
    for batch_num, operations in batches.items():
        print(f"\nBatch {batch_num}:")
        for op in operations:
            qubits_str = ', '.join(map(str, op['qubits_affected']))
            print(f"  - {op['operation_name']} on qubit(s) [{qubits_str}] (row {op['row']})")

def batch_signature(batch):
    """Canonical, hashable signature for a batch."""
    return tuple(
        (op["operation_name"],
         tuple(op.get("qubits_affected", [])),
         tuple(op.get("clbits_affected", [])))
        for op in batch
    )


def roc_smell_present_subsequence(circuit_batches, min_sub_len=1, return_matches=False):
    """
    Detect repeated *consecutive* subsequences of batches and group repeated occurrences.
    Returns grouped matches:
    (start_batch_index, slice_size, repetition_count, last_batch_index)
    """
    batch_indices = sorted(circuit_batches.keys())
    batch_signatures = [batch_signature(circuit_batches[i]) for i in batch_indices]
    num_batches = len(batch_signatures)

    grouped_matches = []
    i = 0

    while i < num_batches:
        max_possible_slice = (num_batches - i) // 2
        best_slice_size = 0
        repetition_count = 0

        # Try each slice size starting from 1
        for slice_size in range(1, max_possible_slice + 1):
            slice1 = batch_signatures[i : i + slice_size]
            slice2 = batch_signatures[i + slice_size : i + 2 * slice_size]

            if slice1 != slice2:
                continue

            # Found a repeating slice — now count how many consecutive repetitions
            count = 1
            next_start = i + slice_size * 2

            while next_start + slice_size <= num_batches:
                next_slice = batch_signatures[next_start : next_start + slice_size]
                if next_slice != slice1:
                    break
                count += 1
                next_start += slice_size

            # Update best if longer sequence found
            if count > 0:
                best_slice_size = slice_size
                repetition_count = count
                break   # keep original algorithm: stop at the first valid slice size

        if best_slice_size > 0:
            # Skip too short subsequences (not enough operations)
            total_ops_first = sum(len(circuit_batches[batch_indices[i + k]])
                                  for k in range(best_slice_size))
            if total_ops_first >= min_sub_len:
                first_batch = batch_indices[i]
                last_batch = batch_indices[i + best_slice_size * (repetition_count + 1) - 1]
                grouped_matches.append(
                    (first_batch, best_slice_size, repetition_count + 1, last_batch)
                )

            # skip entire repeated region
            i += best_slice_size * (repetition_count + 1)
        else:
            i += 1

    has_smell = len(grouped_matches) > 0
    if return_matches:
        return has_smell, grouped_matches
    return has_smell





@Detector.register(ROC)
class ROCDetector(Detector):

    def __init__(self, smell_cls):
        super().__init__(smell_cls)
        self.circuit_tracker = None
        self.operation_tracker = None
        self.position_tracker = None
    


    def detect(self, file):

        debug=False
    
        smells = []
        circuits = analyze_quantum_file(file)

        min_subcircuit_lenght = get_detector_option("ROC", "min_subcircuit_lenght", fallback=1)

        for circuit in circuits:
            circuit_batches = create_circuit_batches(circuits[circuit])

            if debug:
                import pprint
                pprint.pp(circuit_batches)

            has_smell, matches = roc_smell_present_subsequence(
                circuit_batches,
                min_sub_len=min_subcircuit_lenght,
                return_matches=True
            )

            if matches:
                if debug: print(f"ROC smell detected in circuit {circuit}")
                for m in matches:
                    start_batch = m[0]
                    slice_size = m[1]
                    repetition_count = m[2]
                    last_batch = m[3]

                    if debug: 
                        print(
                            f"\n  🔁 Repeated subsequence of {slice_size} batches "
                            f"(repeated {repetition_count} times — from batch {start_batch} to {last_batch})"
                        )

                    # Extract operations of the base pattern (first slice_size batches)
                    base_pattern_batches = range(start_batch, start_batch + slice_size)
                    base_pattern = []
                    repeated_rows = set()
                    for b in base_pattern_batches:
                        ops = circuit_batches[b]
                        base_pattern.extend(ops)
                        repeated_rows.update(op['row'] for op in ops)

                        if debug: 
                            # Print all operations in the batch
                            print(f"     ▸ Batch {b} contains:")
                            for op in ops:
                                print(
                                    f"        - {op['operation_name']} "
                                    f"(row {op['row']}, col {op['column_start']}–{op['column_end']})"
                                )

                    # Print where it repeats
                    if debug:  print("  ↳ Subsequence repeats at batches:")
                    for r in range(repetition_count - 1):
                        offset = (r + 1) * slice_size
                        repeat_start = start_batch + offset
                        repeat_end = repeat_start + slice_size - 1
                        if debug: print(f"     • repetition #{r+1}: batches {repeat_start} → {repeat_end}")

                    # Create the smell object
                    smell = ROC(
                        operations=base_pattern,
                        repetitions=repetition_count - 1,  # we subtract 1 since the first is original
                        rows=list(sorted(repeated_rows)),
                        circuit_name=circuit
                    )
                    smells.append(smell)

            else:
                if debug: print(f"No ROC smell in circuit {circuit}")

        return smells








                    

        """
        for circuit_name, operations in circuits.items():
            #print(f"\n🔍 Checking circuit: {circuit_name}")
            total_ops = len(operations)
            #print(f"Total operations: {total_ops}")

            divisors = get_divisors(total_ops, min_subcircuit_lenght)
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

        
        min_num_smells = get_detector_option("ROC", "min_num_smells", fallback=1)
        if len(smells)>=min_num_smells: return smells
        else: return []
        """