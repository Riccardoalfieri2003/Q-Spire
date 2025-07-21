"""from smells.IM.IMInstrumentation_pre import CircuitAutoTracker,EnhancedQubitTracker

# Track all circuits automatically
circuit_tracker = CircuitAutoTracker()
circuit_tracker.find_circuits_in_globals(globals())  # <- find circuits declared
circuit_tracker.track_all_circuits(globals())        # <- track each found circuit

# Track all operations
operation_tracker = {}
for name, tracker in circuit_tracker.trackers.items():
    operation_tracker[name] = tracker.get_sequential_operations()"""


from smells.IM.IMInstrumentation_pre import CircuitAutoTracker, EnhancedQubitTracker

# Set up circuit tracker
circuit_tracker = CircuitAutoTracker()
circuit_tracker.find_circuits_in_globals(globals())  # Detect circuits
circuit_tracker.track_all_circuits(globals())        # Track operations

# Extract operation sequence
operation_tracker = circuit_tracker.get_all_operations()

# Parse original code for position info
position_tracker = None
if '__ORIGINAL_CODE__' in globals():
    circuit_tracker.find_operations_with_positions(globals()['__ORIGINAL_CODE__'])
    position_tracker = circuit_tracker.get_all_positions()