from smells.Detector import Detector
from smells.NC.NC import NC
from smells.utils.RunExecuteParametersCalls import count_functions
from smells.utils.config_loader import get_detector_option

def group_calls_by_circuit(run_calls, execute_calls, bind_calls, assign_calls):
    """
    Group function calls by circuit name.
    
    Args:
        run_calls, execute_calls, bind_calls, assign_calls: Lists of call dictionaries
        
    Returns:
        dict: Dictionary where keys are circuit names and values contain all call types
    """
    # Dictionary to store results
    circuits = {}
    
    # Define the call types and their corresponding data
    call_types = {
        'run_calls': run_calls,
        'execute_calls': execute_calls,
        'bind_calls': bind_calls,
        'assign_calls': assign_calls
    }
    
    # Process each call type
    for call_type, calls in call_types.items():
        for call in calls:
            circuit_name = call['circuit']
            
            # Skip if circuit name is None
            if circuit_name is None:
                continue
                
            # Initialize circuit entry if it doesn't exist
            if circuit_name not in circuits:
                circuits[circuit_name] = {
                    'run_calls': [],
                    'execute_calls': [],
                    'bind_calls': [],
                    'assign_calls': []
                }
            
            # Add the call to the appropriate list
            circuits[circuit_name][call_type].append(call)
    
    return circuits

@Detector.register(NC)
class NCDetector(Detector):

    smell_cls = NC

    def detect(self, file):
        smells = []

        run_calls, execute_calls, bind_calls, assign_calls = count_functions(file, debug=False)

        grouped_circuits = group_calls_by_circuit(run_calls, execute_calls, bind_calls, assign_calls)


        for circuit in grouped_circuits:

            total_run_execute = len(grouped_circuits[circuit]['run_calls']) + len(grouped_circuits[circuit]['execute_calls'])
            total_bind_assign = len(grouped_circuits[circuit]['bind_calls']) + len(grouped_circuits[circuit]['assign_calls'])

            difference_threshold = get_detector_option("NC", "difference_threshold", fallback=1)

            if total_run_execute-total_bind_assign>=difference_threshold:
                nc_smell = NC(
                    circuit_name=circuit,
                    run_calls=grouped_circuits[circuit]['run_calls'],
                    execute_calls=grouped_circuits[circuit]['execute_calls'],
                    assign_parameter_calls=grouped_circuits[circuit]['assign_calls'],
                    bind_parameter_calls=grouped_circuits[circuit]['bind_calls'],
                    explanation="",
                    suggestion=""
                )
                smells.append(nc_smell)

        return smells