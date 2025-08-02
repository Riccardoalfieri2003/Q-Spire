from smells.Detector import Detector
from smells.NC.NC import NC
from smells.utils.RunExecuteParametersCalls import count_functions
from smells.utils.config_loader import get_detector_option

@Detector.register(NC)
class NCDetector(Detector):

    smell_cls = NC

    def detect(self, file):
        smells = []

        run_calls, execute_calls, bind_calls, assign_calls = count_functions(file, debug=False)

        total_run_execute = len(run_calls) + len(execute_calls)
        total_bind_assign = len(bind_calls) + len(assign_calls)

        difference_threshold = get_detector_option("NC", "difference_threshold", fallback=1)

       #if total_bind_assign > 0 and total_run_execute-total_bind_assign>=difference_threshold:
        if total_run_execute-total_bind_assign>=difference_threshold:
            nc_smell = NC(
                run_calls=run_calls,
                execute_calls=execute_calls,
                assign_parameter_calls=assign_calls,
                bind_parameter_calls=bind_calls,
                explanation="",
                suggestion=""
            )
            smells.append(nc_smell)

        return smells