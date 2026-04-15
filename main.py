from matplotlib import pyplot as plt

from domain.correlation.cpa_correlation import CpaCorrelation
from domain.correlation.cpa_service import CorrelationTask
from domain.leakage.leakage import SboxOutputLeakageModel
from domain.value_object import RangeParameters, Range, KeyByteGuesses, DataSource
from infrastructure.h5_trace_repository import H5TraceRepository
from ui.correlation_plotter import CorrelationPlotter
from ui.key_rank_visualizer import KeyRankVisualizer
from ui.trace_plotter import TracePlotter

byte_location = 1
range_parameter = RangeParameters(
    trace_range=Range(0, 125),
    trace_sample_range=Range(14000, 44000),
    key_guess_list=KeyByteGuesses.from_full256_range(),
)
file_path = "data/smart-card-project/smart_card_project.sx"

leakage_model = SboxOutputLeakageModel()

trace_repo = H5TraceRepository(file_path)
data_source = DataSource.PLAINTEXT

results = []
for i in [0, 1]:
    correlation = CpaCorrelation()
    result = CorrelationTask(
        byte_location=i,
        range_parameters=range_parameter,
        leakage_model=leakage_model,
        correlation=correlation,
        trace_repository=trace_repo,
        data_source=data_source,
    ).run()
    results.append((i, result))


# 1. Initialize with your results (list of 16 arrays)
visualizer = KeyRankVisualizer(results)

# 2. Display the table
# This will show the top 5 candidates for every byte
visualizer.display_rank_table(top_n=5)

# 3. Print the final guessed key hex string
guessed_key = visualizer.get_full_key_guess()
print(f"Guessed Key: {guessed_key.hex().upper()}")

plotter = CorrelationPlotter(results)
plotter.plot(1)
