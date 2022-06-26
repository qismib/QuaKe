""" This model executes an automatized quantum featuremap optimization via genetic algorithm"""

from pathlib import Path
from quake.utils.utils import load_runcard
from quake.models.qsvm.qsvm_tester import get_features
from quake.models.qsvm import genetic

input_folder = Path("../../output/tmp/")
setup = load_runcard("../../output/tmp/cards/runcard.yaml")
output_folder = Path("../../Genetic_prova2")

dataset, labels = get_features(input_folder, "cnn", setup)

data_train, lab_train = genetic.get_subsample(dataset[0], labels[0], 10)
data_val, lab_val = genetic.get_subsample(dataset[1], labels[1], 5)

# Initializing a random generation 0
function_inputs = genetic.initial_population(10)

# Defining inputs for the genetic instance
options = {
    "num_generations": 5,
    "num_parents_mating": 2,
    "initial_population": function_inputs,
    "parent_selection_type": "rank",
    "mutation_by_replacement": True,
    "stop_criteria": "saturate_50",
}

# Running the instance and retrieving data
ga_instance = genetic.genetic_instance(
    options, data_train, lab_train, data_val, lab_val
)
ga_instance.run()
best_sol = ga_instance.best_solution()
fittest_kernel = genetic.to_quantum(best_sol[0])

print("Fittest kernel:")
print(best_sol[0])
print(fittest_kernel)

# Comparing the fittest kernel with classic ones
classic_kernels = ["linear", "poly", "rbf", "sigmoid"]
data_compare, lab_compare = genetic.get_subsample(dataset[2], labels[2], 1000)
genetic.quick_comparison(
    fittest_kernel, classic_kernels, data_train, lab_train, data_compare, lab_compare
)

# Saving results
genetic.save_results(fittest_kernel, ga_instance, output_folder)
