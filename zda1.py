from jmetal.algorithm.multiobjective.gde3 import GDE3
from jmetal.algorithm.multiobjective import NSGAII
from jmetal.algorithm.multiobjective.mocell import MOCell
from jmetal.util.neighborhood import C9
from jmetal.algorithm.multiobjective.omopso import OMOPSO
from jmetal.util.archive import CrowdingDistanceArchive
from jmetal.operator import UniformMutation
from jmetal.operator.mutation import NonUniformMutation
from jmetal.operator import SBXCrossover, PolynomialMutation
from jmetal.lab.visualization import Plot
from jmetal.util.termination_criterion import StoppingByEvaluations
from jmetal.util.solution import (
    print_function_values_to_file,
    print_variables_to_file,
    read_solutions,
)
from jmetal.problem import ZDT1

MAX_EVALUATIONS = 10000
POPULATION_SIZE = 100

problem = ZDT1()
problem.reference_front = read_solutions(filename="./zdt1.pf")


def save_algorithm_to_file(algorithm):
    front = algorithm.get_result()

    # Save results to file
    print_function_values_to_file(front, "FUN." + algorithm.label)
    print_variables_to_file(front, "VAR." + algorithm.label)


def plot_algorithm(algorithm, label, filename):
    front = algorithm.get_result()

    plot_front = Plot(
        title="Pareto front approximation. Problem: ",
        reference_front=problem.reference_front,
        axis_labels=problem.obj_labels,
    )
    plot_front.plot(front, label=label, filename=filename, format="png")


if __name__ == "__main__":
    mutation_probability = 1.0 / problem.number_of_variables
    gde3 = GDE3(problem=problem,
                population_size=POPULATION_SIZE,
                cr=0.5,
                f=0.5,
                termination_criterion=StoppingByEvaluations(
                    max_evaluations=MAX_EVALUATIONS))

    gde3.run()

    save_algorithm_to_file(gde3)
    plot_algorithm(algorithm=gde3, label="GDE-3 ZDT-1", filename="gde3_plot")

    nsga2 = NSGAII(problem=problem,
                   population_size=POPULATION_SIZE,
                   offspring_population_size=100,
                   mutation=PolynomialMutation(probability=1.0 /
                                               problem.number_of_variables,
                                               distribution_index=20),
                   crossover=SBXCrossover(probability=1.0,
                                          distribution_index=20),
                   termination_criterion=StoppingByEvaluations(
                       max_evaluations=MAX_EVALUATIONS))

    nsga2.run()
    save_algorithm_to_file(nsga2)
    plot_algorithm(algorithm=nsga2,
                   label="NSGA-II ZDT-1",
                   filename="nsga2_plot")

    mocell = MOCell(
        problem=problem,
        population_size=POPULATION_SIZE,
        neighborhood=C9(10, 10),
        archive=CrowdingDistanceArchive(100),
        mutation=PolynomialMutation(probability=1.0 /
                                    problem.number_of_variables,
                                    distribution_index=20),
        crossover=SBXCrossover(probability=1.0, distribution_index=20),
        termination_criterion=StoppingByEvaluations(
            max_evaluations=MAX_EVALUATIONS),
    )

    mocell.run()
    save_algorithm_to_file(mocell)
    plot_algorithm(algorithm=mocell,
                   label="MOCell ZDT-1",
                   filename="mocell_plot")

    omopso = OMOPSO(
        problem=problem,
        swarm_size=100,
        epsilon=0.0075,
        uniform_mutation=UniformMutation(probability=mutation_probability,
                                         perturbation=0.5),
        non_uniform_mutation=NonUniformMutation(mutation_probability,
                                                perturbation=0.5,
                                                max_iterations=int(
                                                    MAX_EVALUATIONS / 100)),
        leaders=CrowdingDistanceArchive(100),
        termination_criterion=StoppingByEvaluations(
            max_evaluations=MAX_EVALUATIONS),
    )

    omopso.run()
    save_algorithm_to_file(omopso)
    plot_algorithm(omopso, label="OMOPSO ZDT-1", filename="omopso_plot")