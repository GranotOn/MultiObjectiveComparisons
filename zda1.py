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
from jmetal.problem import ZDT2
from jmetal.problem import ZDT3
from jmetal.problem import ZDT4

MAX_EVALUATIONS = 25000
POPULATION_SIZE = 100

zdt1_problem = ZDT1()
zdt1_problem.reference_front = read_solutions(filename="./zdt1.pf")

zdt2_problem = ZDT2()
zdt2_problem.reference_front = read_solutions(filename="./zdt2.pf")

zdt3_problem = ZDT3()
zdt3_problem.reference_front = read_solutions(filename="./zdt3.pf")

zdt4_problem = ZDT4()
zdt4_problem.reference_front = read_solutions(filename="./zdt4.pf")


def save_algorithm_to_file(algorithm):
    front = algorithm.get_result()

    # Save results to file
    print_function_values_to_file(front, "FUN." + algorithm.label)
    print_variables_to_file(front, "VAR." + algorithm.label)


def plot_algorithm(algorithm, label, filename, problem):
    front = algorithm.get_result()

    plot_front = Plot(
        title="Pareto front approximation. Problem: ",
        reference_front=problem.reference_front,
        axis_labels=problem.obj_labels,
    )
    plot_front.plot(front, label=label, filename=filename, format="png")


def init_gde3(problem):
    gde3 = GDE3(problem=zdt1_problem,
                population_size=POPULATION_SIZE,
                cr=0.5,
                f=0.5,
                termination_criterion=StoppingByEvaluations(
                    max_evaluations=MAX_EVALUATIONS))

    return gde3


def init_mocell(problem):
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
    return mocell


def init_nsga2(problem):
    mutation_probability = 1.0 / problem.number_of_variables
    nsga2 = NSGAII(
        problem=problem,
        population_size=POPULATION_SIZE,
        offspring_population_size=100,
        mutation=PolynomialMutation(probability=mutation_probability,
                                    distribution_index=20),
        crossover=SBXCrossover(probability=0.8, distribution_index=20),
        termination_criterion=StoppingByEvaluations(
            max_evaluations=MAX_EVALUATIONS))
    return nsga2


def init_omopso(problem):
    mutation_probability = 1.0 / problem.number_of_variables
    omopso = OMOPSO(
        problem=problem,
        swarm_size=100,
        epsilon=0.0075,
        uniform_mutation=UniformMutation(probability=mutation_probability,
                                         perturbation=0.5),
        non_uniform_mutation=NonUniformMutation(
            mutation_probability,
            perturbation=0.5,
            max_iterations=int(MAX_EVALUATIONS)),
        leaders=CrowdingDistanceArchive(100),
        termination_criterion=StoppingByEvaluations(
            max_evaluations=MAX_EVALUATIONS),
    )

    return omopso


if __name__ == "__main__":

    ######
    # GDE3
    ######

    # ZDT1
    gde3 = init_gde3(zdt1_problem)
    gde3.run()

    save_algorithm_to_file(gde3)
    plot_algorithm(algorithm=gde3,
                   label="GDE-3 ZDT-1",
                   filename="gde3_zdt1_front",
                   problem=zdt1_problem)

    ########
    # MoCell
    ########

    # ZDT1
    mocell = init_mocell(zdt1_problem)
    mocell.run()

    save_algorithm_to_file(mocell)
    plot_algorithm(algorithm=mocell,
                   label="MOCell ZDT-1",
                   filename="mocell_zdt1_front",
                   problem=zdt1_problem)

    #########
    # NSGA-II
    #########

    # ZDT1
    nsga2 = init_nsga2(zdt1_problem)
    nsga2.run()
    save_algorithm_to_file(nsga2)
    plot_algorithm(algorithm=nsga2,
                   label="NSGA-II ZDT-1",
                   filename="nsga2_zdt1_front",
                   problem=zdt1_problem)

    # ZDT2
    nsga2 = init_nsga2(zdt2_problem)
    nsga2.run()
    save_algorithm_to_file(nsga2)
    plot_algorithm(algorithm=nsga2,
                   label="NSGA-II ZDT-2",
                   filename="nsga2_zdt2_front",
                   problem=zdt2_problem)

    # ZDT3
    nsga2 = init_nsga2(zdt3_problem)
    nsga2.run()
    save_algorithm_to_file(nsga2)
    plot_algorithm(algorithm=nsga2,
                   label="NSGA-II ZDT-3",
                   filename="nsga2_zdt3_front",
                   problem=zdt3_problem)

    # ZDT4
    nsga2 = init_nsga2(zdt4_problem)
    nsga2.run()
    save_algorithm_to_file(nsga2)
    plot_algorithm(algorithm=nsga2,
                   label="NSGA-II ZDT-4",
                   filename="nsga2_zdt4_front",
                   problem=zdt4_problem)

    ########
    # Omopso
    ########

    # ZDT1
    omopso = init_omopso(zdt1_problem)
    omopso.run()
    save_algorithm_to_file(omopso)
    plot_algorithm(omopso,
                   label="OMOPSO ZDT-1",
                   filename="omopso_zdt1_front",
                   problem=zdt1_problem)

    # ZDT2
    omopso = init_omopso(zdt2_problem)
    omopso.run()
    save_algorithm_to_file(omopso)
    plot_algorithm(omopso,
                   label="OMOPSO ZDT-2",
                   filename="omopso_zdt2_front",
                   problem=zdt2_problem)

    # ZDT3
    omopso = init_omopso(zdt3_problem)
    omopso.run()
    save_algorithm_to_file(omopso)
    plot_algorithm(omopso,
                   label="OMOPSO ZDT-3",
                   filename="omopso_zdt3_front",
                   problem=zdt3_problem)

    # ZDT4
    omopso = init_omopso(zdt4_problem)
    omopso.run()
    save_algorithm_to_file(omopso)
    plot_algorithm(omopso,
                   label="OMOPSO ZDT-4",
                   filename="omopso_zdt4_front",
                   problem=zdt4_problem)
