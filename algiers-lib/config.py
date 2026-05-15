from solvers.greedy_solver import GreedySolver
from solvers.grasp_solver import GraspSolver
from solvers.tabu_solver import TabuSolver
from solvers.genetic_solver import GeneticSolver,TailoredGeneticSolver
from solvers.genetic_fitness import PenaltyFitnessFunction, FeasibilityFitnessFunction
from solvers.simulated_annealing_solver import SimulatedAnnealingSolver, AcceptanceFunction

try:
    from solvers.cplex_solver import CPLEXSolver
except ImportError:
    CPLEXSolver = None

solver_configs = [
    (GreedySolver, {'use_ratio': False}, 'Greedy (Score Priority)', '_greedy_score'),
    (GreedySolver, {'use_ratio': True}, 'Greedy (Ratio Priority)', '_greedy_ratio'),
    (
        GraspSolver,
        {
            'iterations': 50,
            'alpha': 0.3,
        },
        'GRASP',
        '_grasp'
    ),
    (
        TabuSolver,
        {},
        'Tabu Search',
        '_tabu'
    ),
    (
        GeneticSolver,
        {
            'fitness_function': PenaltyFitnessFunction(),
        },
        'Genetic',
        '_genetic'
    ),
    (
        TailoredGeneticSolver,
        {
            'fitness_function': FeasibilityFitnessFunction(),
        },
        'Tailored Genetic',
        '_tailored_genetic'
    ),
]

if CPLEXSolver is not None:
    solver_configs.append(
        (
            CPLEXSolver,
            {
                'time_limit': 60,
            },
            'CPLEX (Exact)',
            '_cplex'
        )
    )

solver_configs.extend([
    (
        SimulatedAnnealingSolver,
        {
            'initial_temperature': 50,
            'cooling_rate': 0.95,
            'reheating_rate': 0.95,
            'max_iterations': 100000
        },
        'Simulated Annealing (Boltzmann)',
        '_sa_boltzmann'
    ),
    (
        SimulatedAnnealingSolver,
        {
            'initial_temperature': 50,
            'cooling_rate': 0.95,
            'reheating_rate': 0.95,
            'max_iterations': 100000,
            'acceptance_criterion': AcceptanceFunction.CAUCHY
        },
        'Simulated Annealing (Cauchy)',
        '_sa_cauchy'
    )
])