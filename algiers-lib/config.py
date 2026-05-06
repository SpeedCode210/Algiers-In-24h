from solvers.greedy_solver import GreedySolver
from solvers.simulated_annealing_solver import SimulatedAnnealingSolver, AcceptanceFunction

solver_configs = [
            (GreedySolver, {'use_ratio': False}, 'Greedy (Score Priority)', f'_greedy_score'),
            (GreedySolver, {'use_ratio': True}, 'Greedy (Ratio Priority)', f'_greedy_ratio'),
            (
                SimulatedAnnealingSolver,
                {
                    'initial_temperature': 50,
                    'cooling_rate': 0.95,
                    'reheating_rate': 0.95,
                    'max_iterations': 100000
                },
                'Simulated Annealing (Boltzmann)',
                f'_sa_boltzmann'
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
                f'_sa_cauchy'
            )
        ]