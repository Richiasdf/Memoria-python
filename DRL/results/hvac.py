import os
import sys
from os.path import dirname, abspath
sys.path.append(dirname(dirname(abspath(__file__))))

import gym
import scipy.io
from environments.posible_hvac2 import HVACenv
from agents.policy_gradient_agents.PPO import PPO
from agents.actor_critic_agents.DDPG import DDPG
from agents.actor_critic_agents.DDPG_HER import DDPG_HER
from agents.actor_critic_agents.SAC import SAC
from agents.actor_critic_agents.TD3 import TD3
from agents.Trainer import Trainer
from utilities.data_structures.Config import Config

Use = (scipy.io.loadmat('results/Use.mat'))['Use']
weather = (scipy.io.loadmat('results/weather.mat'))['Temps']

config = Config()
config.seed = 1
config.environment = HVACenv(Use, weather)
config.num_episodes_to_run = 10000
config.file_to_save_data_results = "results/data_and_graphs/hvac.pkl"
config.file_to_save_results_graph = "results/data_and_graphs/hvac.png"
config.show_solution_score = False
config.visualise_individual_results = False
config.visualise_overall_agent_results = True
config.standard_deviation_results = 1.0
config.runs_per_agent = 5
config.use_GPU = False
config.overwrite_existing_results_file = False
config.randomise_random_seed = True
config.save_model = False


config.hyperparameters = {

    "Actor_Critic_Agents": {
            "Actor": {
                "learning_rate": 0.003,
                "linear_hidden_units": [128, 64],
                "final_layer_activation": None,
                "batch_norm": True,
                "tau": 0.005,
                "gradient_clipping_norm": .5,
                "initialiser": "Xavier"
            },

            "Critic": {
                "learning_rate": 0.02,
                "linear_hidden_units": [128, 128],
                "final_layer_activation": "TANH",
                "batch_norm": True,
                "buffer_size": 1000000,
                "tau": 0.005,
                "gradient_clipping_norm": 1,
                "initialiser": "Xavier"
            },

        "min_steps_before_learning": 1000, #for SAC only
        "batch_size": 256,
        "discount_rate": 0.99,
        "mu": 0.0,  # for O-H noise
        "theta": 0.0015,  # for O-H noise
        "sigma": [1.2, 10, 0.1],  # for O-H noise
        "action_noise_std": 0.2,  # for TD3
        "action_noise_clipping_range": 0.5,  # for TD3
        "update_every_n_steps": 20,
        "HER_sample_proportion": 0.8,
        "learning_updates_per_learning_session": 5,
        "automatically_tune_entropy_hyperparameter": True,
        "entropy_term_weight": None,
        "add_extra_noise": True,
        "do_evaluation_iterations": True,
        "clip_rewards": False

    }

}

if __name__ == "__main__":
    AGENTS = [DDPG]
    trainer = Trainer(config, AGENTS)
    trainer.run_games_for_agents()