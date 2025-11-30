"""Hyper-parameter search for SARSA2_learning using Optuna."""

from __future__ import annotations

import numpy as np
import optuna
from optuna import samplers
import gymnasium as gym
from joblib import Parallel, delayed


from problem_2b import make_polynomial_schedule, make_exponential_schedule
from problem_2e3 import SARSA3_learning
from optuna_search import final_rolling_n_average, make_exponential_schedule

N_EPISODES = 200
window = 50


def run_repetition(env_seed, lamda, discount, p, eta, n_episodes, cur, l_rate, momentum):
	env = gym.make("MountainCar-v0")
	env.reset(seed=env_seed)
	try:
		_, rewards = SARSA3_learning(
			env,
			lamda=lamda,
			discount=discount,
			p=p,
			eta=eta,
			n_episodes=n_episodes,
			curiosity=cur,
			l_rate=l_rate,
			plot=False,
			momentum=momentum,
			debug=True, # Disable tqdm for parallel runs
		)
		env.close()
		return final_rolling_n_average(rewards, window)
	except Exception as e:
		env.close()
		return -200.0

def objective(trial, n_reps: int) -> float:
	# Hyper-parameters suggested by Optuna
	lamda = trial.suggest_float("lambda", 0.7, 0.99)
	momentum = trial.suggest_float("momentum", 0.7, 0.999)
	discount = 1
	p = 2
	lr_initial = trial.suggest_float("lr_initial", 5e-7, 5e-3, log=True)
	lr_end = 0
	lr_scale = trial.suggest_float("lr_scale", 1, N_EPISODES, log=True)
	lr_power = trial.suggest_float("lr_power", 0.5, 1)	
	curiosity_start = trial.suggest_float("curiosity_start", 0, 2)
	curiosity_end = 0
	curiosity_decay = trial.suggest_float("curiosity_decay", 0.90, 0.999)


	# Scheduler factories
	curiosity_schedule = make_exponential_schedule(curiosity_start, curiosity_end, curiosity_decay)
	learning_rate_schedule = make_polynomial_schedule(lr_initial, lr_end, lr_scale, lr_power)

	cur = lambda episode: curiosity_schedule(episode)
	l_rate = lambda episode: learning_rate_schedule(episode)

	eta = np.array([[i, j] for i in range(p + 1) for j in range(p + 1)]).T

	# Parallelize the repetitions
	all_rewards = Parallel(n_jobs=-1)(
		delayed(run_repetition)(
			trial.number + i, # Different seed for each rep
			lamda, discount, p, eta, N_EPISODES, cur, l_rate, momentum
		) for i in range(n_reps)
	)

	end_reward = np.mean(all_rewards)
	trial.set_user_attr(f"reward", end_reward) # Lets be conservative here

	# Maximize rolling mean
	return end_reward


def summarize_best_trials(study) -> None:
	best_trial = study.best_trial
	if best_trial is None:
		print("No completed trials to summarise.")
		return

	attrs = best_trial.user_attrs
	print("Best trial:")
	print(
		f"Trial #{best_trial.number}: value={best_trial.value:.3f}, "
		f"rolling_mean_last_{window}={attrs.get(f'reward'):.3f}, "
		f"params={best_trial.params}"
	)


def main(n_trials: int = 256) -> None:
	study = optuna.create_study(
		directions=["maximize"],
		study_name="sarsa3_hyperparam_search",
	)

	study.optimize(lambda trial: objective(trial, n_reps=25), n_trials=n_trials, show_progress_bar=True)
	summarize_best_trials(study)


if __name__ == "__main__":
	main()