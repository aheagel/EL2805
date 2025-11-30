"""Hyper-parameter search for SARSA2_learning using Optuna."""

from __future__ import annotations

import numpy as np
import optuna
from optuna import samplers
import gymnasium as gym
from joblib import Parallel, delayed

from problem_2b import SARSA2_learning, make_polynomial_schedule, make_exponential_schedule
from problem2 import running_average


N_EPISODES = 200
window = 25

def final_min_last_n(rewards: list[float], window: int) -> float:
	"""Return the minimum reward in the last n episodes."""

	rewards_arr = np.asarray(rewards, dtype=np.float64)
	if rewards_arr.size < window:
		return float("-inf")

	return float(np.min(rewards_arr[-window:]))

def final_rolling_n_average(rewards: list[float], window: int) -> float:
	"""Return the minimum reward in the last n episodes."""

	rewards_arr = np.asarray(rewards, dtype=np.float64)
	if rewards_arr.size < window:
		return float("-inf")

	return float(np.mean(rewards_arr[-window:]))

def run_repetition(seed, lamda, discount, p, eta, n_episodes, eps, l_rate, momentum):
	"""Helper function to run a single repetition in parallel."""
	env = gym.make("MountainCar-v0")
	env.reset(seed=seed)
	try:
		_, rewards = SARSA2_learning(
			env,
			lamda=lamda,
			discount=discount,
			p=p,
			eta=eta,
			n_episodes=n_episodes,
			eps=eps,
			l_rate=l_rate,
			momentum=momentum,
			plot=False,
			debug=True, # Disable tqdm progress bar for parallel runs
		)
	except Exception:
		# Fallback for failed runs (e.g. numerical instability)
		rewards = [-200.0] * n_episodes
	
	return final_rolling_n_average(rewards, window)

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
	eps_start = trial.suggest_float("eps_start", 1e-7, 0.5, log=True)
	eps_end = 0 # Sarsa converge therefore we want it to go to zero
	eps_decay = trial.suggest_float("eps_decay", 0.90, 0.999)

	# Scheduler factories
	epsilon_schedule = make_exponential_schedule(eps_start, eps_end, eps_decay)
	learning_rate_schedule = make_polynomial_schedule(lr_initial, lr_end, lr_scale, lr_power)

	eps = lambda episode: epsilon_schedule(episode)
	l_rate = lambda episode: learning_rate_schedule(episode)

	eta = np.array([[i, j] for i in range(p + 1) for j in range(p + 1)]).T

	# Parallelize the repetitions
	all_rewards = Parallel(n_jobs=-1)(
		delayed(run_repetition)(
			seed=trial.number + i,
			lamda=lamda,
			discount=discount,
			p=p,
			eta=eta,
			n_episodes=N_EPISODES,
			eps=eps,
			l_rate=l_rate,
			momentum=momentum
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
		study_name="sarsa2_hyperparam_search",
	)

	study.optimize(lambda trial: objective(trial, n_reps=25), n_trials=n_trials, show_progress_bar=True)
	summarize_best_trials(study)


if __name__ == "__main__":
	main()