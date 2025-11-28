"""Hyper-parameter search for SARSA2_learning using Optuna."""

from __future__ import annotations

import numpy as np
import optuna
from optuna import samplers
import gymnasium as gym

from problem2b import SARSA2_learning
from problem2 import running_average


N_EPISODES = 200
window = 50

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

def make_exponential_schedule(start: float, end: float, decay: float):
	"""Create an exponential decay schedule that never drops below `end`."""

	def schedule(step: int) -> float:
		value = start * (decay ** max(step - 1, 0))
		return max(value, end)

	return schedule


def objective(trial, n_reps: int) -> float:
	env = gym.make("MountainCar-v0")
	env.reset(seed=trial.number)

	# Hyper-parameters suggested by Optuna
	lamda = trial.suggest_float("lambda", 0.6, 0.99)
	discount = 1
	p = 2
	eps_start = trial.suggest_float("eps_start", 0.0, 0.5)
	eps_end = 0 # Sarsa converge therefore we want it to go to zero
	eps_decay = trial.suggest_float("eps_decay", 0.90, 0.999)
	lr_initial = trial.suggest_float("lr_initial", 5e-7, 5e-3, log=True)
	lr_end = 0 #trial.suggest_float("lr_end", 1e-12, lr_initial, log=True)	
	lr_decay = trial.suggest_float("lr_decay", 0.90, 0.999)
	momentum = trial.suggest_float("momentum", 0.0, 0.999)

	# Scheduler factories
	epsilon_schedule = make_exponential_schedule(eps_start, eps_end, eps_decay)
	learning_rate_schedule = make_exponential_schedule(lr_initial, lr_end, lr_decay)

	eps = lambda episode: epsilon_schedule(episode)
	l_rate = lambda episode: learning_rate_schedule(episode)

	eta = np.array([[i, j] for i in range(p + 1) for j in range(p + 1)]).T

	all_rewards = np.zeros(n_reps)
	for i in range(n_reps):
		_, rewards = SARSA2_learning(
			env,
			lamda=lamda,
			discount=discount,
			p=p,
			eta=eta,
			n_episodes=N_EPISODES,
			eps=eps,
			l_rate=l_rate,
			momentum=momentum,
			plot=False,
			debug=True,
		)
		
		rolling_peak = final_rolling_n_average(rewards, window)
		all_rewards[i] = rolling_peak

	end_reward = np.min(all_rewards)
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


def main(n_trials: int = 200) -> None:
	study = optuna.create_study(
		directions=["maximize"],
		study_name="sarsa2_hyperparam_search",
	)

	study.optimize(lambda trial: objective(trial, n_reps=5), n_trials=n_trials, show_progress_bar=True)
	summarize_best_trials(study)


if __name__ == "__main__":
	main()