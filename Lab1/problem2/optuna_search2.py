"""Hyper-parameter search for SARSA2_learning using Optuna."""

from __future__ import annotations

import numpy as np
import optuna
from optuna import samplers
import gymnasium as gym

from problem2e3 import SARSA3_learning
from optuna_search import final_rolling_n_average, make_exponential_schedule

N_EPISODES = 200
window = 50


def objective(trial, n_reps: int) -> float:
	env = gym.make("MountainCar-v0")
	env.reset(seed=trial.number)

	# Hyper-parameters suggested by Optuna
	discount = 1
	p = 2
	lamda = trial.suggest_float("lambda", 0.01, 0.99)
	lr_initial = trial.suggest_float("lr_initial", 5e-7, 5e-3, log=True)
	lr_end = 0 #trial.suggest_float("lr_end", 1e-12, lr_initial, log=True)	
	lr_decay = trial.suggest_float("lr_decay", 0.90, 0.999)
	momentum = trial.suggest_float("momentum", 0.0, 0.999)
	curiosity_start = trial.suggest_float("curiosity_start", 0, 10)
	curiosity_end = 0 #trial.suggest_float("curiosity_end", 0, curiosity_start)
	curiosity_decay = trial.suggest_float("curiosity_decay", 0.90, 0.999)


	# Scheduler factories
	curiosity_schedule = make_exponential_schedule(curiosity_start, curiosity_end, curiosity_decay)
	learning_rate_schedule = make_exponential_schedule(lr_initial, lr_end, lr_decay)

	cur = lambda episode: curiosity_schedule(episode)
	l_rate = lambda episode: learning_rate_schedule(episode)

	eta = np.array([[i, j] for i in range(p + 1) for j in range(p + 1)]).T

	all_rewards = np.zeros(n_reps)
	for i in range(n_reps):
		_, rewards = SARSA3_learning(
			env,
			lamda=lamda,
			discount=discount,
			p=p,
			eta=eta,
			n_episodes=N_EPISODES,
			curiosity=cur,
			l_rate=l_rate,
			plot=False,
			momentum=momentum,
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
		study_name="sarsa3_hyperparam_search",
	)

	study.optimize(lambda trial: objective(trial, n_reps=5), n_trials=n_trials, show_progress_bar=True)
	summarize_best_trials(study)


if __name__ == "__main__":
	main()