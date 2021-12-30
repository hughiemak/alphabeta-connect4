import argparse
import time
import numpy as np
from connect4 import Connect4
from pdb import set_trace

H, W = 6, 7
DEFAULT_SEARCH_DEPTH = 4
INF = 1e10
processor = Connect4(H,W)

def result(s,a,is_max):
	"""
	Returns: 
		s -- new state
		(row, col) -- position of last piece
	"""
	return processor.get_next_state(s,a,is_max)

def terminal_test(s, last_pc_position, is_max, return_winner=False):
	is_terminal, utility, max_win = processor.check_win(s, last_pc_position, is_max)
	return is_terminal, utility, max_win # max_win: True (MAX wins), False (MIN wins), None (Otherwise)

def evaluate(s, last_pc_position):
	return processor.heuristic_eval(s, last_pc_position)

def minimax_alpha_beta(s, depth, alpha, beta, is_max, last_pc_position):
	"""
	Minimax with alpha beta pruning
	
	Arguments:
		s -- state
		depth -- the search depth (set to -1 to search until terminal state)
		alpha -- alpha value
		beta -- beta value
		is_max -- True if s is a state for MAX and FALSE if s is a state for MIN
		last_pc_position -- position of last piece (row, col)

	Returns:
		best_a -- best action (None if s is a terminal state)
		v -- the value of the best action
	"""
	
	is_terminal, utility, _ = terminal_test(s, last_pc_position, not is_max)
	if is_terminal:
		return None, utility

	if depth == 0:
		return None, evaluate(s, last_pc_position)

	A = processor.get_valid_actions(s)

	if is_max:
		v, best_a = -INF, None
		for a in A:
			s_, pos =  result(s,a,is_max)
			_, score = minimax_alpha_beta(s_, depth-1, alpha, beta, False, pos)
			if score > v:
				v, best_a = score, a
			if v >= beta:
				break
			alpha = max(alpha, v)
		return best_a, v
	else:
		v, best_a = INF, None
		for a in A:
			s_, pos = result(s,a,is_max)
			_, score = minimax_alpha_beta(s_, depth-1, alpha, beta, True, pos)
			if score < v:
				v, best_a = score, a
			if v <= alpha:
				break
			beta = min(beta, v)
		return best_a, v

def human_turn(state, is_max, verbose=True):
	a = int(input("Insert column: "))
	while a >= W or a < 0:
		print("Invalid column.")
		a = int(input("Insert column: "))
	state, pos = processor.get_next_state(state, a, is_max)
	if verbose:
		print(state, end="\n\n")

	return state, pos

def ai_turn(state, is_max, pos, search_depth, verbose=True):
	if verbose:
		print("AI is taking move ...")
	a, v = minimax_alpha_beta(state, search_depth, -INF, INF, is_max, pos)
	# print(f"AI taking {a} with value {v}.")
	state, pos = processor.get_next_state(state, a, is_max)
	if verbose:
		print(state, end="\n\n")

	return state, pos

def random_agent_turn(state, is_max, verbose=False):
	valid_actions = processor.get_valid_actions(state)
	a = np.random.choice(valid_actions)
	state, pos = processor.get_next_state(state, a, is_max)
	if verbose:
		print(state, end="\n\n")

	return state, pos

if __name__=="__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('depth', type=int, default=DEFAULT_SEARCH_DEPTH, nargs='?')
	parser.add_argument('--aifirst', action='store_true')
	args = parser.parse_args()

	print(args)

	search_depth = args.depth
	wins = 0
	human_first = not args.aifirst
	eval_random_agent = False
	verbose = not eval_random_agent

	if eval_random_agent:
		start = time.time()

	for i in range(100):
		t = 0
		state, pos = processor.get_initial_state()
		if verbose:
			print(state, end="\n\n")
		while True:
			is_max = t % 2 == 0
			if is_max:
				if eval_random_agent:
					state, pos = random_agent_turn(state, is_max, verbose=verbose)
				else: 
					if human_first:
						state, pos = human_turn(state, is_max, verbose=verbose)
					else:
						state, pos = ai_turn(state, is_max, pos, search_depth, verbose=verbose)
			else:
				if eval_random_agent:
					state, pos = ai_turn(state, is_max, pos, search_depth, verbose=verbose)
				else:
					if not human_first:
						state, pos = human_turn(state, is_max, verbose=verbose)
					else:
						state, pos = ai_turn(state, is_max, pos, search_depth, verbose=verbose)

			is_terminal, _, max_win = terminal_test(state, pos, is_max)
			if is_terminal:
				if max_win == True:
					if verbose:
						print(f"MAX wins.", end="\n\n")
					wins += 1
				elif max_win == False:
					if verbose:
						print(f"MIN wins.", end="\n\n")
					wins -= 1
				else:
					if verbose:
						print(f"Draw.", end="\n\n")
				break
			t = t+1
	
	if eval_random_agent:
		print(wins)
		end = time.time()
		print(f"time: {end - start}")
