import numpy as np
from pdb import set_trace

class Connect4:
	def __init__(self, H, W):
		self.H, self.W = H, W
		self.MAX, self.MIN = 1, 2
		self.WIN_REWARD = 1000.

	def get_initial_state(self):
		return np.zeros((self.H, self.W)), None

	def __get_row_for_insertion(self, s, a):
		return np.sum(s[:,a]==0)-1

	def get_piece(self, is_max):
		return self.MAX if is_max else self.MIN

	def __is_action_valid(self, a):
		return a >= 0 and a < self.W

	def __is_column_not_full(self, s, a):
		return s[0,a] == 0

	def get_valid_actions(self, s):
		valid_actions = []
		for a in range(self.W):
			if self.__is_column_not_full(s, a):
				valid_actions.append(a) 
		return valid_actions

	def get_win_reward(self, is_max):
		reward = self.WIN_REWARD
		sign = 1. if is_max else -1.
		return sign * reward

	def get_next_state(self, s, a, is_max):
		"""
		Get the next state concat with the current action
		
		Arguments:
			s -- current state, np.array of dim (self.H, self.W)
			a -- current action
			is_max -- boolean indicating whether it is MAX or MIN turn

		Returns:
			s_ -- next state
			(row, col) -- last piece position
		"""

		assert self.__is_action_valid(a), f"Invalid action {a}."
		assert self.__is_column_not_full(s, a), f"Trying to insert piece to column {a} but it is full."
		s_ = np.copy(s)
		row_to_insert = self.__get_row_for_insertion(s_, a)
		s_[row_to_insert][a] = self.get_piece(is_max)
		return s_, (row_to_insert, a)

	def get_bounds(self, row, column):
		maxRow = min(row+3, self.H-1)
		maxCol = min(column+3, self.W-1)
		minRow = max(row-3, 0)
		minCol = max(column-3, 0)

		# downward diagonals
		downward_min_offset = min(row-minRow, column-minCol)
		downward_start_row = row-downward_min_offset
		downward_start_col = column-downward_min_offset
		downward_max_offset = min(maxRow-row, maxCol-column)
		downward_end_row = row + downward_max_offset
		downward_end_col = column + downward_max_offset

		# upward diagonals
		upward_start_offset = min(maxRow-row, column-minCol)
		upward_start_row = row + upward_start_offset
		upward_start_col = column - upward_start_offset
		upward_max_offset = min(row-minRow, maxCol-column)
		upward_end_row = row - upward_max_offset
		upward_end_col = column + upward_max_offset

		return minRow, minCol, maxRow, maxCol, \
		downward_start_row, downward_start_col, \
		downward_end_row, downward_end_col, \
		upward_start_row, upward_start_col, \
		upward_end_row, upward_end_col

	def board_is_full(self, s):
		return 0 not in s

	def check_win(self, s, last_pc_position, is_max):
		"""
		Returns:
			is_terminal -- boolean
			win_reward -- -1000, 0, 1000, or None
			is_max -- who wins?
		"""

		if last_pc_position == None:
			return False, None, None

		row, column = last_pc_position
		player = self.get_piece(is_max)

		minRow, minCol, maxRow, maxCol, downward_start_row, downward_start_col, downward_end_row, downward_end_col, upward_start_row, upward_start_col, upward_end_row, upward_end_col = self.get_bounds(row, column)

		# check horizontal
		hCount = 0
		for x in range(minCol,maxCol+1):
			if (s[row][x]!=player):
				hCount = 0
			else:
				hCount += 1
			# print(hCount)
			if hCount==4:
				# print(f"horizontal at {row}, {minCol} -> {row}, {maxCol}")
				return True, self.get_win_reward(is_max), is_max

		# check vertical
		vCount = 0
		for y in range(minRow,maxRow+1):
			# print(y)
			if (s[y][column]!=player):
				vCount = 0
			else:
				vCount += 1
			# print(vCount)
			if vCount==4:
				# print(f"vertical at {minRow}, {column} -> {maxRow}, {column}")
				return True, self.get_win_reward(is_max), is_max

		# check downward diagonal (start top left, end: bottom right)
		ddCount = 0

		for i in range(downward_end_col - downward_start_col + 1):
			if (s[downward_start_row+i][downward_start_col+i]!=player):
				ddCount = 0
			else:
				ddCount += 1
			# print(ddCount)
			if ddCount==4:
				# print(f"downward diagonal at {downward_start_row}, {downward_start_col} -> {downward_end_row}, {downward_end_col}")
				return True, self.get_win_reward(is_max), is_max

		# check upward diagonal (start: bottom left, end: top right)
		udCount = 0
		for i in range(upward_end_col - upward_start_col + 1):
			if (s[upward_start_row - i][upward_start_col + i]!=player):
				udCount = 0
			else:
				udCount += 1
			# print(udCount)
			if udCount==4:
				# print(f"upward diagonal at {upward_start_row}, {upward_start_col} -> {upward_end_row}, {upward_end_col}")
				return True, self.get_win_reward(is_max), is_max

		if self.board_is_full(s):
			# Draw: no win + board is full
			return True, 0.0, None

		# No one win and board is not full
		return False, None, None

	def score_segment(self, segment):
		k = len(segment)
		assert k == 4, f"Segment length should be 4, got {k}."
		segment = np.array(segment)
		has_max_pcs = self.MAX in segment
		has_min_pcs = self.MIN in segment
		if has_max_pcs and has_min_pcs:
			return 0.
		else:
			if has_max_pcs:
				return 1 * np.sum(segment == self.MAX)
			elif has_min_pcs:
				return -1 * np.sum(segment == self.MIN)
			return 0.

	def score_array(self, array):
		n = len(array)
		assert n >= 4, f"Array length should be >= 4, got {n}."
		score = 0.
		for i in range(n-3):
			score += self.score_segment(array[i:i+4])
		return score

	def score_horizontal(self, s, row, minCol, maxCol):
		return self.score_array(s[row,minCol:maxCol+1])

	def score_vertical(self, s, col, minRow, maxRow):
		return self.score_array(s[minRow:maxRow+1,col])

	def score_downward_diagonal(self, s, downward_start_row, downward_start_col, downward_end_row, downward_end_col):
		k = downward_end_col - downward_start_col + 1
		if k < 4:
			return 0.
		array = []
		for i in range(k):
			array.append(s[downward_start_row+i,downward_start_col+i])
		return self.score_array(array)

	def score_upward_diagonal(self, s, upward_start_row, upward_start_col, upward_end_row, upward_end_col):
		k = upward_end_col - upward_start_col + 1
		if k < 4:
			return 0.
		array = []
		for i in range(k):
			array.append(s[upward_start_row-i,upward_start_col+i])
		return self.score_array(array)

	def heuristic_eval(self, s, last_pc_position):
		row, column = last_pc_position

		prev_state = np.copy(s)
		prev_state[last_pc_position] = 0

		prev_score = 0.
		score = 0.

		minRow, minCol, maxRow, maxCol, downward_start_row, downward_start_col, downward_end_row, downward_end_col, upward_start_row, upward_start_col, upward_end_row, upward_end_col = self.get_bounds(row, column)

		# horizontal
		prev_score += self.score_horizontal(prev_state, row, minCol, maxCol)
		score += self.score_horizontal(s, row, minCol, maxCol)
		# print(self.score_horizontal(prev_state, row, minCol, maxCol), self.score_horizontal(s, row, minCol, maxCol))

		# vertical
		prev_score += self.score_vertical(prev_state, column, minRow, maxRow)
		score += self.score_vertical(s, column, minRow, maxRow)
		# print(self.score_vertical(prev_state, column, minRow, maxRow), self.score_vertical(s, column, minRow, maxRow))

		# downward diagonal
		prev_score += self.score_downward_diagonal(prev_state, downward_start_row, downward_start_col, downward_end_row, downward_end_col)
		score += self.score_downward_diagonal(s, downward_start_row, downward_start_col, downward_end_row, downward_end_col)
		# print(self.score_downward_diagonal(prev_state, downward_start_row, downward_start_col, downward_end_row, downward_end_col), self.score_downward_diagonal(s, downward_start_row, downward_start_col, downward_end_row, downward_end_col))

		# upward diagonal
		prev_score += self.score_upward_diagonal(prev_state, upward_start_row, upward_start_col, upward_end_row, upward_end_col)
		score += self.score_upward_diagonal(s, upward_start_row, upward_start_col, upward_end_row, upward_end_col)
		# print(self.score_upward_diagonal(prev_state, upward_start_row, upward_start_col, upward_end_row, upward_end_col), self.score_upward_diagonal(s, upward_start_row, upward_start_col, upward_end_row, upward_end_col))

		return score - prev_score
