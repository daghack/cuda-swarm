class DFA():
	def __init__(self, start, transitions, accept):
		self.start = start
		self.transitions = transitions #{(state, symbol) : state'}
		self.accept = frozenset(accept)

	def trans(self, state, symbol):
		if (state, symbol) in self.transitions:
			return self.transitions[(state, symbol)]
		return None

	def process_symbol(self, state, symbol):
		return self.trans(state, symbol)
	
	def process_string(self, state, current, head, tail, linecount):
		if not tail:
			return (head + [(state, current)], linecount)
		else:
			k = self.process_symbol(state, tail[0])
			if tail[0] == '\n':
				linecount += 1
			if k:
				return self.process_string(k, current+tail[0], head, tail[1:], linecount)
			else:
				if state == self.start:
					return self.process_string(self.start, '', head + [("ERROR", tail[0])], tail[1:], linecount)
				return self.process_string(self.start, '', head + [(state, current)], tail, linecount)
	
	def process(self, str):
		return self.process_string(self.start, '', [], str)

