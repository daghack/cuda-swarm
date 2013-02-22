def genTransformedRangeTrans(f, b, t, s, state, pstate):
	k = {}
	for i in map(f, range(b, t, s)):
		k[(state, i)] = pstate
	return k

class TokenizerFactory(object):
	@staticmethod
	def process_symbol(state, symbol, trans):
		return TokenizerFactory.fetch((a, b), trans)
	
	@staticmethod
	def next_token(tokenizer, state, current, tail):
		k = tokenizer[state, tail[0:1]]
		if not k:
			if tokenizer.is_accept_state(state):
				return (state, current, tail)
			else:
				return None
		else:
			l = TokenizerFactory.next_token(tokenizer, k, current + tail[0], tail[1:])
			if not l:
				if tokenizer.is_accept_state(state):
					return (state, current, tail)
				else:
					return None
			else:
				return l
		return ("ERROR", 'tail', '')
	
	@staticmethod
	def process(tokenizer, str):
		while True:
			k =  tokenizer.next_token(str)
			if k:
				(state, token, str) = k
				yield (state, token)
			else:
				if str:
					yield ("ERROR", str)
				break
			
	
	@staticmethod
	def fetch(a, b):
		if a in b:
			return b[a]
		return None
	
	def __new__(cls, name, start, accept, trans):
		h = {'trans':dict(trans), 'start':start, 'is_accept_state':(lambda self, state: state in list(accept))}
		h['next_token'] = lambda self, str: TokenizerFactory.next_token(self, start, '', str)
		h['process'] = lambda self, str: TokenizerFactory.process(self, str)
		h['__getitem__'] = lambda self, k: TokenizerFactory.fetch(k, self.trans)
		return type(name, (), h)

#STATES
#	NEWLINE
#	WHITESPACE
#	NUMBER
#	IDENTIFIERS
#	LEFTBRACKET
#	RIGHTBRACKET
#	LEFTPAREN
#	RIGHTPAREN
#	PRIMATIVEFUNCTIONS
