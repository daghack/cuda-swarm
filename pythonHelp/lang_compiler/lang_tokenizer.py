from tokenizer_factory import TokenizerFactory, genTransformedRangeTrans

def main_tokenizer():
	trans = genTransformedRangeTrans(str, 0, 9, 1, 'DimVal', 'DimValNum')
	trans.update(genTransformedRangeTrans(str, 0, 9, 1, 'DimValNum', 'DimValNum'))
	trans.update(genTransformedRangeTrans(str, 0, 9, 1, 'Start', 'Float'))
	trans.update(genTransformedRangeTrans(str, 0, 9, 1, 'Float', 'Float'))
	trans.update({('Float', '.'):'FloatPoint'})
	trans.update(genTransformedRangeTrans(str, 0, 9, 1, 'FloatPoint', 'Float'))
	trans.update({('Start', 'd'):'DimVal', ('Start', ' '):'Space'})
	symbols = {('Start', '+'):'Addition', ('Start', '-'):'Subtraction',\
		('Start', '*'):'Multiplication', ('Start', '/'):'Division',\
		('Start', '^'):'Power', ('Start', '='):'Equal', ('Start', '@'):'AT',\
		('AT', 'C'):'CCODE', ('Start', '('):'LParen', ('Start', ')'):'RParen'}
	trans.update(symbols)
	start = 'Start'
	accept = ['DimValNum', 'Space', 'Addition', 'Subtraction',\
		'Multiplication', 'Division', 'Power', 'Float', 'CCODE',\
		'LParen', 'RParen']
	name = 'TestTokenizer'
	return TokenizerFactory(name, start, accept, trans, ['Space', 'Tab'])

