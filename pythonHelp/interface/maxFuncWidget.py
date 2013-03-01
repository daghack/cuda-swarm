from PyQt4.QtGui import QApplication, QWidget, QAction
from PyQt4.QtGui import QPushButton, QTextEdit, QLabel, QLineEdit, QGridLayout

def testFunc():
	print "Hello World"

class MaxFuncWidget(QWidget):
	def a(self):
		def k():
			print self
		return k
	def __init__(self):
		super(MaxFuncWidget, self).__init__()
		self.initUI()
	
	def initUI(self):
		funcName = QLabel('Function Name')
		dimCount = QLabel('# of Dimensions')
		funcDef = QLabel('Function Definition')
		
		funcNameLine = QLineEdit()
		dimCountLine = QLineEdit()
		funcDefText = QTextEdit()
		
		compileButton = QPushButton('Compile')
		compileButton.clicked.connect(self.a())
		
		grid = QGridLayout()
		grid.setSpacing(10)
		
		grid.addWidget(funcName, 0, 0)
		grid.addWidget(funcNameLine, 0, 1, 1, 3)
		
		grid.addWidget(dimCount, 1, 0)
		grid.addWidget(dimCountLine, 1, 1, 1, 3)
		
		grid.addWidget(funcDef, 2, 0)
		grid.addWidget(funcDefText, 2, 1, 6, 3)
		
		grid.addWidget(compileButton, 8, 3)
		
		compileAction = QAction
		
		self.setLayout(grid)
		self.setGeometry(300, 300, 650, 350)
		self.show()

def test():
	app = QApplication([])
	ex = MaxFuncWidget()
	app.exec_()

test()
