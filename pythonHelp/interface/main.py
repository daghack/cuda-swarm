#!/usr/bin/python
from PyQt4.QtGui import *

class Example(QMainWindow):
	def __init__(self):
		super(Example, self).__init__()
		self.initUI()
	
	def initUI(self):
		self.setGeometry(300, 300, 250, 250)
		self.setWindowTitle('Test GUI')
		
		textEdit = QTextEdit()
		self.setCentralWidget(textEdit)
		exitAction = QAction(QIcon.fromTheme("window-close"), 'Exit', self)
		exitAction.setShortcut('Ctrl+W')
		exitAction.setStatusTip('Exit Application')
		exitAction.triggered.connect(self.close)
		self.statusBar()
		menubar = self.menuBar()
		fileMenu = menubar.addMenu('&File')
		fileMenu.addAction(exitAction)
		
		toolbar = self.addToolBar('Exit')
		toolbar.addAction(exitAction)
		
		self.center()
		self.show()
	
	def closeEvent(self, event):
		reply = QMessageBox.question(self, 'Confirm',\
			"Are you sure you want to quit?", QMessageBox.Yes | QMessageBox.No,\
			QMessageBox.Yes)
		if reply == QMessageBox.Yes:
			event.accept()
		else:
			event.ignore()
	
	def center(self):
		qr = self.frameGeometry()
		cp = QDesktopWidget().availableGeometry().center()
		qr.moveCenter(cp)
		self.move(qr.topLeft())

def main():
	app = QApplication([])
	w = Example()
	app.exec_()

main()
