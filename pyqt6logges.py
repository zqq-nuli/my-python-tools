import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QTextEdit, QVBoxLayout, QWidget, QPushButton
from PyQt6.QtCore import Qt


class LoggerWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Logger Window")

        # 创建主部件
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        # 创建布局
        self.layout = QVBoxLayout(self.central_widget)

        # 创建 QTextEdit 作为日志框
        self.log_text_edit = QTextEdit(self)
        self.log_text_edit.setReadOnly(True)  # 设置为只读

        # 创建按钮来生成日志消息
        self.log_button = QPushButton("Add Log", self)
        self.log_button.clicked.connect(self.add_log_message)

        # 添加部件到布局
        self.layout.addWidget(self.log_text_edit)
        self.layout.addWidget(self.log_button)

    def add_log_message(self):
        # 添加一些日志消息
        self.log_text_edit.append("This is a log message.")

        # 自动滚动到底部
        self.log_text_edit.verticalScrollBar().setValue(
            self.log_text_edit.verticalScrollBar().maximum())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = LoggerWindow()
    window.resize(400, 300)
    window.show()
    sys.exit(app.exec())
