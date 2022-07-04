from PyQt5 import QtCore, QtGui, QtWidgets

import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QWidget, QMainWindow, QHBoxLayout, QSizePolicy
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd



class MatplotlibWidget(QWidget):
    def __init__(self, parent=None):
        QWidget.__init__(self, parent)
        self.canvas = FigureCanvas(Figure())

        vertical_layout = QHBoxLayout()
        vertical_layout.addWidget(self.canvas)

        self.canvas.axes = self.canvas.figure.add_subplot(111)
        self.setLayout(vertical_layout)
        self.fig = Figure(figsize=(5, 4), dpi=100)
        self.lines = []
        self.labels = []


    def start_static_plot(self):
        self.fig.suptitle('figure')
        df = pd.read_csv("./Book1.csv")
        print(df.tail())
        source_select = ["r", "b"]
        type_select = ["a"]
        d_num_select = ["d18"]
        for source in source_select:
            for type in type_select:
                for d_num in d_num_select:
                    df_s = df.query('source==@source & type==@type & d_num==@d_num')
                    print("df_s is:", df_s)
                    self.lines += self.canvas.axes.plot(df_s['power'], df_s['pd_cur'], label="{}-{}-{}".format(source, type, d_num))
                    # plt.legend(labels=[source+"-"+type+"-"+d_num+"_"])
        self.labels = [l.get_label() for l in self.lines]

        # 测试其他静态代码
        # t = np.arange(0.0, 3.0, 0.01)
        # s = np.sin(2 * np.pi * t)
        # plt.legend(self.lines, self.labels)
        # self.canvas.axes.plot(t, s)
        # self.canvas.draw()

        # 添加必要的图例（）
        self.canvas.axes.set_title('This is a static chart')  # remeber to modify
        self.canvas.axes.set_xlabel('power')
        self.canvas.axes.set_xlabel('pd_cur')
        self.canvas.axes.legend(self.lines, self.labels)

        self.canvas.draw()

        # 清除以上画的(该行一添加，则所有图像无)
        # self.canvas.axes.clear()


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.plot = QtWidgets.QPushButton(self.centralwidget)
        self.plot.setGeometry(QtCore.QRect(530, 500, 89, 25))
        self.plot.setObjectName("plot")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(540, 30, 131, 17))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(540, 150, 131, 17))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(530, 430, 131, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.pushButton = QtWidgets.QPushButton(self.centralwidget)
        self.pushButton.setGeometry(QtCore.QRect(630, 500, 89, 25))
        self.pushButton.setObjectName("pushButton")
        self.widget = QtWidgets.QWidget(self.centralwidget)
        self.widget.setGeometry(QtCore.QRect(140, 220, 120, 80))
        self.widget.setObjectName("widget")
        self.widget_2 = MatplotlibWidget(self.centralwidget)
        self.widget_2.setGeometry(QtCore.QRect(50, 40, 451, 471))
        self.widget_2.setObjectName("widget_2")
        self.widget1 = QtWidgets.QWidget(self.centralwidget)
        self.widget1.setGeometry(QtCore.QRect(540, 180, 224, 228))
        self.widget1.setObjectName("widget1")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.widget1)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.checkBox_6 = QtWidgets.QCheckBox(self.widget1)
        self.checkBox_6.setChecked(False)
        self.checkBox_6.setObjectName("checkBox_6")
        self.gridLayout_3.addWidget(self.checkBox_6, 0, 0, 1, 1)
        self.checkBox_7 = QtWidgets.QCheckBox(self.widget1)
        self.checkBox_7.setObjectName("checkBox_7")
        self.gridLayout_3.addWidget(self.checkBox_7, 0, 1, 1, 1)
        self.checkBox_9 = QtWidgets.QCheckBox(self.widget1)
        self.checkBox_9.setObjectName("checkBox_9")
        self.gridLayout_3.addWidget(self.checkBox_9, 0, 2, 1, 1)
        self.checkBox_10 = QtWidgets.QCheckBox(self.widget1)
        self.checkBox_10.setObjectName("checkBox_10")
        self.gridLayout_3.addWidget(self.checkBox_10, 0, 3, 1, 1)
        self.checkBox_8 = QtWidgets.QCheckBox(self.widget1)
        self.checkBox_8.setObjectName("checkBox_8")
        self.gridLayout_3.addWidget(self.checkBox_8, 1, 0, 1, 1)
        self.checkBox_11 = QtWidgets.QCheckBox(self.widget1)
        self.checkBox_11.setObjectName("checkBox_11")
        self.gridLayout_3.addWidget(self.checkBox_11, 1, 1, 1, 1)
        self.checkBox_12 = QtWidgets.QCheckBox(self.widget1)
        self.checkBox_12.setObjectName("checkBox_12")
        self.gridLayout_3.addWidget(self.checkBox_12, 1, 2, 1, 1)
        self.checkBox_13 = QtWidgets.QCheckBox(self.widget1)
        self.checkBox_13.setObjectName("checkBox_13")
        self.gridLayout_3.addWidget(self.checkBox_13, 1, 3, 1, 1)
        self.checkBox_16 = QtWidgets.QCheckBox(self.widget1)
        self.checkBox_16.setObjectName("checkBox_16")
        self.gridLayout_3.addWidget(self.checkBox_16, 2, 0, 1, 1)
        self.checkBox_15 = QtWidgets.QCheckBox(self.widget1)
        self.checkBox_15.setObjectName("checkBox_15")
        self.gridLayout_3.addWidget(self.checkBox_15, 2, 1, 1, 1)
        self.checkBox_14 = QtWidgets.QCheckBox(self.widget1)
        self.checkBox_14.setObjectName("checkBox_14")
        self.gridLayout_3.addWidget(self.checkBox_14, 2, 2, 1, 1)
        self.checkBox_17 = QtWidgets.QCheckBox(self.widget1)
        self.checkBox_17.setObjectName("checkBox_17")
        self.gridLayout_3.addWidget(self.checkBox_17, 2, 3, 1, 1)
        self.checkBox_18 = QtWidgets.QCheckBox(self.widget1)
        self.checkBox_18.setObjectName("checkBox_18")
        self.gridLayout_3.addWidget(self.checkBox_18, 3, 0, 1, 1)
        self.checkBox_19 = QtWidgets.QCheckBox(self.widget1)
        self.checkBox_19.setObjectName("checkBox_19")
        self.gridLayout_3.addWidget(self.checkBox_19, 3, 1, 1, 1)
        self.checkBox_20 = QtWidgets.QCheckBox(self.widget1)
        self.checkBox_20.setObjectName("checkBox_20")
        self.gridLayout_3.addWidget(self.checkBox_20, 3, 2, 1, 1)
        self.checkBox_21 = QtWidgets.QCheckBox(self.widget1)
        self.checkBox_21.setObjectName("checkBox_21")
        self.gridLayout_3.addWidget(self.checkBox_21, 3, 3, 1, 1)
        self.checkBox_24 = QtWidgets.QCheckBox(self.widget1)
        self.checkBox_24.setObjectName("checkBox_24")
        self.gridLayout_3.addWidget(self.checkBox_24, 4, 0, 1, 1)
        self.checkBox_25 = QtWidgets.QCheckBox(self.widget1)
        self.checkBox_25.setObjectName("checkBox_25")
        self.gridLayout_3.addWidget(self.checkBox_25, 4, 1, 1, 1)
        self.checkBox_22 = QtWidgets.QCheckBox(self.widget1)
        self.checkBox_22.setObjectName("checkBox_22")
        self.gridLayout_3.addWidget(self.checkBox_22, 4, 2, 1, 1)
        self.checkBox_23 = QtWidgets.QCheckBox(self.widget1)
        self.checkBox_23.setObjectName("checkBox_23")
        self.gridLayout_3.addWidget(self.checkBox_23, 4, 3, 1, 1)
        self.checkBox_29 = QtWidgets.QCheckBox(self.widget1)
        self.checkBox_29.setObjectName("checkBox_29")
        self.gridLayout_3.addWidget(self.checkBox_29, 5, 0, 1, 1)
        self.checkBox_26 = QtWidgets.QCheckBox(self.widget1)
        self.checkBox_26.setObjectName("checkBox_26")
        self.gridLayout_3.addWidget(self.checkBox_26, 5, 1, 1, 1)
        self.checkBox_27 = QtWidgets.QCheckBox(self.widget1)
        self.checkBox_27.setObjectName("checkBox_27")
        self.gridLayout_3.addWidget(self.checkBox_27, 5, 2, 1, 1)
        self.checkBox_28 = QtWidgets.QCheckBox(self.widget1)
        self.checkBox_28.setObjectName("checkBox_28")
        self.gridLayout_3.addWidget(self.checkBox_28, 5, 3, 1, 1)
        self.checkBox_32 = QtWidgets.QCheckBox(self.widget1)
        self.checkBox_32.setObjectName("checkBox_32")
        self.gridLayout_3.addWidget(self.checkBox_32, 6, 0, 1, 1)
        self.checkBox_30 = QtWidgets.QCheckBox(self.widget1)
        self.checkBox_30.setObjectName("checkBox_30")
        self.gridLayout_3.addWidget(self.checkBox_30, 6, 1, 1, 1)
        self.checkBox_31 = QtWidgets.QCheckBox(self.widget1)
        self.checkBox_31.setObjectName("checkBox_31")
        self.gridLayout_3.addWidget(self.checkBox_31, 6, 2, 1, 1)
        self.checkBox_33 = QtWidgets.QCheckBox(self.widget1)
        self.checkBox_33.setObjectName("checkBox_33")
        self.gridLayout_3.addWidget(self.checkBox_33, 6, 3, 1, 1)
        self.checkBox_34 = QtWidgets.QCheckBox(self.widget1)
        self.checkBox_34.setObjectName("checkBox_34")
        self.gridLayout_3.addWidget(self.checkBox_34, 7, 0, 1, 1)
        self.checkBox_37 = QtWidgets.QCheckBox(self.widget1)
        self.checkBox_37.setObjectName("checkBox_37")
        self.gridLayout_3.addWidget(self.checkBox_37, 7, 1, 1, 1)
        self.checkBox_36 = QtWidgets.QCheckBox(self.widget1)
        self.checkBox_36.setObjectName("checkBox_36")
        self.gridLayout_3.addWidget(self.checkBox_36, 7, 2, 1, 2)
        self.widget2 = QtWidgets.QWidget(self.centralwidget)
        self.widget2.setGeometry(QtCore.QRect(540, 60, 221, 54))
        self.widget2.setObjectName("widget2")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.widget2)
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.checkBox = QtWidgets.QCheckBox(self.widget2)
        self.checkBox.setObjectName("checkBox")
        self.gridLayout_4.addWidget(self.checkBox, 0, 0, 1, 1)
        self.checkBox_2 = QtWidgets.QCheckBox(self.widget2)
        self.checkBox_2.setObjectName("checkBox_2")
        self.gridLayout_4.addWidget(self.checkBox_2, 0, 1, 1, 1)
        self.checkBox_3 = QtWidgets.QCheckBox(self.widget2)
        self.checkBox_3.setObjectName("checkBox_3")
        self.gridLayout_4.addWidget(self.checkBox_3, 1, 0, 1, 1)
        self.widget3 = QtWidgets.QWidget(self.centralwidget)
        self.widget3.setGeometry(QtCore.QRect(530, 460, 231, 25))
        self.widget3.setObjectName("widget3")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.widget3)
        self.gridLayout_5.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.checkBox_4 = QtWidgets.QCheckBox(self.widget3)
        self.checkBox_4.setObjectName("checkBox_4")
        self.gridLayout_5.addWidget(self.checkBox_4, 0, 0, 1, 1)
        self.checkBox_5 = QtWidgets.QCheckBox(self.widget3)
        self.checkBox_5.setObjectName("checkBox_5")
        self.gridLayout_5.addWidget(self.checkBox_5, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 22))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.pushButton.clicked['bool'].connect(MainWindow.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        # plot
        self.widget_2.start_static_plot()

        self.plot.setText(_translate("MainWindow", "Plot"))
        self.label.setText(_translate("MainWindow", "source select"))
        self.label_2.setText(_translate("MainWindow", "d_num"))
        self.label_3.setText(_translate("MainWindow", "Type"))
        self.pushButton.setText(_translate("MainWindow", "Close"))
        self.checkBox_6.setText(_translate("MainWindow", "d14"))
        self.checkBox_7.setText(_translate("MainWindow", "d15"))
        self.checkBox_9.setText(_translate("MainWindow", "d16"))
        self.checkBox_10.setText(_translate("MainWindow", "d17"))
        self.checkBox_8.setText(_translate("MainWindow", "d18"))
        self.checkBox_11.setText(_translate("MainWindow", "d19"))
        self.checkBox_12.setText(_translate("MainWindow", "d20"))
        self.checkBox_13.setText(_translate("MainWindow", "d21"))
        self.checkBox_16.setText(_translate("MainWindow", "d22"))
        self.checkBox_15.setText(_translate("MainWindow", "d23"))
        self.checkBox_14.setText(_translate("MainWindow", "d24"))
        self.checkBox_17.setText(_translate("MainWindow", "d25"))
        self.checkBox_18.setText(_translate("MainWindow", "d26"))
        self.checkBox_19.setText(_translate("MainWindow", "d27"))
        self.checkBox_20.setText(_translate("MainWindow", "d28"))
        self.checkBox_21.setText(_translate("MainWindow", "d29"))
        self.checkBox_24.setText(_translate("MainWindow", "d30"))
        self.checkBox_25.setText(_translate("MainWindow", "d31"))
        self.checkBox_22.setText(_translate("MainWindow", "d32"))
        self.checkBox_23.setText(_translate("MainWindow", "d33"))
        self.checkBox_29.setText(_translate("MainWindow", "d34"))
        self.checkBox_26.setText(_translate("MainWindow", "d35"))
        self.checkBox_27.setText(_translate("MainWindow", "d36"))
        self.checkBox_28.setText(_translate("MainWindow", "d37"))
        self.checkBox_32.setText(_translate("MainWindow", "d38"))
        self.checkBox_30.setText(_translate("MainWindow", "d39"))
        self.checkBox_31.setText(_translate("MainWindow", "d40"))
        self.checkBox_33.setText(_translate("MainWindow", "d41"))
        self.checkBox_34.setText(_translate("MainWindow", "d42"))
        self.checkBox_37.setText(_translate("MainWindow", "d43"))
        self.checkBox_36.setText(_translate("MainWindow", "d44"))
        self.checkBox.setText(_translate("MainWindow", "r"))
        self.checkBox_2.setText(_translate("MainWindow", "b"))
        self.checkBox_3.setText(_translate("MainWindow", "g"))
        self.checkBox_4.setText(_translate("MainWindow", "b"))
        self.checkBox_5.setText(_translate("MainWindow", "bbrk"))




if __name__ == '__main__':
    app = QApplication(sys.argv)
    MainWindow = QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())
