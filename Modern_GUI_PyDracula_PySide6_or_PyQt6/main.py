# ///////////////////////////////////////////////////////////////
#
# BY: WANDERSON M.PIMENTA
# PROJECT MADE WITH: Qt Designer and PySide6
# V: 1.0.0
#
# This project can be used freely for all uses, as long as they maintain the
# respective credits only in the Python scripts, any information in the visual
# interface (GUI) can be modified without any implication.
#
# There are limitations on Qt licenses if you want to use your products
# commercially, I recommend reading them on the official website:
# https://doc.qt.io/qtforpython/licenses.html
#
# ///////////////////////////////////////////////////////////////

import sys
import os
import platform
import wave
import time
from pydub import AudioSegment
from GenreClass.transformer.predict import finalPredict
# IMPORT / GUI AND MODULES AND WIDGETS
# ///////////////////////////////////////////////////////////////
from modules import *
from widgets import *
from PySide6.QtCore import QThread, Signal
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel
from PySide6.QtMultimedia import QMediaPlayer, QMediaFormat, QAudioOutput
os.environ["QT_FONT_DPI"] = "96" # FIX Problem for High DPI and Scale above 100%
global widgets
# SET AS GLOBAL WIDGETS
# ///////////////////////////////////////////////////////////////
# widgets = None
# 尝试创建工作线程类防止卡顿

global output_file
class WorkerThread(QThread):
    progress = Signal(int)
    finished = Signal

    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.output_player = QMediaPlayer()
        self._audio_output2 = QAudioOutput()
        self.output_player.setAudioOutput(self._audio_output2)
        self.output_file = ''

    def run(self):
        self.genre_list, self.tempo, self.file_list = finalPredict(self.filename)
        print(self.file_list)
        widgets.bpm.setText(f"bpm：{self.tempo}")
        widgets.genre.setText(f"风格检测：{self.genre_list}")
        widgets.drum_midi_dir.setText(f"鼓组midi文件目录：" + os.getcwd() + "\latest_drum.mid")
        widgets.label.setText(f"合成文件目录：{self.file_list[0]}")
        self.output_file = self.file_list[0]
        QCoreApplication.processEvents()

        widgets.running_status.setText("生成完成！")



class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        # SET AS GLOBAL WIDGETS
        # ///////////////////////////////////////////////////////////////
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.filename = None
        self._audio_output = QAudioOutput()
        self.player = QMediaPlayer()
        self.player.setAudioOutput(self._audio_output)
        self.output_player = QMediaPlayer()
        self._audio_output2 = QAudioOutput()
        self.output_player.setAudioOutput(self._audio_output2)
        # 进度条
        self.player.positionChanged.connect(self.on_update_slider)
        self.player.durationChanged.connect(self.on_media_changed)
        self.output_player.positionChanged.connect(self.on_update_slider_output)
        self.output_player.durationChanged.connect(self.on_media_changed_output)
        global widgets
        widgets = self.ui
        # self.output = output_file
        widgets.input_progress.installEventFilter(self)
        widgets.output_progress.installEventFilter(self)

        # USE CUSTOM TITLE BAR | USE AS "False" FOR MAC OR LINUX
        # ///////////////////////////////////////////////////////////////
        Settings.ENABLE_CUSTOM_TITLE_BAR = True

        # APP NAME
        # ///////////////////////////////////////////////////////////////
        title = "Druuum"
        description = "Druuum."
        # APPLY TEXTS
        self.setWindowTitle(title)
        widgets.titleRightInfo.setText(description)

        # TOGGLE MENU
        # ///////////////////////////////////////////////////////////////
        widgets.toggleButton.clicked.connect(lambda: UIFunctions.toggleMenu(self, True))

        # SET UI DEFINITIONS
        # ///////////////////////////////////////////////////////////////
        UIFunctions.uiDefinitions(self)

        # QTableWidget PARAMETERS
        # ///////////////////////////////////////////////////////////////
        widgets.tableWidget.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)

        # BUTTONS CLICK
        # ///////////////////////////////////////////////////////////////

        # LEFT MENUS
        widgets.btn_home.clicked.connect(self.buttonClick)
#         widgets.btn_widgets.clicked.connect(self.buttonClick)
        widgets.btn_new.clicked.connect(self.buttonClick)
        widgets.btn_save.clicked.connect(self.buttonClick)
        widgets.uploadFile.clicked.connect(self.buttonClick)
        widgets.generate.clicked.connect(self.buttonClick)
        widgets.play.clicked.connect(self.buttonClick)
        widgets.pause.clicked.connect(self.buttonClick)
        widgets.play_2.clicked.connect(self.buttonClick)
        widgets.pause_2.clicked.connect(self.buttonClick)
        widgets.input_progress.sliderMoved.connect(self.on_slider_moved)
        widgets.input_progress.installEventFilter(self)
        widgets.output_progress.sliderMoved.connect(self.on_slider_moved_output)
        widgets.output_progress.installEventFilter(self)

        # 新增更改主题功能
        # widgets.btn_changeTheme.clicked.connect(self.buttonClick)
        # EXTRA LEFT BOX
        def openCloseLeftBox():
            UIFunctions.toggleLeftBox(self, True)
        widgets.toggleLeftBox.clicked.connect(openCloseLeftBox)
        widgets.extraCloseColumnBtn.clicked.connect(openCloseLeftBox)

        # EXTRA RIGHT BOX
        def openCloseRightBox():
            UIFunctions.toggleRightBox(self, True)
        widgets.settingsTopBtn.clicked.connect(openCloseRightBox)

        # SHOW APP
        # ///////////////////////////////////////////////////////////////
        self.show()
        # 防止打包后路径识别问题
        if getattr(sys, 'frozen', False):
            absPath = os.path.dirname(os.path.abspath(sys.executable))
        elif __file__:
            absPath = os.path.dirname(os.path.abspath(__file__))
        # SET CUSTOM THEME
        # ///////////////////////////////////////////////////////////////
        useCustomTheme = False
        self.useCustomTheme = useCustomTheme
        self.absPath = absPath
        themeFile = "themes\py_dracula_light.qss"

        # SET THEME AND HACKS
        if useCustomTheme:
            # LOAD AND APPLY STYLE
            UIFunctions.theme(self, themeFile, True)

            # SET HACKS
            AppFunctions.setThemeHack(self)

        # SET HOME PAGE AND SELECT MENU
        # ///////////////////////////////////////////////////////////////
        # widgets.stackedWidget.setCurrentWidget(widgets.home)
        widgets.btn_home.setStyleSheet(UIFunctions.selectMenu(widgets.btn_home.styleSheet()))


    # BUTTONS CLICK
    # Post here your functions for clicked buttons
    # ///////////////////////////////////////////////////////////////
    def buttonClick(self):
        # GET BUTTON CLICKED
        btn = self.sender()
        btnName = btn.objectName()

        # SHOW HOME PAGE
        if btnName == "btn_home":
            widgets.stackedWidget.setCurrentWidget(widgets.home)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))

        # SHOW WIDGETS PAGE
        if btnName == "btn_widgets":
            widgets.stackedWidget.setCurrentWidget(widgets.widgets)
            UIFunctions.resetStyle(self, btnName)
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet()))

        # SHOW NEW PAGE
        if btnName == "btn_new":
            widgets.stackedWidget.setCurrentWidget(widgets.new_page) # SET PAGE
            UIFunctions.resetStyle(self, btnName) # RESET ANOTHERS BUTTONS SELECTED
            btn.setStyleSheet(UIFunctions.selectMenu(btn.styleSheet())) # SELECT MENU

        if btnName == "btn_save":
            print("Save BTN clicked!")
            QMessageBox.information(self, "oops", "还在开发中...", QMessageBox.Yes)

        if btnName == "btn_changeTheme":
            if self.useCustomTheme:
                themeFile = os.path.abspath(os.path.join(self.absPath, "themes\py_dracula_dark.qss"))
                UIFunctions.theme(self, themeFile, True)
                AppFunctions.setThemeHack(self)
                self.useCustomTheme = False
            else:
                themeFile = os.path.abspath(os.path.join(self.absPath, "themes\py_dracula_light.qss"))
                UIFunctions.theme(self, themeFile, True)
                AppFunctions.setThemeHack(self)
                self.useCustomTheme = True

        if btnName == "uploadFile":
            self.filename, _ = QFileDialog.getOpenFileName(self, "请选择音频文件", "·")
            if self.filename:
                self.player.setSource(QUrl.fromLocalFile(self.filename))
                # self.player.play()
                widgets.file_dir.setText(self.filename)
                sound = AudioSegment.from_mp3(self.filename)
                sound.export('test.wav', format="wav")
                f = wave.open('test.wav')
                self.sample_rate = f.getframerate()
                widgets.sample_rate.setText(f"采样率：{self.sample_rate}Hz")
                self.channels = f.getnchannels()
                self.sample_width = f.getsampwidth() * 8
                self.bit_rate = self.sample_rate * self.channels * self.sample_width
                widgets.bit_rate.setText(f"比特率：{self.bit_rate}bps")
                self.nframes = f.getnframes()
                self.duration = self.nframes / float(self.sample_rate)
                self.minutes = int(self.duration // 60)
                self.seconds = int(self.duration % 60)
                widgets.duration.setText(f"时长：{self.minutes}分{self.seconds}秒")

        if btnName == "play":
            self.player.play()
        if btnName == "pause":
            self.player.pause()




        if btnName == "generate":
            widgets.running_status.setText("生成中...")
            self.progressDialog = QProgressDialog()
            self.progressDialog.setLabelText("生成中...")
            self.progressDialog.setMaximum(100)
            self.progressDialog.setCancelButton(None)
            self.progressDialog.setStyleSheet(u"QProgressBar::chunk\n"
                                              "{\n"
                                              "border-radius:11px;\n"
                                              "background:qlineargradient(spread:pad,x1:0,y1:0,x2:1,y2:0,stop:0 #01FAFF,stop:1  #26B4FF);\n"
                                              "}\n"
                                              "QProgressBar#progressBar\n"
                                              "{\n"
                                              "height:22px;\n"
                                              "text-align:center;/*\u6587\u672c\u4f4d\u7f6e*/\n"
                                              "font-size:14px;\n"
                                              "color:black;\n"
                                              "border-radius:11px;\n"
                                              "background: #1D5573 ;\n"
                                              "}")
            self.progressDialog.show()

            if self.filename:
                self.worker_thread = WorkerThread(self.filename)
                self.worker_thread.progress.connect(self.update_progress)
                self.worker_thread.finished.connect(self.process_completed)
                self.progress_value = 0
                self.timer = QTimer()
                self.timer.timeout.connect(self.update_progress)
                self.timer.start(2000)
                self.worker_thread.start()
                # print("output：", self.worker_thread.output_file)
                self.output_player.setSource(QUrl.fromLocalFile(self.worker_thread.output_file))



            else:
                QMessageBox.information(self, "oops", "请先上传一个文件", QMessageBox.Yes)
                widgets.running_status.setText("未工作")



        if btnName == "play_2":
            self.output_player.setSource(QUrl.fromLocalFile(self.worker_thread.output_file))
            self.output_player.play()
        if btnName == "pause_2":
            self.output_player.pause()

        # PRINT BTN NAME
        print(f'Button "{btnName}" pressed!')

    def update_progress(self):
        if self.progress_value < 99:
            self.progress_value += 1
            self.progressDialog.setValue(self.progress_value)

    # def processing(self):
    #     self.genre_list, self.tempo, self.file_list = finalPredict(self.filename)
    #     print(self.file_list)
    #     widgets.bpm.setText(f"bpm：{self.tempo}")
    #     widgets.genre.setText(f"风格检测：{self.genre_list}")
    #     widgets.drum_midi_dir.setText(f"鼓组midi文件目录：" + os.getcwd() + "\latest_drum.mid")
    #     widgets.label.setText(f"合成文件目录：{self.file_list[0]}")
    #     self.output_player.setSource(QUrl.fromLocalFile(self.file_list[0]))
    #     widgets.running_status.setText("生成完成！")
    #     QCoreApplication.processEvents()

    def process_completed(self):
        widgets.running_status.setText("生成完成！")
        self.progressDialog.setValue(100)
        self.progressDialog.close()

    # 当进度条被滑动或点击
    def on_slider_moved(self, value):
        self.player.setPosition(value)
    def on_slider_moved_output(self, value):
        self.output_player.setPosition(value)
    # 当时间变化改变进度条时间
    def on_update_slider(self, position):
        widgets.input_progress.setValue(position)
    def on_update_slider_output(self, position):
        widgets.output_progress.setValue(position)

    # 当音频文件状态改变时，更新进度条
    def on_media_changed(self, time):
        widgets.input_progress.setRange(0, time)
    def on_media_changed_output(self, time):
        widgets.output_progress.setRange(0, time)


    # 过滤获取进度条的点击事件
    def eventFilter(self, obj, event):
        if obj is QSlider:
            if self.player.PlaybackState not in (QMediaPlayer.MediaStatus.NoMedia, QMediaPlayer.MediaStatus.LoadingMedia):
                if event.type() == QEvent.Type.MouseButtonPress:
                    mouse_event = QMouseEvent(event)
                    if mouse_event.button() == Qt.MouseButton.LeftButton:
                        # 计算点击位置
                        range = widgets.input_progress.maximum()
                        width = widgets.input_progress.width()
                        pos = mouse_event.position().x() / width * range
                        self.player.setPosition(pos)
                        if self.player.PlaybackState == QMediaPlayer.MediaStatus.LoadedMedia:
                            self.player.play()
            elif self.output_player.PlaybackState not in (QMediaPlayer.MediaStatus.NoMedia, QMediaPlayer.MediaStatus.LoadingMedia):
                if event.type() == QEvent.Type.MouseButtonPress:
                    mouse_event = QMouseEvent(event)
                    if mouse_event.button() == Qt.MouseButton.LeftButton:
                        # 计算点击位置
                        range = widgets.output_progress.maximum()
                        width = widgets.output_progress.width()
                        pos = mouse_event.position().x() / width * range
                        self.output_player.setPosition(pos)
                        if self.output_player.PlaybackState == QMediaPlayer.MediaStatus.LoadedMedia:
                            self.output_player.play()

            else:
                widgets.input_progress.setValue(0)
                widgets.output_progress.setValue(0)

        return super().eventFilter(obj, event)


    # RESIZE EVENTS
    # ///////////////////////////////////////////////////////////////
    def resizeEvent(self, event):
        # Update Size Grips
        UIFunctions.resize_grips(self)

    # MOUSE CLICK EVENTS
    # ///////////////////////////////////////////////////////////////
    def mousePressEvent(self, event):
        # SET DRAG POS WINDOW
        self.dragPos = event.globalPos()

        # PRINT MOUSE EVENTS
        if event.buttons() == Qt.LeftButton:
            print('Mouse click: LEFT CLICK')
        if event.buttons() == Qt.RightButton:
            print('Mouse click: RIGHT CLICK')

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setWindowIcon(QIcon("icon.ico"))
    window = MainWindow()
    sys.exit(app.exec_())
