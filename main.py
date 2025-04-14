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
import time
import datetime
from typing import Optional

import PySide6.QtCore

# IMPORT / GUI AND MODULES AND WIDGETS
# ///////////////////////////////////////////////////////////////
from modules import *
from widgets import *
os.environ["QT_FONT_DPI"] = "96" # FIX Problem for High DPI and Scale above 100%


from model.all import *

# SET AS GLOBAL WIDGETS
# ///////////////////////////////////////////////////////////////
widgets = None

check_mail_times = 1


from PySide6.QtWidgets import *
from PySide6.QtCore import Signal, Slot

class Worker(QThread):
    finished = Signal()
    intReady = Signal(int)
    @Slot()
    def procCounter(self):
        # box = QMessageBox()
        # box.setText("procCounter called")
        # box.exec()
        
        for i in range(1, 10):
            print("Worker started!")
            time.sleep(1)
            self.intReady.emit(i)

        self.finished.emit()

    # @Slot()  # QtCore.Slot
    # def run(self):
    #     '''
    #     Your code goes in this function
    #     '''
    #     print("Thread start")
    #     time.sleep(5)
    #     print("Thread complete")

class MainWindow(QMainWindow):
    def __init__(self):
        QMainWindow.__init__(self)

        # SET AS GLOBAL WIDGETS
        # ///////////////////////////////////////////////////////////////
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        global widgets
        widgets = self.ui

        # USE CUSTOM TITLE BAR | USE AS "False" FOR MAC OR LINUX
        # ///////////////////////////////////////////////////////////////
        Settings.ENABLE_CUSTOM_TITLE_BAR = True

        # APP NAME
        # ///////////////////////////////////////////////////////////////
        title = "PyDracula - Modern GUI"
        description = "PyDracula APP - Theme with colors based on Dracula for Python."
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

        self.ui.tableWidget.setHorizontalHeaderLabels(["Accuracy", "Is Spam", "Time Taken"])
        self.ui.tableWidget.setItem(0, 0, QTableWidgetItem(str("Accuracy")))
        self.ui.tableWidget.setItem(0, 1, QTableWidgetItem(str("Is Spam")))
        self.ui.tableWidget.setItem(0, 2, QTableWidgetItem(str("Time Taken")))

        # BUTTONS CLICK
        # ///////////////////////////////////////////////////////////////

        # LEFT MENUS
        widgets.btn_home.clicked.connect(self.buttonClick)
        widgets.btn_widgets.clicked.connect(self.buttonClick)

        # 
        widgets.pushButton_openFile.clicked.connect(self.pushButton_openFileClicked)
        widgets.pushButton_checkMail.clicked.connect(self.pushButton_checkMailClicked)
        
        
        self.obj = Worker()
        self.thread = QThread()
        
        self.obj.moveToThread(self.thread)
        self.obj.intReady.connect(self.update_ready_ui)
        self.obj.finished.connect(self.update_finished_ui)
        self.thread.started.connect(self.obj.procCounter)
        

        # EXTRA LEFT BOX
        def openCloseLeftBox():
            UIFunctions.toggleLeftBox(self, True)
        # widgets.toggleLeftBox.clicked.connect(openCloseLeftBox)
        widgets.extraCloseColumnBtn.clicked.connect(openCloseLeftBox)

        # EXTRA RIGHT BOX
        def openCloseRightBox():
            UIFunctions.toggleRightBox(self, True)
        # widgets.settingsTopBtn.clicked.connect(openCloseRightBox)

        # SHOW APP
        # ///////////////////////////////////////////////////////////////
        self.show()

        # SET CUSTOM THEME
        # ///////////////////////////////////////////////////////////////
        useCustomTheme = False
        themeFile = "themes\py_dracula_light.qss"

        # SET THEME AND HACKS
        if useCustomTheme:
            # LOAD AND APPLY STYLE
            UIFunctions.theme(self, themeFile, True)

            # SET HACKS
            AppFunctions.setThemeHack(self)

        # SET HOME PAGE AND SELECT MENU
        # ///////////////////////////////////////////////////////////////
        widgets.stackedWidget.setCurrentWidget(widgets.home)
        widgets.btn_home.setStyleSheet(UIFunctions.selectMenu(widgets.btn_home.styleSheet()))

    def update_ready_ui(self, i):
        # box = QMessageBox()
        # box.setText("update_readd_ui called" + str(i))
        # box.exec()
        self.ui.plainTextEdit.appendPlainText(str(i) + "\n")
        # app.processEvents()
        
    @Slot()
    def update_finished_ui(self):
        self.ui.plainTextEdit.appendPlainText("Finished" + "\n")
        self.thread.quit()

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

        # PRINT BTN NAME
        print(f'Button "{btnName}" pressed!')
        
    def pushButton_openFileClicked(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "", "All Files (*);;Python Files (*.py)", options=options)

        if fileName:
            self.ui.lineEdit.setText(fileName)

    def pushButton_checkMailClicked(self):
        global check_mail_times

        welcome_message = display_welcome_message()

        # Create and display the message box
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Information)
        msg_box.setText(welcome_message)
        msg_box.setWindowTitle("Welcome Message")
        msg_box.setStandardButtons(QMessageBox.Ok)
        msg_box.exec()

        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.ui.plainTextEdit_modelOutput.appendPlainText(f"Current time: {current_time}")
        self.ui.plainTextEdit_modelOutput.appendPlainText(welcome_message)
        # show current pwd
        print(os.getcwd())

        # dataset path
        file_path = "emails.csv"        
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.ui.plainTextEdit_modelOutput.appendPlainText(f"Current time: {current_time}")
        self.ui.plainTextEdit_modelOutput.appendPlainText("Starting program with file: emails.csv")
    
        # Check if the file exists
        try:
            check_file_exists(file_path)
        except FileNotFoundError as e:
            print(e)
            return
    
        # Load the data
        try:
            df = load_data(file_path)
        except Exception as e:
            print(e)
            return
    
        # Validate the DataFrame
        try:
            validate_dataframe(df)
        except ValueError as e:
            print(e)
            return
    
        # Display data sample and summary
        display_data_sample(df)
        summarize_data(df)
    
        # Prepare data for training
        X_text = df['text']  # Email content
        y = df['spam']       # Labels: 1 for spam, 0 for not spam
    
        # Split the data
        X_train_text, X_test_text, y_train, y_test = split_data(X_text, y)
    
        # Vectorize the text data
        vectorizer, X_train, X_test = vectorize_text(X_train_text, X_test_text)

        # check if email is from file or string
        if self.ui.radioButton_inputFile.isChecked():
            print("File selected.")
            filename = self.ui.lineEdit.text()
            if not filename:
                print("No file selected.")
                return
            try:
                with open(filename, 'r') as file:
                    user_email = file.read()
            except Exception as e:
                print(e)
                return
        elif self.ui.radioButton_inputStrings.isChecked():
            print("String selected.")
            user_email = self.ui.plainTextEdit_mailInput.toPlainText()
        else:
            print("No input method selected.")
            # pop message waring
            warning_msg = QMessageBox()
            warning_msg.setIcon(QMessageBox.Warning)
            warning_msg.setText("Please select an input method (file or string).")
            warning_msg.setWindowTitle("Input Method Warning")
            warning_msg.setStandardButtons(QMessageBox.Ok)
            warning_msg.exec()
            return



        # Display the user email
        self.ui.plainTextEdit_modelOutput.appendPlainText("User email:")
        self.ui.plainTextEdit_modelOutput.appendPlainText(user_email)
        
    
        # Get and confirm user choice
        # choice = get_user_choice()
        # if not confirm_user_choice(choice):
        #     print("Operation cancelled by user.")
        #     display_farewell_message()
        #     return

        choice = self.ui.comboBox.currentText()

        accuracy = 0
        is_spam = 'False'
        time_taken = 0

    
        # Execute the chosen classifier
        if choice == "RandomForest":
            accuracy, is_spam, time_taken = run_random_forest(X_train, y_train, X_test, y_test, vectorizer, user_email, self.ui)
        elif choice == "SVM":
            accuracy, is_spam, time_taken =  run_svm(X_train, y_train, X_test, y_test, vectorizer, user_email, self.ui)
        elif choice == "NaiveBayes":
            accuracy, is_spam, time_taken = run_naive_bayes(X_train, y_train, X_test, y_test, vectorizer, user_email, self.ui)
        else:
            print("Invalid choice. Please enter 1, 2, or 3.")
            logger.warning(f"Invalid choice entered: {choice}")
    
        # Display farewell message
        # display_farewell_message()

        # self.ui.tableWidget.clear()
        # self.ui.tableWidget.setRowCount(1)
        # self.ui.tableWidget.setColumnCount(3)

        self.ui.tableWidget.setItem(check_mail_times, 0, QTableWidgetItem(str(accuracy)))
        self.ui.tableWidget.setItem(check_mail_times, 1, QTableWidgetItem(is_spam))
        self.ui.tableWidget.setItem(check_mail_times, 2, QTableWidgetItem(str(time_taken)))
        check_mail_times += 1


        



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
