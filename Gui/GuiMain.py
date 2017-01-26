# This file is part of GridCal.
#
# GridCal is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GridCal is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GridCal.  If not, see <http://www.gnu.org/licenses/>.

import os.path
import sys
from collections import OrderedDict
from enum import Enum
from Gui.gui import *
from PyQt5.QtWidgets import *
from matplotlib.colors import LinearSegmentedColormap
from multiprocessing import cpu_count

__author__ = 'Santiago Peñate Vera'

"""
This class is the handler of the main gui of GridCal.
"""

########################################################################################################################
# Main Window
########################################################################################################################


class MainGUI(QMainWindow):

    def __init__(self, parent=None):
        """

        @param parent:
        """

        # create main window
        QWidget.__init__(self, parent)
        self.ui = Ui_mainWindow()
        self.ui.setupUi(self)

        ################################################################################################################
        # Connections
        ################################################################################################################
        self.ui.actionNew_project.triggered.connect(self.new_project)

        self.ui.actionOpen_file.triggered.connect(self.open_file)

        self.ui.actionSave.triggered.connect(self.save_file)

        # # Buttons
        #
        # self.ui.cancelButton.clicked.connect(self.set_cancel_state)
        #
        # self.ui.new_profiles_structure_pushButton.clicked.connect(self.new_profiles_structure)
        #
        # self.ui.delete_profiles_structure_pushButton.clicked.connect(self.delete_profiles_structure)
        #
        # self.ui.set_profile_state_button.clicked.connect(self.set_profiles_state_to_grid)
        #
        # self.ui.profile_import_pushButton.clicked.connect(self.import_profiles)
        #
        # self.ui.profile_display_pushButton.clicked.connect(self.display_profiles)
        #
        # self.ui.plot_pushButton.clicked.connect(self.item_results_plot)
        #
        # self.ui.select_all_pushButton.clicked.connect(self.ckeck_all_result_objects)
        #
        # self.ui.select_none_pushButton.clicked.connect(self.ckeck_none_result_objects)
        #
        # self.ui.saveResultsButton.clicked.connect(self.save_results_df)
        #
        # self.ui.set_profile_state_button.clicked.connect(self.set_state)
        #
        # self.ui.setValueToColumnButton.clicked.connect(self.set_value_to_column)
        #
        # # node size
        # self.ui.actionBigger_nodes.triggered.connect(self.grid_editor.bigger_nodes)
        #
        # self.ui.actionSmaller_nodes.triggered.connect(self.grid_editor.smaller_nodes)
        #
        # self.ui.actionCenter_view.triggered.connect(self.grid_editor.center_nodes)
        #
        # # list clicks
        # self.ui.result_listView.clicked.connect(self.update_available_results_in_the_study)
        # self.ui.result_type_listView.clicked.connect(self.result_type_click)
        #
        # self.ui.dataStructuresListView.clicked.connect(self.view_objects_data)
        #
        # # combobox
        # self.ui.profile_device_type_comboBox.currentTextChanged.connect(self.profile_device_type_changed)

    def new_project(self):
        print('new_project')

    def open_file(self):
        print('open_file')

    def save_file(self):
        print('save_file')


def run():
    app = QApplication(sys.argv)
    window = MainGUI()
    window.resize(1.61 * 700, 700)  # golden ratio
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    run()
