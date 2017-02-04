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
from datetime import datetime, timedelta
import sys
from collections import OrderedDict
from enum import Enum
from Gui.gui import *
from PyQt5.QtWidgets import *
from Gui.GuiFunctions import PandasModel
import pandas as pd
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from multiprocessing import cpu_count

from Engine.microgrid_dispatch import *

__author__ = 'Santiago Peñate Vera'

"""
This class is the handler of the main gui of GridCal.
"""

########################################################################################################################
# Main Window
########################################################################################################################


class MainGUI(QMainWindow):
    # Prices
    prices = None

    # solar irradiation in W/m2
    solar_radiation_profile = None

    # wind speed in m/s
    wind_speed_profile = None

    # load the wind turbine power curve
    ag_curve_df = None

    # Normalized demand values (negative for the sign convention)
    demand_profile = None

    time = None

    micro_grid = None

    #
    project_directory = None

    def __init__(self, parent=None):
        """

        @param parent:
        """

        # create main window
        QWidget.__init__(self, parent)
        self.ui = Ui_mainWindow()
        self.ui.setupUi(self)

        self.available_results = list()
        self.available_results.append('Optimization plot')
        self.available_results.append('Storage power')
        self.available_results.append('Aggregated power profile')
        self.available_results.append('Power to the battery')
        self.available_results.append('Power demanded to the grid')
        self.available_results.append('Battery power effectively processed')
        self.available_results.append('Battery state of charge')
        self.available_results.append('LCOE vs. Grid energy')
        self.available_results.append('LCOE vs. Investment')
        self.ui.results_comboBox.addItems(self.available_results)

        self.obj_fun_dict = dict()

        self.obj_fun_dict['LCOE'] = ObjectiveFunctionType.LCOE
        self.obj_fun_dict['Grid usage'] = ObjectiveFunctionType.GridUsage
        self.obj_fun_dict['Grid usage cost'] = ObjectiveFunctionType.GridUsageCost
        self.obj_fun_dict['Grid usage cost x LCOE'] = ObjectiveFunctionType.GridUsageCost_times_LCOE
        k = list(self.obj_fun_dict.keys())
        k.sort()
        self.ui.obj_function_comboBox.addItems(k)


        ################################################################################################################
        # Connections
        ################################################################################################################
        self.ui.actionNew_project.triggered.connect(self.new_project)

        self.ui.actionOpen_file.triggered.connect(self.open_file)

        self.ui.actionSave.triggered.connect(self.save_file)

        # # Buttons
        #
        self.ui.demand_show_profile_pushButton.clicked.connect(self.plot_demand)

        self.ui.wind_farm_show_profile_pushButton.clicked.connect(self.plot_wind)

        self.ui.solar_farm_show_profile_pushButton.clicked.connect(self.plot_solar)

        self.ui.wind_turbine_show_curve_pushButton.clicked.connect(self.plot_ag_curve)

        self.ui.size_simulate_pushButton.clicked.connect(self.size_devices)

        self.ui.plot_results_pushButton.clicked.connect(self.plot_results)

        ################################################################################################################
        # Connections
        ################################################################################################################

        self.unlock()

        self.set_default_ui_values()

    def set_ui_state(self, val=True):
        self.ui.progress_frame.setVisible(val)

    def lock(self):
        self.set_ui_state(True)

    def unlock(self):
        self.set_ui_state(False)

    def set_default_ui_values(self):
        """

        :return:
        """
        # demand
        self.ui.demand_nominal_power_doubleSpinBox.setValue(1000)
        self.ui.start_dateEdit.setDate(QtCore.QDate(datetime.today().year, 1, 1))
        self.ui.investment_years_spinBox.setValue(20)
        self.ui.interest_rate_doubleSpinBox.setValue(0.03)

        # wind
        self.ui.wind_farm_min_nominal_power_doubleSpinBox.setValue(0)
        self.ui.wind_farm_nominal_power_doubleSpinBox.setValue(1000)
        self.ui.wind_farm_unitary_cost_doubleSpinBox.setValue(900)

        # solar
        self.ui.solar_farm_min_nominal_power_doubleSpinBox.setValue(0)
        self.ui.solar_farm_nominal_power_doubleSpinBox.setValue(1000)
        self.ui.solar_farm_unitary_cost_doubleSpinBox.setValue(200)

        # storage
        self.ui.storage_min_nominal_energy_doubleSpinBox.setValue(0)
        self.ui.storage_max_nominal_energy_doubleSpinBox.setValue(10000)
        self.ui.storage_max_unitary_cost_doubleSpinBox.setValue(900)
        self.ui.storage_max_soc_doubleSpinBox.setValue(0.98)
        self.ui.storage_min_soc_doubleSpinBox.setValue(0.3)
        self.ui.storage_charge_eff_doubleSpinBox.setValue(0.8)
        self.ui.storage_discharge_eff_doubleSpinBox.setValue(0.8)

    def make_simulation_object(self):
        """

        :return:
        """

        # assert(self.demand_profile is not None)
        if self.demand_profile is None:
            self.msg('There are no profiles')
            return

        nominal_power = self.ui.demand_nominal_power_doubleSpinBox.value()
        start = self.ui.start_dateEdit.dateTime().toPyDateTime()
        investment_years = self.ui.investment_years_spinBox.value()
        investment_rate = self.ui.interest_rate_doubleSpinBox.value()
        demand = Demand(self.demand_profile,
                        nominal_power=nominal_power)

        # wind
        wind_power_min = self.ui.wind_farm_min_nominal_power_doubleSpinBox.value()
        wind_power_max = self.ui.wind_farm_nominal_power_doubleSpinBox.value()
        unitary_cost = self.ui.wind_farm_unitary_cost_doubleSpinBox.value()
        wind_farm = WindFarm(self.wind_speed_profile, self.ag_curve_df,
                             wind_power_min=wind_power_min,
                             wind_power_max=wind_power_max,
                             unitary_cost=unitary_cost)

        # solar
        solar_power_min = self.ui.solar_farm_min_nominal_power_doubleSpinBox.value()
        solar_power_max = self.ui.solar_farm_nominal_power_doubleSpinBox.value()
        unitary_cost = self.ui.solar_farm_unitary_cost_doubleSpinBox.value()
        solar_farm = SolarFarm(self.solar_radiation_profile,
                               solar_power_min=solar_power_min,
                               solar_power_max=solar_power_max,
                               unitary_cost=unitary_cost)

        # storage
        battery_energy_min = self.ui.storage_min_nominal_energy_doubleSpinBox.value()
        battery_energy_max = self.ui.storage_max_nominal_energy_doubleSpinBox.value()
        unitary_cost = self.ui.storage_max_unitary_cost_doubleSpinBox.value()
        max_soc = self.ui.storage_max_soc_doubleSpinBox.value()
        min_soc = self.ui.storage_min_soc_doubleSpinBox.value()
        charge_efficiency = self.ui.storage_charge_eff_doubleSpinBox.value()
        discharge_efficiency = self.ui.storage_discharge_eff_doubleSpinBox.value()
        battery = BatterySystem(charge_efficiency=charge_efficiency,
                                discharge_efficiency=discharge_efficiency,
                                max_soc=max_soc,
                                min_soc=min_soc,
                                battery_energy_min=battery_energy_min,
                                battery_energy_max=battery_energy_max,
                                unitary_cost=unitary_cost)

        nt = len(self.demand_profile)
        start = self.ui.start_dateEdit.dateTime().toPyDateTime()
        self.time = [start + timedelta(hours=h) for h in range(nt)]

        max_eval = self.ui.max_eval_spinBox.value()

        # simulation type
        sel = self.ui.obj_function_comboBox.currentText()
        obj_fun_type = self.obj_fun_dict[sel]

        if self.ui.brute_force_radioButton.isChecked():

            self.micro_grid = MicroGridBrute(solar_farm=solar_farm,
                                             wind_farm=wind_farm,
                                             battery_system=battery,
                                             demand_system=demand,
                                             time_arr=self.time,
                                             LCOE_years=investment_years,
                                             spot_price=self.prices[:, 0] / 1000,
                                             band_price=self.prices[:, 1] / 1000,
                                             maxeval=max_eval,
                                             obj_fun_type=obj_fun_type)
        else:

            self.micro_grid = MicroGrid(solar_farm=solar_farm,
                                        wind_farm=wind_farm,
                                        battery_system=battery,
                                        demand_system=demand,
                                        time_arr=self.time,
                                        LCOE_years=investment_years,
                                        spot_price=self.prices[:, 0] / 1000,
                                        band_price=self.prices[:, 1] / 1000,
                                        maxeval=max_eval,
                                        obj_fun_type=obj_fun_type)

    def new_project(self):
        print('new_project')

    def open_file(self):
        """

        :return:
        """

        # declare the allowed file types
        files_types = "Excel 97 (*.xls);;Excel (*.xlsx)"
        # call dialog to select the file

        filename, type_selected = QFileDialog.getOpenFileName(self, 'Open file',
                                                              directory=self.project_directory,
                                                              filter=files_types)

        if len(filename) > 0:
            # load file
            xl = pd.ExcelFile(filename)

            self.project_directory = os.path.dirname(filename)

            # assert that the requires sheets exist. This sort of determines if the excel file is the right one
            c1 = 'prices' in xl.sheet_names
            c2 = 'irradiation' in xl.sheet_names
            c3 = 'wind' in xl.sheet_names
            c4 = 'AG_CAT' in xl.sheet_names
            c5 = 'demand' in xl.sheet_names

            cond = c1 and c2 and c3 and c4 and c5

            if cond:

                # store the working directory
                self.prices = xl.parse(sheetname='prices')[['Secondary_reg_price', 'Spot_price']].values

                # load the solar irradiation in W/M2 and convert it to kW
                self.solar_radiation_profile = xl.parse(sheetname='irradiation')['irradiation (MW/m2)'].values

                # Load the wind speed in m/s
                self.wind_speed_profile = xl.parse(sheetname='wind')['VEL(m/s):60'].values

                # load the wind turbine power curve and normalize it
                self.ag_curve_df = xl.parse(sheetname='AG_CAT')['P (kW)']

                # load the demand values and set it negative for the sign convention
                self.demand_profile = xl.parse(sheetname='demand')['normalized_demand'].values

                nt = len(self.demand_profile)
                start = self.ui.start_dateEdit.dateTime().toPyDateTime()
                self.time = [start + timedelta(hours=h) for h in range(nt)]

                self.plot_demand()
            else:

                self.msg('The file format is not right.')

    def save_results(self):
        """

        :return:
        """
        if self.micro_grid is not None:
            self.micro_grid.x_fx.to_excel('results.xlsx')

    def size_devices(self):
        """

        :return:
        """

        # create a micro grid object
        self.make_simulation_object()

        if self.micro_grid is not None:
            print('Working on it...')
            self.lock()

            # make connections
            self.micro_grid.progress_signal.connect(self.ui.progressBar.setValue)
            self.micro_grid.done_signal.connect(self.unlock)
            self.micro_grid.done_signal.connect(self.post_size_devices)

            # thread start
            self.micro_grid.start()

    def post_size_devices(self):
        """
        Actions to run after the optimization
        :return:
        """
        # set the solution as the current micro grid state
        res = self.micro_grid(self.micro_grid.solution, verbose=True)

        print('Done!')

        # self.micro_grid.plot()
        # plt.plot()
        self.plot_text_results()
        self.plot_results()
        self.ui.results_tableView.setModel(PandasModel(self.micro_grid.x_fx))

        years_arr = [10, 15, 20, 25, 30, 40]
        inv_rate_arr = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1]
        df_lcoe = self.micro_grid.economic_sensitivity(years_arr, inv_rate_arr)
        self.ui.economic_tableView.setModel(PandasModel(df_lcoe))

    def plot_text_results(self):
        """
        Print the test results
        :return:
        """

        self.ui.plainTextEdit.clear()

        inv_map = {v: k for k, v in self.obj_fun_dict.items()}

        val = 'Conditions: \n\n'

        val += 'Amortization years:\t' + '{0:.0f}'.format(self.micro_grid.lcoe_years) + '\n'
        val += 'Discount rate:\t' + '{0:.2f}'.format(self.micro_grid.investment_rate) + '\n'
        val += 'Optimizing for:\t' + inv_map[self.micro_grid.obj_fun_type] + '\n'

        val += '\n'

        val += 'Results: \n\n'

        val += 'Demand size:\t' + '{0:.2f}'.format(self.micro_grid.demand_system.nominal_power) + ' kW.\n'
        val += 'Solar farm size:\t' + '{0:.2f}'.format(self.micro_grid.solution[0]) + ' kW.\n'
        val += 'Wind farm size:\t' + '{0:.2f}'.format(self.micro_grid.solution[1]) + ' kW.\n'
        val += 'Storage size:\t' + '{0:.2f}'.format(self.micro_grid.solution[2]) + ' kWh.\n'

        val += '\n'

        val += 'Solar farm cost:\t' + '{0:.2f}'.format(self.micro_grid.solar_farm.cost()) + ' €.\n'
        val += 'Wind farm cost:\t' + '{0:.2f}'.format(self.micro_grid.wind_farm.cost()) + ' €.\n'
        val += 'Storage cost:\t' + '{0:.2f}'.format(self.micro_grid.battery_system.cost()) + ' €.\n'

        val += '\n'

        val += 'Grid energy:\t' + '{0:.2f}'.format(self.micro_grid.grid_energy) + ' kWh.\n'
        val += 'Energy cost:\t' + '{0:.2f}'.format(self.micro_grid.energy_cost) + ' €.\n'
        val += 'investment_cost:\t' + '{0:.2f}'.format(self.micro_grid.investment_cost) + ' €.\n'
        val += 'LCOE:\t\t' + '{0:.2f}'.format(self.micro_grid.lcoe_val) + ' €/kWh.\n'

        self.ui.plainTextEdit.setPlainText(val)

    def plot_results(self):
        """
        Plot the simulation results
        :return:
        """

        if self.micro_grid is not None:

            sel = self.ui.results_comboBox.currentText()
            self.ui.resultsPlot.clear(force=True)
            ax = self.ui.resultsPlot.get_axis()
            fig = self.ui.resultsPlot.get_figure()

            if sel == 'Optimization plot':

                self.micro_grid.plot_optimization(ax=ax)

            elif sel == 'Storage power':

                self.micro_grid.battery_system.plot(fig=fig)

            elif sel == 'Aggregated power profile':

                ax.plot(self.micro_grid.aggregated_demand_profile, label=sel)
                ax.set_ylabel('kW')
                ax.set_title(sel)
                ax.legend()

            elif sel == 'Power to the battery':

                ax.plot(self.micro_grid.battery_state_of_charge, label=sel)
                ax.set_ylabel('kW')
                ax.set_title(sel)
                ax.legend()

            elif sel == 'Power demanded to the grid':

                ax.plot(self.micro_grid.grid_power, label=sel)
                ax.set_ylabel('kW')
                ax.set_title(sel)
                ax.legend()

            elif sel == 'Battery power effectively processed':

                ax.plot(self.micro_grid.battery_state_of_charge, label=sel)
                ax.set_ylabel('kW')
                ax.set_title(sel)
                ax.legend()

            elif sel == 'Battery state of charge':

                ax.plot(self.micro_grid.battery_state_of_charge, label=sel)
                ax.set_ylabel('Per unit')
                ax.set_title(sel)
                ax.legend()

            elif sel == 'LCOE vs. Grid energy':
                x = self.micro_grid.x_fx['Grid energy'].values
                y = self.micro_grid.x_fx['LCOE'].values
                ax.scatter(x, y)
                ax.set_xlabel('Grid energy [kWh]')
                ax.set_ylabel('LCOE [€/kWh]')
                ax.set_title(sel)
                ax.legend()

            elif sel == 'LCOE vs. Investment':
                x = self.micro_grid.x_fx['Investment'].values
                y = self.micro_grid.x_fx['LCOE'].values
                ax.scatter(x, y)
                ax.set_xlabel('Investment [€]')
                ax.set_ylabel('LCOE [€/kWh]')
                ax.set_title(sel)
                ax.legend()

            else:
                print()

            self.ui.resultsPlot.redraw()

    def plot_input(self, arr, ylabel='', xlabel='', title=''):
        """

        :param arr:
        :param ylabel:
        :param xlabel:
        :param title:
        :return:
        """
        self.ui.inputsPlot.clear()
        ax = self.ui.inputsPlot.get_axis()

        ax.plot(self.time, arr)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        self.ui.inputsPlot.redraw()

    def plot_demand(self):
        """

        :return:
        """
        if self.demand_profile is not None:
            self.plot_input(self.demand_profile, ylabel='kW', xlabel='Time', title='Demand')

    def plot_wind(self):
        """

        :return:
        """
        if self.wind_speed_profile is not None:
            self.plot_input(self.wind_speed_profile, ylabel='m/s', xlabel='Time', title='Wind speed')

    def plot_ag_curve(self):
        """

        :return:
        """
        if self.ag_curve_df is not None:
            self.ui.inputsPlot.clear()
            ax = self.ui.inputsPlot.get_axis()

            ax.plot(self.ag_curve_df)
            ax.set_xlabel('wind speed')
            ax.set_ylabel('kW')
            ax.set_title('Wind turbine curve')

            self.ui.inputsPlot.redraw()

    def plot_solar(self):
        """

        :return:
        """
        if self.solar_radiation_profile is not None:
            self.plot_input(self.solar_radiation_profile, ylabel='W/m^2', xlabel='Time', title='Irradiation')

    def msg(self, text):
        """
        Message box
        :param text:
        :return:
        """
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Information)
        msg.setText(text)
        # msg.setInformativeText("This is additional information")
        msg.setWindowTitle("Aviso")
        # msg.setDetailedText("The details are as follows:")
        msg.setStandardButtons(QMessageBox.Ok)
        retval = msg.exec_()


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
