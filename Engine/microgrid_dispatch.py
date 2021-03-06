import pandas as pd
import numpy as np
from itertools import product
from enum import Enum
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from PyQt5.QtCore import QThread, QRunnable, pyqtSignal
from pySOT import *
from poap.controller import ThreadController, BasicWorkerThread, SerialController
plt.style.use('ggplot')


class ObjectiveFunctionType(Enum):
    LCOE = 0,
    GridUsage = 1,
    GridUsageCost = 2,
    GridUsageCost_times_LCOE = 3


class Demand:

    def __init__(self, profile, nominal_power):
        """

        Args:
            profile: scale profile of the demand (all ones, for no change)
            nominal_power: Nominal power of the demand facility
        """
        self.index = None

        self.nominal_power = nominal_power

        self.normalized_power = profile * -1  # the convention is that loads are negative

    def power(self):
        """
        Returns the demanded power
        Returns: Array
        """
        return self.nominal_power * self.normalized_power


class SolarFarm:

    def __init__(self, profile, solar_power_min=0, solar_power_max=10000, unitary_cost=200):
        """

        Args:
            profile: Solar horizontal irradiation profile (W/m2) [1D array]
            solar_power_max: Maximum power in kW to consider when sizing
            unitary_cost: Cost peer installed kW of the solar facility (€/kW)
        """
        self.index = None

        self.nominal_power = None

        self.irradiation = profile

        self.normalized_power = profile / 1000.0

        self.max_power = solar_power_max

        self.min_power = solar_power_min

        self.unitary_cost = unitary_cost

    def power(self):
        """
        Returns the generated power
        Returns: Array
        """
        return self.nominal_power * self.normalized_power

    def cost(self):
        return self.unitary_cost * self.nominal_power



class WindFarm:

    def __init__(self, profile, wt_curve_df, wind_power_min=0, wind_power_max=10000, unitary_cost=900):
        """

        Args:
            profile: Wind profile in m/s
            wt_curve_df: Wind turbine power curve in a DataFrame (Power [any unit, values] vs. Wind speed [m/s, index])
            wind_power_max: Maximum nominal power of the wind park considered when sizing
            unitary_cost: Unitary cost of the wind park in €/kW

            Example of unitary cost:
            A wind park with 4 turbines of 660 kW cost 2 400 000 €
            2400000 / (4 * 660) = 909 €/kW installed
        """
        self.index = None

        self.nominal_power = None

        self.wt_curve_df = wt_curve_df

        # load the wind turbine power curve and normalize it
        ag_curve = interp1d(wt_curve_df.index, wt_curve_df.values / wt_curve_df.values.max())

        self.wind_speed = profile

        self.normalized_power = ag_curve(profile)

        self.max_power = wind_power_max

        self.min_power = wind_power_min

        self.unitary_cost = unitary_cost

    def power(self):
        """
        Returns the generated power
        Returns: Array
        """
        return self.nominal_power * self.normalized_power

    def cost(self):
        return self.unitary_cost * self.nominal_power


class BatterySystem:
    demanded_power = None
    energy = None
    power = None
    grid_power = None
    soc = None
    time = None

    def __init__(self, charge_efficiency=0.9, discharge_efficiency=0.9, max_soc=0.99, min_soc=0.3,
                 battery_energy_min=0, battery_energy_max=100000, unitary_cost=900):
        """

        Args:
            charge_efficiency: Efficiency when charging
            discharge_efficiency:  Efficiency when discharging
            max_soc: Maximum state of charge
            min_soc: Minimum state of charge
            battery_energy_max: Maximum energy in kWh allowed for sizing the battery
            unitary_cost: Cost per kWh of the battery (€/kWh)
        """

        self.index = None

        self.nominal_energy = None

        self.charge_efficiency = charge_efficiency

        self.discharge_efficiency = discharge_efficiency

        self.max_soc = max_soc

        self.min_soc = min_soc

        self.min_soc_charge = (self.max_soc + self.min_soc) / 2  # SoC state to force the battery charge

        self.charge_per_cycle = 0.1  # charge 10% per cycle

        self.max_energy = battery_energy_max

        self.min_energy = battery_energy_min

        self.unitary_cost = unitary_cost

        self.results = None

    def cost(self):
        return self.unitary_cost * self.nominal_energy

    def simulate_array(self, P, soc_0, time, charge_if_needed=False):
        """
        The storage signs are the following

        supply power: positive
        recharge power: negative

        this means that a negative power will ask the battery to charge and
        a positive power will ask the battery to discharge

        to match these signs to the give profiles, we should invert the
        profiles sign
        Args:
            P: Power array that is sent to the battery [Negative charge, positive discharge]
            soc_0: State of charge at the beginning [0~1]
            time: Array of DataTime values
            charge_if_needed: Allow the battery to take extra power that is not given in P. This limits the growth of
            the battery system in the optimization since the bigger the battery, the more grid power it will take to
            charge when RES cannot cope. Hence, since we're minimizing the grid usage, there is an optimum battery size
        Returns:
            energy: Energy effectively processed by the battery
            power: Power effectively processed by the battery
            grid_power: Power dumped array
            soc: Battery state of charge array
        """

        if self.nominal_energy is None:
            raise Exception('You need to set the battery nominal power!')

        # initialize arrays
        P = np.array(P)
        nt = len(P)
        energy = np.zeros(nt + 1)
        power = np.zeros(nt + 1)
        soc = np.zeros(nt + 1)
        grid_power = np.zeros(nt + 1)
        energy[0] = self.nominal_energy * soc_0
        soc[0] = soc_0

        charge_energy_per_cycle = self.nominal_energy * self.charge_per_cycle

        for t in range(nt-1):

            if np.isnan(P[t]):
                print('NaN found!!!!!!')

            # pick the right efficiency value
            if P[t] >= 0:
                eff = self.discharge_efficiency
            else:
                eff = self.charge_efficiency

            # the time comes in nanoseconds, we need the time step in hours
            dt = (time[t + 1] - time[t]).seconds / 3600

            # compute the proposed energy. Later we check how much is actually possible
            proposed_energy = energy[t] - P[t] * dt * eff

            # charge the battery from the grid if the SoC is too low and we are allowing this behaviour
            if charge_if_needed and soc[t] < self.min_soc_charge:
                proposed_energy -= charge_energy_per_cycle / dt  # negative is for charging

            # Check the proposed energy
            if proposed_energy > self.nominal_energy * self.max_soc:  # Truncated, too high

                energy[t + 1] = self.nominal_energy * self.max_soc
                power[t + 1] = (energy[t]-energy[t + 1]) / (dt * eff)
                grid_power[t + 1] = - power[t + 1] + P[t]

            elif proposed_energy < self.nominal_energy * self.min_soc:  # Truncated, too low

                energy[t + 1] = self.nominal_energy * self.min_soc
                power[t + 1] = (energy[t]-energy[t + 1]) / (dt * eff)
                grid_power[t + 1] = - power[t + 1] + P[t]

            else:  # everything is within boundaries

                energy[t + 1] = proposed_energy
                power[t + 1] = P[t]
                grid_power[t + 1] = 0

            # Update the state of charge
            soc[t + 1] = energy[t + 1] / self.nominal_energy

        # Compose a results DataFrame
        self.demanded_power = np.r_[0, P[:-1]]
        self.energy = energy[:-1]
        self.power = power[:-1]
        self.grid_power = grid_power[:-1]
        self.soc = soc[:-1]
        self.time = time

        # d = np.c_[np.r_[0, P[:-1]], power[:-1], grid_power[:-1], energy[:-1], soc[:-1] * 100]
        # cols = ['P request', 'P', 'grid', 'E', 'SoC']
        # self.results = pd.DataFrame(data=d, columns=cols)

        return energy[:-1], power[:-1], grid_power[:-1], soc[:-1]

    def plot(self, fig=None):

        if fig is None:
            fig = plt.figure(figsize=(12, 8))
        ax1 = fig.add_subplot(111)
        ax1.stackplot(self.time, self.power, self.grid_power)
        ax1.plot(self.time, self.demanded_power, linewidth=4)
        ax1.set_ylabel('kW')

        ax2 = ax1.twinx()
        ax2.plot(self.time, self.soc, color='k')
        ax2.set_ylabel('SoC')

        ax1.legend()
        plt.show()


class MicroGrid(QThread):

    progress_signal = pyqtSignal(float)
    progress_text_signal = pyqtSignal(str)
    done_signal = pyqtSignal()

    dim = 3  # 3 variables to optimize

    def __init__(self, solar_farm: SolarFarm, wind_farm: WindFarm, demand_system: Demand, battery_system: BatterySystem,
                 time_arr, LCOE_years=20, investment_rate=0.03, spot_price=None, band_price=None, maxeval=1000,
                 obj_fun_type: ObjectiveFunctionType=ObjectiveFunctionType.GridUsage):
        """

        :param solar_farm:
        :param wind_farm:
        :param demand_system:
        :param battery_system:
        :param time_arr:
        :param LCOE_years:
        :param investment_rate:
        :param spot_price:
        :param band_price:
        :param maxeval:
        :param obj_fun_type:
        """

        QThread.__init__(self)

        # variables for the optimization
        self.xlow = np.zeros(self.dim)  # lower bounds
        self.xup = np.array([solar_farm.max_power, wind_farm.max_power, battery_system.max_energy])
        self.info = "Microgrid with Wind turbines, Photovoltaic panels and storage coupled to a demand"  # info
        self.integer = np.array([0])  # integer variables
        self.continuous = np.arange(1, self.dim)  # continuous variables

        # assign the device list
        self.solar_farm = solar_farm

        self.wind_farm = wind_farm

        self.demand_system = demand_system

        self.battery_system = battery_system

        self.spot_price = spot_price

        self.band_price = band_price

        self.obj_fun_type = obj_fun_type

        # create a time index matching the length
        self.time = time_arr

        # economic variables
        self.lcoe_years = LCOE_years

        self.investment_rate = investment_rate

        # Results

        self.aggregated_demand_profile = None

        self.solar_power_profile = None

        self.wind_power_profile = None

        self.grid_power = None

        self.Energy = None
        self.battery_output_power = None
        self.battery_output_current = None
        self.battery_voltage = None
        self.battery_losses = None
        self.battery_state_of_charge = None
        self.iteration = None
        self.max_eval = maxeval

        self.optimization_values = None
        self.raw_results = None
        self.solution = None

        self.grid_energy = None
        self.energy_cost = None
        self.investment_cost = None
        self.lcoe_val = None

        self.x_fx = list()

    def __call__(self, x, verbose=False):
        """
        Call for this object, performs the dispatch given a vector x of facility sizes
        Args:
            x: vector [solar nominal power, wind nominal power, storage nominal power]

        Returns: Value of the objective function for the given x vector

        """
        return self.objfunction(x, verbose)

    def objfunction(self, x, verbose=False):
        """

        Args:
            x: optimsation vector [solar nominal power, wind nominal power, storage nominal power]

        Returns:

        """
        ################################################################################################################
        # Set the devices nominal power
        ################################################################################################################
        self.solar_farm.nominal_power = x[0]

        self.wind_farm.nominal_power = x[1]

        self.battery_system.nominal_energy = x[2]

        '''
        The profiles sign as given are:

            demand: negative
            generation: positive
        '''

        ################################################################################################################
        # Compute the battery desired profile
        ################################################################################################################

        self.aggregated_demand_profile = self.demand_system.power() + self.wind_farm.power() + self.solar_farm.power()

        ################################################################################################################
        # Compute the battery real profile: processing the desired profile
        ################################################################################################################

        '''
        The storage signs are the following

        supply power: positive
        recharge power: negative

        this means that a negative power will ask the battery to charge and
        a positive power will ask the battery to discharge

        to match these signs to the give profiles, we should invert the
        profiles sign
        '''

        # reverse the power profile sign (the surplus, should go in the battery)
        demanded_power = - self.aggregated_demand_profile
        # initial state of charge
        SoC0 = 0.5

        # calculate the battery values: process the desired power
        # energy, power, grid_power, soc
        self.Energy, \
        self.battery_output_power, \
        self.grid_power, \
        self.battery_state_of_charge = self.battery_system.simulate_array(P=demanded_power, soc_0=SoC0,
                                                                          time=self.time, charge_if_needed=True)

        # the processed values are 1 value shorter since we have worked with time increments

        # calculate the grid power as the difference of the battery power
        # and the profile required for perfect auto-consumption
        # self.grid_power = demanded_power - self.battery_output_power

        # compute the investment cost
        self.investment_cost = self.solar_farm.cost() + self.wind_farm.cost() + self.battery_system.cost()

        # compute the LCOE Levelized Cost Of Electricity
        res = self.lcoe(generated_power_profile=self.grid_power, investment_cost=self.investment_cost,
                        discount_rate=self.investment_rate, years=self.lcoe_years, verbose=verbose)
        self.grid_energy = res[0]
        self.energy_cost = res[1]
        self.investment_cost = res[2]
        self.lcoe_val = res[3]

        # select the objective function
        if self.obj_fun_type is ObjectiveFunctionType.LCOE:
            fx = abs(self.lcoe_val)
        elif self.obj_fun_type is ObjectiveFunctionType.GridUsage:
            fx = sum(abs(self.grid_power))
        elif self.obj_fun_type is ObjectiveFunctionType.GridUsageCost:
            fx = sum(abs(self.grid_power * self.spot_price))
        elif self.obj_fun_type is ObjectiveFunctionType.GridUsageCost_times_LCOE:
            fx = sum(abs(self.grid_power * self.spot_price)) * (1-abs(self.lcoe_val))
        else:
            fx = 0

        # store the values
        self.raw_results.append([fx] + list(x) + [self.lcoe_val, sum(abs(self.grid_power)), self.investment_cost])

        self.iteration += 1
        prog = self.iteration / self.max_eval
        # print('progress:', prog)
        self.progress_signal.emit(prog * 100)

        return fx

    def lcoe(self, generated_power_profile, investment_cost, discount_rate, years, verbose=False):
        """

        :param generated_power_profile:
        :param investment_cost:
        :param discount_rate:
        :param verbose:
        :return:
        """
        grid_energy = generated_power_profile.sum()
        energy_cost = (generated_power_profile * self.spot_price).sum()

        # build the arrays for the n years
        I = np.zeros(years)  # investment
        I[0] = investment_cost
        E = np.ones(years) * grid_energy  # gains/cost of electricity
        M = np.ones(years) * investment_cost * 0.1  # cost of maintenance

        dr = np.array([(1 + discount_rate)**(i+1) for i in range(years)])
        A = (I + M / dr).sum()
        B = (E / dr).sum()

        if verbose:
            print('Grid energy', grid_energy, 'kWh')
            print('Energy cost', energy_cost, '€')
            print('investment_cost', investment_cost, '€')
            print('dr', dr)
            print('A:', A, 'B:', B)
            print('lcoe_val', A/B)

        # self.grid_energy = grid_energy
        # self.energy_cost = energy_cost
        # self.investment_cost = investment_cost
        lcoe_val = A / B

        return grid_energy, energy_cost, investment_cost, lcoe_val

    def economic_sensitivity(self, years_arr, inv_rate_arr):
        """

        :param years_arr:
        :param inv_rate_arr:
        :return:
        """
        ny = len(years_arr)
        ni = len(inv_rate_arr)
        values = np.zeros((4, ni, ny))

        for i, years in enumerate(years_arr):
            for j, inv_rate in enumerate(inv_rate_arr):

                grid_energy, \
                energy_cost, \
                investment_cost, \
                lcoe_val = self.lcoe(generated_power_profile=self.grid_power,
                                     investment_cost=self.investment_cost,
                                     discount_rate=inv_rate,
                                     years=years,
                                     verbose=False)

                values[:, j, i] = np.array([grid_energy, energy_cost, investment_cost, lcoe_val])

        df_lcoe = pd.DataFrame(data=values[3, :, :], index=inv_rate_arr, columns=years_arr)

        return df_lcoe

    def optimize(self):
        self.run()
        
    def run(self):
        """
        Function that optimizes a MicroGrid Object
        Args:

        Returns:

        """
        self.iteration = 0
        self.raw_results = list()

        self.progress_signal.emit(0)
        self.progress_text_signal.emit('Optimizing facility sizes by surrogate optimization...')

        # (1) Optimization problem
        # print(data.info)

        # (2) Experimental design
        # Use a symmetric Latin hypercube with 2d + 1 samples
        exp_des = SymmetricLatinHypercube(dim=self.dim, npts=2 * self.dim + 1)

        # (3) Surrogate model
        # Use a cubic RBF interpolant with a linear tail
        surrogate = RBFInterpolant(kernel=CubicKernel, tail=LinearTail, maxp=self.max_eval)

        # (4) Adaptive sampling
        # Use DYCORS with 100d candidate points
        adapt_samp = CandidateDYCORS(data=self, numcand=100 * self.dim)

        # Use the serial controller (uses only one thread)
        controller = SerialController(self.objfunction)

        # (5) Use the synchronous strategy without non-bound constraints
        strategy = SyncStrategyNoConstraints(
            worker_id=0, data=self, maxeval=self.max_eval, nsamples=1,
            exp_design=exp_des, response_surface=surrogate,
            sampling_method=adapt_samp)
        controller.strategy = strategy

        # Run the optimization strategy
        result = controller.run()

        # Print the final result
        print('Best value found: {0}'.format(result.value))
        print('Best solution found: {0}'.format(
            np.array_str(result.params[0], max_line_width=np.inf,
                         precision=5, suppress_small=True)))
        self.solution = result.params[0]

        # Extract function values from the controller
        self.optimization_values = np.array([o.value for o in controller.fevals])

        # turn the results into a DataFrame
        data = np.array(self.raw_results)
        self.x_fx = pd.DataFrame(data=data[:, 1:], index=data[:, 0], columns=['Solar', 'Wind', 'Battery', 'LCOE', 'Grid energy', 'Investment'])
        self.x_fx.sort_index(inplace=True)

        self.progress_text_signal.emit('Done!')
        self.done_signal.emit()

    def plot(self):
        """
        Plot the dispatch values
        Returns:

        """
        # plot results
        plot_cols = 3
        plot_rows = 2

        steps_number = len(self.demand_system.normalized_power)

        plt.figure(figsize=(16, 10))
        plt.subplot(plot_rows, plot_cols, 1)
        plt.plot(self.demand_system.power(), label="Demand Power")
        plt.plot(self.solar_farm.power(), label="Photovoltaic Power")
        plt.plot(self.wind_farm.power(), label="Wind Power")
        plt.ylabel('kW')
        plt.legend()

        plt.subplot(plot_rows, plot_cols, 2)
        plt.plot(self.aggregated_demand_profile, label="Aggregated power profile")
        plt.plot(np.zeros(steps_number), 'k')
        plt.ylabel('kW')
        plt.legend()

        plt.subplot(plot_rows, plot_cols, 3)
        plt.plot(- self.aggregated_demand_profile, label="Power demanded to the battery")
        plt.plot(np.zeros(steps_number), 'k')
        plt.ylabel('kW')
        plt.legend()

        plt.subplot(plot_rows, plot_cols, 4)
        plt.plot(self.grid_power, label="Power demanded to the grid")
        plt.plot(np.zeros(steps_number), 'k')
        plt.ylabel('kW')
        plt.legend()

        plt.subplot(plot_rows, plot_cols, 5)
        plt.plot(self.battery_output_power, label="Battery power")
        plt.plot(np.zeros(steps_number), 'k')
        plt.ylabel('kW')
        plt.legend()

        plt.subplot(plot_rows, plot_cols, 6)
        plt.plot(self.battery_state_of_charge, label="Battery SoC")
        plt.plot(np.zeros(steps_number), 'k')
        plt.ylabel('Per unit')
        plt.legend()

    def plot_optimization(self, ax=None):
        """
        Plot the optimization convergence
        Returns:

        """
        if self.optimization_values is not None:
            max_eval = len(self.optimization_values)

            if ax is None:
                f, ax = plt.subplots()
            # Points
            ax.plot(np.arange(0, max_eval), self.optimization_values, 'bo')
            # Best value found
            ax.plot(np.arange(0, max_eval), np.minimum.accumulate(self.optimization_values), 'r-', linewidth=3.0)
            ax.set_xlabel('Evaluations')
            ax.set_ylabel('Function Value')
            ax.set_title('Optimization convergence')

    def export(self, file_name):
        """
        Export definition and results to excel
        :param file_name:
        :return:
        """

        writer = pd.ExcelWriter(file_name)

        # Solar irradiation
        pd.DataFrame(data=self.solar_farm.irradiation,
                     index=self.time,
                     columns=['irradiation (MW/m2)']).to_excel(writer, 'irradiation')

        # wind speed
        pd.DataFrame(data=self.wind_farm.wind_speed,
                     index=self.time,
                     columns=['VEL(m/s):60']).to_excel(writer, 'wind')

        # AG curve
        self.wind_farm.wt_curve_df.to_excel(writer, 'AG_CAT')

        # demand
        pd.DataFrame(data=self.demand_system.normalized_power * -1,
                     index=self.time,
                     columns=['normalized_demand']).to_excel(writer, 'demand')

        # prices
        pd.DataFrame(data=np.c_[self.spot_price, self.band_price],
                     index=self.time,
                     columns=['Secondary_reg_price', 'Spot_price']).to_excel(writer, 'prices')

        # Results
        self.x_fx.to_excel(writer, 'results')

        writer.save()


class MicroGridBrute(MicroGrid):

    def __init__(self, solar_farm: SolarFarm, wind_farm: WindFarm, demand_system: Demand, battery_system: BatterySystem,
                 time_arr, LCOE_years=20, investment_rate=0.03, spot_price=None, band_price=None, maxeval=1000,
                 obj_fun_type: ObjectiveFunctionType=ObjectiveFunctionType.GridUsage):
        """

        :param solar_farm:
        :param wind_farm:
        :param demand_system:
        :param battery_system:
        :param time_arr:
        :param LCOE_years:
        :param investment_rate:
        :param spot_price:
        :param band_price:
        :param maxeval:
        :param obj_fun_type:
        """

        MicroGrid.__init__(self, solar_farm, wind_farm, demand_system, battery_system,
                           time_arr, LCOE_years, investment_rate, spot_price, band_price,
                           maxeval, obj_fun_type)

    def run(self):
        """

        :return:
        """

        self.progress_signal.emit(0)
        self.progress_text_signal.emit('Optimizing facility sizes by brute force...')

        step = self.max_eval
        solar_range = np.arange(self.solar_farm.min_power, self.solar_farm.max_power + step, step)
        wind_range = np.arange(self.wind_farm.min_power, self.wind_farm.max_power + step, step)
        storage_range = np.arange(self.battery_system.min_energy, self.battery_system.max_energy + step, step)
        self.max_eval = len(solar_range) * len(wind_range) * len(storage_range)
        self.iteration = 0
        self.raw_results = list()

        # Run all the combinations
        for solar_x, wind_x, storage_x in product(solar_range, wind_range, storage_range):
            x = np.array([solar_x, wind_x, storage_x])
            res = self.objfunction(x)
            # results.append(np.r_[x, res])

        data = np.array(self.raw_results)
        self.x_fx = pd.DataFrame(data=data[:, 1:], index=data[:, 0], columns=['Solar', 'Wind', 'Battery', 'LCOE', 'Grid energy', 'Investment'])

        # Extract function values from the controller
        self.optimization_values = data[:, 0].copy()

        # Sort to pick the best
        self.x_fx.sort_index(inplace=True)

        # the best solution, after sorting, is the first one
        self.solution = self.x_fx.values[0, :]

        self.progress_text_signal.emit('Done!')
        self.done_signal.emit()

if __name__ == '__main__':

    # Create devices
    fname = 'data.xls'

    prices = pd.read_excel(fname, sheetname='prices')[['Secondary_reg_price', 'Spot_price']].values

    # load the solar irradiation in W/M2 and convert it to kW
    solar_radiation_profile = pd.read_excel(fname, sheetname='irradiation')['irradiation (MW/m2)'].values

    #  create the solar farm object
    solar_farm = SolarFarm(solar_radiation_profile)

    # Load the wind speed in m/s
    wind_speed_profile = pd.read_excel(fname, sheetname='wind')['VEL(m/s):60'].values

    # load the wind turbine power curve and normalize it
    ag_curve_df = pd.read_excel(fname, sheetname='AG_CAT')['P (kW)']

    # create the wind farm object
    wind_farm = WindFarm(wind_speed_profile, ag_curve_df)

    # load the demand values and set it negative for the sign convention
    demand_profile = pd.read_excel(fname, sheetname='demand')['normalized_demand'].values

    # Create the demand facility
    desalination_plant = Demand(demand_profile, nominal_power=1000)

    # Create a Battery system
    battery = BatterySystem()

    nt = len(wind_speed_profile)
    time = [datetime(2016, 1, 1) + timedelta(hours=h) for h in range(nt)]

    # Create a MicroGrid with the given devices
    # Divide the prices by thousand because they represent €/MWh and we need €/kWh
    micro_grid = MicroGrid(solar_farm=solar_farm,
                           wind_farm=wind_farm,
                           battery_system=battery,
                           demand_system=desalination_plant,
                           time_arr=time,
                           LCOE_years=20,
                           spot_price=prices[:, 0] / 1000,
                           band_price=prices[:, 1] / 1000,
                           maxeval=100)
    micro_grid.optimize()
    res = micro_grid(micro_grid.solution, verbose=True)
    micro_grid.x_fx.to_excel('results.xlsx')
    # print(micro_grid.x_fx)

    micro_grid.plot()
    micro_grid.plot_optimization()

    battery.plot()



    plt.show()

    # print(res)


