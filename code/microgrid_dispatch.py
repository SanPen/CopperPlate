import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from pySOT import *
from poap.controller import ThreadController, BasicWorkerThread, SerialController
plt.style.use('ggplot')


class Demand:

    def __init__(self, profile, nominal_power):
        self.index = None

        self.nominal_power = nominal_power

        self.normalized_power = profile * -1  # the convention is that loads are negative

    def power(self):
        return self.nominal_power * self.normalized_power


class SolarFarm:

    def __init__(self, profile, solar_power_max=10000):
        self.index = None

        self.nominal_power = None

        self.normalized_power = profile / 1000.0

        self.max_power = solar_power_max

    def power(self):
        return self.nominal_power * self.normalized_power


class WindFarm:

    def __init__(self, profile, wt_curve_df, wind_power_max=10000):

        self.index = None

        self.nominal_power = None

        # load the wind turbine power curve and normalize it
        ag_curve = interp1d(wt_curve_df.index, wt_curve_df.values / wt_curve_df.values.max())

        self.normalized_power = ag_curve(profile)

        self.max_power = wind_power_max

    def power(self):
        return self.nominal_power * self.normalized_power


class BatterySystem:

    def __init__(self, charge_efficiency=0.9, discharge_efficiency=0.9, max_soc=0.99, min_soc=0.3, batery_energy_max=100000):
        """

        Args:
            charge_efficiency:
            discharge_efficiency:
            max_soc:
            min_soc:
        """

        self.index = None

        self.nominal_energy = None

        self.charge_efficiency = charge_efficiency

        self.discharge_efficiency = discharge_efficiency

        self.max_soc = max_soc

        self.min_soc = min_soc

        self.max_energy = batery_energy_max

        self.results = None


    def simulate_array(self, P, soc_0, time):
        """
        The storage signs are the following

        supply power: positive
        recharge power: negative

        this means that a negative power will ask the battery to charge and
        a positive power will ask the battery to discharge

        to match these signs to the give profiles, we should invert the
        profiles sign
        Args:
            P: Power array: Negative charge, positive discharge
            soc_0: State of charge at the beginning [0~1]
            time: Array of datatime values
        Returns:
            energy: Energy effectively processed by the battery
            power: Power effectively processed by the battery
            grid_power: Power dumped array
            soc: Battery state of charge array
        """

        if self.nominal_energy is None:
            raise Exception('You need to set the battery nominal power!')

        P = np.array(P)
        nt = len(P)
        energy = np.zeros(nt + 1)
        power = np.zeros(nt + 1)
        soc = np.zeros(nt + 1)
        grid_power = np.zeros(nt + 1)
        energy[0] = self.nominal_energy * soc_0
        soc[0] = soc_0
        for t in range(nt-1):

            if np.isnan(P[t]):
                print('NaN found!!!!!!')

            if P[t] >= 0:
                eff = self.discharge_efficiency
            else:
                eff = self.charge_efficiency

            # the time comes in nanoseconds, we need the time step in hours
            dt = (time[t + 1] - time[t]).seconds / 3600

            proposed_energy = energy[t] - P[t] * dt * eff

            if proposed_energy > self.nominal_energy * self.max_soc:

                # print('truncated: too high')

                energy[t + 1] = self.nominal_energy * self.max_soc
                power[t + 1] = (energy[t]-energy[t + 1]) / (dt * eff)
                grid_power[t + 1] = - power[t + 1] + P[t]

            elif proposed_energy < self.nominal_energy * self.min_soc:

                # print('truncated: too low')

                energy[t + 1] = self.nominal_energy * self.min_soc
                power[t + 1] = (energy[t]-energy[t + 1]) / (dt * eff)
                grid_power[t + 1] = - power[t + 1] + P[t]

            else:

                # print('ok: within boundaries')

                energy[t + 1] = proposed_energy
                power[t + 1] = P[t]
                grid_power[t + 1] = 0

            soc[t + 1] = energy[t + 1] / self.nominal_energy

        d = np.c_[np.r_[0, P[:-1]], power[:-1], grid_power[:-1], energy[:-1], soc[:-1] * 100]
        cols = ['P request', 'P', 'grid', 'E', 'SoC']
        self.results = pd.DataFrame(data=d, columns=cols)

        return energy[:-1], power[:-1], grid_power[:-1], soc[:-1]


class MicroGrid:

    dim = 3  # 3 variables to optimize

    def __init__(self, solar_farm:SolarFarm, wind_farm:WindFarm, demand_system:Demand, battery_system:BatterySystem,
                 start=datetime(2016, 1, 1)):

        # optimizator variables
        self.xlow = np.zeros(self.dim)  # lower bounds
        self.xup = np.array([solar_farm.max_power, wind_farm.max_power, battery_system.max_energy])
        self.info = "Our own " + str(self.dim) + "-dimensional function"  # info
        self.integer = np.array([0])  # integer variables
        self.continuous = np.arange(1, self.dim)  # continuous variables

        # assign the device list
        self.solar_farm = solar_farm

        self.wind_farm = wind_farm

        self.demand_system = demand_system

        self.battery_system = battery_system

        # create a time index matching the length
        nt = len(wind_speed_profile)
        self.time = [start + timedelta(hours=h) for h in range(nt)]

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

        self.optimization_values = None

    def __call__(self, x):
        return self.objfunction(x)

    def objfunction(self, x):
        """

        Args:
            x: optimsation vector [solar nominal power, wind nominal power, storage nominal power]

        Returns:

        """
        ###############################################################################
        # Set the devices nominal power
        ###############################################################################
        self.solar_farm.nominal_power = x[0]

        self.wind_farm.nominal_power = x[1]

        self.battery_system.nominal_energy = x[2]

        '''
        The profiles sign as given are:

            demand: negative
            generation: positive
        '''

        ###############################################################################
        # Compute the battery desired profile
        ###############################################################################

        self.aggregated_demand_profile = self.demand_system.power() + self.wind_farm.power() + self.solar_farm.power()

        ###############################################################################
        # Compute the battery real profile: processing the desired profile
        ###############################################################################

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
        grid_power, \
        self.battery_state_of_charge = self.battery_system.simulate_array(P=demanded_power, soc_0=SoC0, time=self.time)

        # the processed values are 1 value shorter since we have worked with time increments

        # calculate the grid power as the difference of the battery power
        # and the profile required for perfect auto-consumption
        self.grid_power = demanded_power - self.battery_output_power

        return sum(abs(self.grid_power))

    def plot(self):
        # plot results
        plot_cols = 3
        plot_rows = 2

        steps_number = len(self.demand_system.normalized_power)

        plt.figure(figsize=(16, 10))
        plt.subplot(plot_rows, plot_cols, 1)
        plt.plot(self.demand_system.power(), label="Demand Power")
        plt.plot(self.solar_farm.power(), label="Photovoltaic Power")
        plt.plot(self.wind_farm.power(), label="Wind Power")
        plt.legend()

        plt.subplot(plot_rows, plot_cols, 2)
        plt.plot(self.aggregated_demand_profile, label="Aggregated power profile")
        plt.plot(np.zeros(steps_number), 'k')
        plt.legend()

        plt.subplot(plot_rows, plot_cols, 3)
        plt.plot(- self.aggregated_demand_profile, label="Power demanded to the battery")
        plt.plot(np.zeros(steps_number), 'k')
        plt.legend()

        plt.subplot(plot_rows, plot_cols, 4)
        plt.plot(self.grid_power, label="Power demanded to the grid")
        plt.plot(np.zeros(steps_number), 'k')
        plt.legend()

        plt.subplot(plot_rows, plot_cols, 5)
        plt.plot(self.battery_output_power, label="Battery power")
        plt.plot(np.zeros(steps_number), 'k')
        plt.legend()

        plt.subplot(plot_rows, plot_cols, 6)
        plt.plot(self.battery_state_of_charge, label="Battery SoC")
        plt.plot(np.zeros(steps_number), 'k')
        plt.legend()

    def optimize(self, maxeval=1000):
        """
        Function that optimizes a MicroGrid Object
        Args:

        Returns:

        """
        # (1) Optimization problem
        # print(data.info)

        # (2) Experimental design
        # Use a symmetric Latin hypercube with 2d + 1 samples
        exp_des = SymmetricLatinHypercube(dim=self.dim, npts=2 * self.dim + 1)

        # (3) Surrogate model
        # Use a cubic RBF interpolant with a linear tail
        surrogate = RBFInterpolant(kernel=CubicKernel, tail=LinearTail, maxp=maxeval)

        # (4) Adaptive sampling
        # Use DYCORS with 100d candidate points
        adapt_samp = CandidateDYCORS(data=self, numcand=100 * self.dim)

        # Use the serial controller (uses only one thread)
        controller = SerialController(self.objfunction)

        # (5) Use the sychronous strategy without non-bound constraints
        strategy = SyncStrategyNoConstraints(
            worker_id=0, data=self, maxeval=maxeval, nsamples=1,
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

        # Extract function values from the controller
        self.optimization_values = np.array([o.value for o in controller.fevals])

        return result.params[0]

    def plot_optimization(self):
        """
        Plot the optimization convergence
        Returns:

        """
        if self.optimization_values is not None:
            maxeval = len(self.optimization_values)
            f, ax = plt.subplots()
            ax.plot(np.arange(0, maxeval), self.optimization_values, 'bo')  # Points
            ax.plot(np.arange(0, maxeval), np.minimum.accumulate(self.optimization_values), 'r-',
                    linewidth=3.0)  # Best value found
            plt.xlabel('Evaluations')
            plt.ylabel('Function Value')
            plt.title('Optimization convergence')

if __name__ == '__main__':

    # Create devices
    fname = 'data.xls'

    # load the solar irradiation in W/M2 and convert it to kW
    solar_radiation_profile = pd.read_excel(fname, sheetname='radiacion')['Radiación (MW/m2)'].values

    #  create the solar farm object
    solar_farm = SolarFarm(solar_radiation_profile)

    # Load the wind speed in m/s
    wind_speed_profile = pd.read_excel(fname, sheetname='viento')['VEL(m/s):60'].values

    # load the wind turbine power curve and normalize it
    ag_curve_df = pd.read_excel(fname, sheetname='AG_CAT')['P (MW)']

    # create the wind farm object
    wind_farm = WindFarm(wind_speed_profile, ag_curve_df)

    # load the demand values and set it negative for the sign convention
    demand_profile = pd.read_excel(fname, sheetname='desaladora')['Demanda normalizada'].values

    # Create the demand facility
    desalination_plant = Demand(demand_profile, nominal_power=1000)

    # Create a Battery system
    battery = BatterySystem()

    # Create a MicroGrid with the given devices
    micro_grid = MicroGrid(solar_farm=solar_farm,
                           wind_farm=wind_farm,
                           battery_system=battery,
                           demand_system=desalination_plant,
                           start=datetime(2016, 1, 1))
    res_x = micro_grid.optimize(maxeval=500)
    # res_x = [500, 600, 10000]
    res = micro_grid(res_x)
    micro_grid.plot()
    micro_grid.plot_optimization()

    plt.show()

    # print(res)

    # battery = Battery(nominal_energy=200, charge_efficiency=1.0, discharge_efficiency=1.0, max_soc=1.0, min_soc=0.3)
    #
    # vals = [100, -25, -10, 30, -200, 50, 10]
    #
    # start = datetime(2016, 1, 1)
    # nt = len(vals)
    # idx = [start + timedelta(hours=h) for h in range(nt)]
    #
    # cols = ['P']
    # data = pd.DataFrame(data=vals, index=idx, columns=cols)
    #
    # battery.simulate_array(P=vals, soc_0=0.5, time=data.index)
    #
    # print(battery.results)
    #
    # battery.results.plot()
    # plt.show()
