import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from datetime import datetime, timedelta
from pySOT import *
from poap.controller import ThreadController, BasicWorkerThread, SerialController
plt.style.use('ggplot')


class SolarFarm:

    def __init__(self):
        print()


class WindFarm:

    def __init__(self):
        print()


class Battery:

    def __init__(self, nominal_energy, charge_efficiency=0.9, discharge_efficiency=0.9, max_soc=0.99, min_soc=0.3):
        """

        Args:
            nominal_energy: Battery energy in kWh
        """

        self.nominal_energy = nominal_energy

        self.charge_efficiency = charge_efficiency

        self.discharge_efficiency = discharge_efficiency

        self.max_soc = max_soc

        self.min_soc = min_soc

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
            dt = (time[t + 1] - time[t]).value / 1e9 / 3600

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

    def __init__(self, fname='datos_estudio.xls', start=datetime(2016, 1, 1),
                 desalination_noinal_power=1000, wind_power_max=10000, solar_power_max=10000,
                 batery_energy_max=100000):

        # optimizator variables
        self.xlow = np.zeros(self.dim)  # lower bounds
        self.xup = np.array([solar_power_max, wind_power_max, batery_energy_max])
        self.info = "Our own " + str(self.dim) + "-dimensional function"  # info
        self.integer = np.array([0])  # integer variables
        self.continuous = np.arange(1, self.dim)  # continuous variables

        # load the solar irradiation in W/M2 and convert it to kW
        solar_radiation_profile = pd.read_excel(fname, sheetname='radiacion')['Radiación (MW/m2)'].values

        self.normalized_solar_power = solar_radiation_profile / 1000.0

        # Load the wind speed in m/s
        wind_speed_profile = pd.read_excel(fname, sheetname='viento')['VEL(m/s):60'].values

        # load the wind turbine power curve and normalize it
        df = pd.read_excel(fname, sheetname='AG_CAT')['P (MW)']
        ag_curve = interp1d(df.index, df.values / df.values.max())

        self.normalized_wind_power = ag_curve(wind_speed_profile)

        # load the demand values and set it negative for the sign convention
        self.demand_profile = pd.read_excel(fname, sheetname='desaladora')['Demanda normalizada'].values * -1.0 * desalination_noinal_power

        # create a time index matching the length
        nt = len(wind_speed_profile)
        idx = [start + timedelta(hours=h) for h in range(nt)]

        d = np.c_[self.demand_profile, self.normalized_wind_power, self.normalized_solar_power]
        cols = ['Demand', 'Wind', 'Solar']
        data = pd.DataFrame(data=d, index=idx, columns=cols)

        self.time = data.index

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

    def __call__(self, x):
        return self.objfunction(x)

    def objfunction(self, x):
        """

        Args:
            x: optimsation vector [solar nominal power, wind nominal power, storage nominal power]

        Returns:

        """
        ###############################################################################
        # Read the data file (this is a yearly profile)
        ###############################################################################
        self.solar_power_profile = self.normalized_solar_power * x[0]

        self.wind_power_profile = self.normalized_wind_power * x[1]

        '''
        The profiles sign as given are:

            demand: negative
            generation: positive
        '''

        ###############################################################################
        # Compute the battery desired profile
        ###############################################################################

        self.aggregated_demand_profile = self.demand_profile + self.wind_power_profile + self.solar_power_profile

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

        # specify the parameters
        demanded_power = - self.aggregated_demand_profile  # reverse the power profile sign (the surplus, should go in the battery)
        SoC0 = 0.5  # initial state of charge
        Nominal_Energy_Capacity = x[2]  # kWh or the same power unit as the power profile (this is the value to optimize)
        CellInternalResistance = 0.0008  # given by the battery definition
        NumberOfCells = 300  # given by the battery definition
        AutoDischargeRate = 0.0  # given by the battery definition
        Maximum_Power_In = -np.abs(np.max(demanded_power))  # should be given by the battery definition, but here we pick the values from the desired profile
        Maximum_Power_Out = np.abs(np.min(demanded_power))  # should be given by the battery definition, but here we pick the values from the desired profile
        Maximum_SoC = 0.99  # given by the battery definition
        Minimum_SoC = 0.3  # given by the battery definition

        # create battery model
        model = Battery(Nominal_Energy_Capacity)

        # calculate the battery values: process the desired power
        # energy, power, grid_power, soc
        self.Energy, \
        self.battery_output_power, \
        grid_power, \
        self.battery_state_of_charge = model.simulate_array(P=demanded_power, soc_0=SoC0, time=self.time)

        # the processed values are 1 value shorter since we have worked with time increments

        # calculate the grid power as the difference of the battery power
        # and the profile required for perfect auto-consumption
        self.grid_power = demanded_power - self.battery_output_power

        return sum(abs(self.grid_power))

    # @staticmethod
    # def eval_eq_constraints(x):
    #     """
    #     The equality constraint is that the summation of x must be 1
    #     :param x:
    #     :return:
    #     """
    #     return np.sum(x)
    #
    # @staticmethod
    # def projection(x):
    #     """
    #     Function that brings unfeasible points into the feasible region
    #     In this case this is mandatory because we have an equality constraint and all the
    #     generated points that do not satisfy the constraint must be 'made' to satisfy it.
    #     In this case if we normalize the vector x (x_nex = x / sum(x), it is sufficient to
    #     fulfill that sum(x_nex) = 1
    #     :param x:
    #     :return:
    #     """
    #     return x / np.sum(x)

    def plot(self):
        # plot results
        plot_cols = 3
        plot_rows = 2

        steps_number = len(self.demand_profile)

        plt.figure(figsize=(16, 10))
        plt.subplot(plot_rows, plot_cols, 1)
        plt.plot(self.demand_profile, label="Consumption Power")
        plt.plot(self.solar_power_profile, label="Photovoltaic Power")
        plt.plot(self.wind_power_profile, label="Photovoltaic Power")
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


def optimize(data: MicroGrid):
    """
    Function that optimizes a MicroGrid Object
    Args:
        data: MicroGrid instance

    Returns:

    """
    # Decide how many evaluations we are allowed to use
    maxeval = 1000

    # (1) Optimization problem
    # print(data.info)

    # (2) Experimental design
    # Use a symmetric Latin hypercube with 2d + 1 samples
    exp_des = SymmetricLatinHypercube(dim=data.dim, npts=2 * data.dim + 1)

    # (3) Surrogate model
    # Use a cubic RBF interpolant with a linear tail
    surrogate = RBFInterpolant(kernel=CubicKernel, tail=LinearTail, maxp=maxeval)

    # (4) Adaptive sampling
    # Use DYCORS with 100d candidate points
    adapt_samp = CandidateDYCORS(data=data, numcand=100 * data.dim)

    # Use the serial controller (uses only one thread)
    controller = SerialController(data.objfunction)

    # (5) Use the sychronous strategy without non-bound constraints
    strategy = SyncStrategyNoConstraints(
        worker_id=0, data=data, maxeval=maxeval, nsamples=1,
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
    fvals = np.array([o.value for o in controller.fevals])

    f, ax = plt.subplots()
    ax.plot(np.arange(0, maxeval), fvals, 'bo')  # Points
    ax.plot(np.arange(0, maxeval), np.minimum.accumulate(fvals), 'r-', linewidth=4.0)  # Best value found
    plt.xlabel('Evaluations')
    plt.ylabel('Function Value')
    plt.title(data.info)
    # plt.show()

    return result.params[0]

if __name__ == '__main__':

    micro_grid = MicroGrid(fname='data.xls', start=datetime(2016, 1, 1), desalination_noinal_power=1000)

    res_x = optimize(micro_grid)

    res = micro_grid(res_x)

    micro_grid.plot()
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
