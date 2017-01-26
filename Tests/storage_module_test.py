from datetime import datetime, timedelta

import pandas as pd
from matplotlib import pyplot as plt

from Engine.microgrid_dispatch import BatterySystem

battery = BatterySystem(charge_efficiency=1.0, discharge_efficiency=1.0, max_soc=1.0, min_soc=0.3)
battery.nominal_energy = 200

vals = [100, -25, -10, 30, -200, 50, 10]

start = datetime(2016, 1, 1)
nt = len(vals)
idx = [start + timedelta(hours=h) for h in range(nt)]

cols = ['P']
data = pd.DataFrame(data=vals, index=idx, columns=cols)

battery.simulate_array(P=vals, soc_0=0.5, time=data.index)

print(battery.results)

battery.plot()
plt.show()
