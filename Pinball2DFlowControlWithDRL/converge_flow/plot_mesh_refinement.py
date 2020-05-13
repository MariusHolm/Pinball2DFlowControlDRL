import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import sys
import os

if not os.path.exists('plots'):
    os.makedirs('plots')

matplotlib.rcParams.update({'font.size': 15})

scaling_factor_CD = (-2.0) / (1.0 * 1.0**2 * 1) # 1.0 / (2 * 1.5 / 3)**2 / 0.1
scaling_factor_CL = ( 2.0 ) / (1.0 * 1.0**2 * 1)

num_steps = 1200 # Define when to stop plotting

loc_data = "results/drag_lift.csv"

data = pd.read_csv(loc_data, delimiter=',')
df = pd.DataFrame(data)

single_runner = ['time', 'drag0', 'drag1', 'drag2', 'lift0', 'lift1', 'lift2']

df['sum_drag'] = df[['drag0', 'drag1', 'drag2']].sum(axis=1)

df['sum_lift'] = df[['lift0', 'lift1', 'lift2']].sum(axis=1)


list_number_Qs = [1, 2, 3]
color_idx = np.linspace(0, 1, len(list_number_Qs))

#**************************************************************************
# Calculate average drag and lift over last 100 steps
drag_force_last_half = np.array(df[["sum_drag"]][2200:2400])
drag_last_half = drag_force_last_half #* scaling_factor_CD

lift_force_last_half = np.array(df[["sum_lift"]][2200:2400])
lift_last_half = lift_force_last_half * scaling_factor_CL

drag_mean = np.mean(drag_last_half)
drag_std = np.std(drag_last_half)

lift_mean = np.mean(lift_last_half)
lift_std = np.std(lift_last_half)

print("\nmean(sum drag) =", drag_mean)
print("std sum drag =", drag_std)

print("\nmean sum lift =", lift_mean)
print("std sum lift =", lift_std)

mean_drag_per_time = df[["drag0", "drag1", "drag2"]][2200:2400].mean(axis=1)
list_mean_drags = np.array(mean_drag_per_time)
final_mean_last_100 = np.mean(list_mean_drags)

print("\n", final_mean_last_100)

#**************************************************************************
# PLOT DRAG RESULTS MESH REFINEMENT

for crrt_color_id, crrt_Qs in zip(color_idx, list_number_Qs):
    plt.figure()

    # Choose color
    crrt_color = plt.cm.winter(crrt_color_id) #coolwarm(crrt_color_id)

    # choose linestyle and line width
    linewidth = 1.0
    linestyle = "-"

    plt.plot(df['time'], df['drag{}'.format(crrt_Qs-1)] * scaling_factor_CD, color=crrt_color, linewidth=linewidth, linestyle=linestyle, label='Cylinder {}'.format(crrt_Qs-1))

    plt.xlim([0, num_steps])

    plt.legend(loc=7, prop={'size': 12})

    plt.xlabel("Non-dim time")
    plt.ylabel("$C_D$")

    plt.tight_layout()

    plt.savefig("plots/drag{}.pdf".format(crrt_Qs-1))
    plt.savefig("plots/drag{}.png".format(crrt_Qs-1))

    plt.show()


plt.figure()

crrt_color = plt.cm.coolwarm(1.5) #coolwarm(crrt_color_id)

plt.plot(df['time'], df['sum_drag'] * scaling_factor_CD, color=crrt_color, linewidth=linewidth, linestyle=linestyle, label='All cylinders')

plt.xlim([0, num_steps])

plt.legend(loc=7, prop={'size': 12})

plt.xlabel("Non-dim time")
plt.ylabel("$C_D$")

plt.tight_layout()

plt.savefig("plots/sum_drag.pdf")
plt.savefig("plots/sum_drag.png")

plt.show()


#****************************************************************************
# PLOT LIFT RESULTS SINGLE RUNNER

for crrt_color_id, crrt_Qs in zip(color_idx, list_number_Qs):
    plt.figure()

    # Choose color
    crrt_color = plt.cm.winter(crrt_color_id) #coolwarm(crrt_color_id)

    # choose linestyle and line width
    linewidth = 1.0
    linestyle = "-"

    plt.plot(df['time'], df['lift{}'.format(crrt_Qs-1)] * scaling_factor_CL, color=crrt_color, linewidth=linewidth, linestyle=linestyle, label='Cylinder {}'.format(crrt_Qs-1))

    plt.xlim([0, num_steps])

    plt.legend(loc=8, prop={'size': 12})

    plt.xlabel("Non-dim time")
    plt.ylabel("$C_L$")

    plt.tight_layout()

    plt.savefig("plots/lift{}.pdf".format(crrt_Qs-1))
    plt.savefig("plots/lift{}.png".format(crrt_Qs-1))

    plt.show()


plt.figure()

crrt_color = plt.cm.coolwarm(1.5) #coolwarm(crrt_color_id)

plt.plot(df['time'], df['sum_lift'] * scaling_factor_CL, color=crrt_color, linewidth=linewidth, linestyle=linestyle, label='All cylinders')

plt.xlim([0, num_steps])

plt.legend(loc=8, prop={'size': 12})

plt.xlabel("Non-dim time")
plt.ylabel("$C_L$")

plt.tight_layout()

plt.savefig("plots/sum_lift.pdf")
plt.savefig("plots/sum_lift.png")

plt.show()
