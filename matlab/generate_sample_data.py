import pandas as pd
import numpy as np
import os

NUM_NODES = 100
NUM_ROUNDS = 50

CLONE_PERCENT = 0.10
MALICIOUS_PERCENT = 0.05

rows = []

for r in range(1, NUM_ROUNDS + 1):

    for node in range(1, NUM_NODES + 1):

        label = 0

        rand = np.random.rand()

        if rand < CLONE_PERCENT:
            label = 1
        elif rand < CLONE_PERCENT + MALICIOUS_PERCENT:
            label = 2

        # --------------------------------
        # Packet rate (overlapping ranges)
        # --------------------------------

        if label == 0:
            packet_rate = np.random.normal(30, 8)

        elif label == 1:
            packet_rate = np.random.normal(35, 8)

        else:
            packet_rate = np.random.normal(40, 10)

        packet_rate = max(5, packet_rate)

        # --------------------------------
        # Energy model
        # --------------------------------

        energy_remaining = max(
            0.1,
            1 - r * 0.01 + np.random.normal(0, 0.05)
        )

        energy_consumed = np.random.uniform(150, 500)

        # --------------------------------
        # Network topology
        # --------------------------------

        x = np.random.uniform(0, 100)
        y = np.random.uniform(0, 100)

        dist = np.random.uniform(5, 60)

        is_ch = np.random.choice([0,1], p=[0.9,0.1])

        rows.append([
            node,
            r,
            packet_rate,
            energy_remaining,
            energy_consumed,
            dist,
            is_ch,
            x,
            y,
            label
        ])

        # --------------------------------
        # REAL CLONE INJECTION
        # same node id appears twice
        # --------------------------------

        if label == 1 and np.random.rand() < 0.6:

            clone_x = np.random.uniform(0, 100)
            clone_y = np.random.uniform(0, 100)

            clone_packet = packet_rate + np.random.normal(0,5)
            clone_energy = max(0.1, energy_remaining - np.random.normal(0,0.05))

            rows.append([
                node,
                r,
                clone_packet,
                clone_energy,
                energy_consumed,
                dist,
                is_ch,
                clone_x,
                clone_y,
                label
            ])


df = pd.DataFrame(rows, columns=[
    "node_id",
    "round",
    "packet_rate",
    "energy_remaining",
    "energy_consumed_uJ",
    "dist_to_ch_bs",
    "is_cluster_head",
    "x_pos",
    "y_pos",
    "label"
])


# save dataset
base = os.path.dirname(os.path.dirname(__file__))
data_dir = os.path.join(base, "data")

os.makedirs(data_dir, exist_ok=True)

file_path = os.path.join(data_dir, "wsn_data.csv")

df.to_csv(file_path, index=False)

print("Improved WSN dataset generated at:", file_path)
