import pandas as pd
import math

from typing import Sequence, Collection


def get_costs(heads: Sequence[int], zones: Sequence[int]) -> list[int]:
    zone2cost: dict[int, int] = {z: 0 for z in zones}
    for z, h in zip(zones, heads):
        if h == 0 or h > len(zone2cost):
            continue
        if (h-1) < z:
            if z in zone2cost:
                zone2cost[z] += z-(h-1)
        else:
            if h-1 in zone2cost:
                zone2cost[h-1] += (h-1)-z

    return list(zone2cost.values())


def get_costs_referents(
        heads: Sequence[int], zones: Sequence[int],
        pos_tags: Sequence[str],
        referents: Collection[str] = ("NN", "VB")) -> list[int]:
    zone2cost: dict[int, int] = {z: 0 for z in zones}
    is_referent: list[bool] = [p in referents for p in pos_tags]

    for z, h, p in zip(zones, heads, pos_tags):
        if p in referents:
            continue
        if h == 0 or h > len(zone2cost):
            continue
        if (h-1) < z:
            if z in zone2cost:
                zone2cost[z] += sum(is_referent[h:z])
        else:
            if h-1 in zone2cost:
                zone2cost[h-1] += sum(is_referent[z+1:h-1])

    return list(zone2cost.values())


infile = "./data/EWT_RT_train_preprocessed_exp1_4_0.csv"
outfile = "./data/EWT_RT_train_preprocessed_exp1_4_0_headcost.csv"

df = pd.read_csv(infile)

head_costs: list[int] = []
current_heads: list[int] = []
current_zones: list[int] = []
current_pos_tags: list[str] = []
current_item: int = 0

assert "head" in df.columns

for i in range(len(df)):
    row = df.loc[i]

    if row["item"] > current_item:
        costs = get_costs_referents(
            current_heads, current_zones, current_pos_tags)
        assert len(costs) == len(current_heads), (
            len(costs), len(current_heads), len(current_zones))
        head_costs.extend(costs)
        if math.isnan(row["head"].item()):
            current_heads = [0]
        else:
            current_heads = [int(row["head"].item())]
        current_zones = [int(row["zone"].item())]
        current_pos_tags = [str(row["pos"])]
    else:
        if math.isnan(row["head"].item()):
            current_heads.append(0)
        else:
            current_heads.append(int(row["head"].item()))
        current_zones.append(int(row["zone"].item()))
        current_pos_tags.append(str(row["pos"]))

    current_item = row["item"].item()
costs = get_costs_referents(
    current_heads, current_zones, current_pos_tags)
assert len(costs) == len(current_heads), (
    len(costs), len(current_heads), len(current_zones))
head_costs.extend(costs)

df["costs"] = head_costs
df.to_csv(outfile)
