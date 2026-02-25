# -*- coding: utf-8 -*-

# Column name conventions
TARGET_COL = "Calculated_SOH(%)"
GROUP_COL = "battery_id"
CYCLE_COL = "Cycle_Index"
PROTOCOL_COL = "protocol"

# Data locations / sheets
FOLDER_PATHS_CONST = [
    "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_33",
    "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_34",
    "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_35",
    "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_36",
    "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_37",
    "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_38",
]

SHEET_NAMES_CONST = [
    ["Channel_1-006"],
    ["Channel_1-007"],
    ["Channel_1-008"],
    ["Channel_1-009"],
    ["Channel_1-010"],
    ["Channel_1-011"],
]

FOLDER_PATHS_VAR = [
    "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_9",
    "C:/Users/HYACINTH OSEJI/OneDrive - University of Bath/Documents/PYTHON STUFF/CS2_3",
]

SHEET_NAMES_VAR = [
    ["Channel_1-007", "Channel_1-007_1", "Channel_1-007_2"],
    ["Channel_1-008"],
]

# Global filters / settings
EOL_CUTOFF_SOH = 70
CHARGE_DURATION_MIN_H = 2

# Holdout batteries for evaluation
TEST_BATTERIES_HOLDOUT = ["CS2_38"]

# Variable-discharge spike removal thresholds
SPIKE_DIFF_THRESH = 5

# Early-life forecast settings
EARLY_LIFE_BATTERIES = ["CS2_33", "CS2_34", "CS2_35", "CS2_36", "CS2_37", "CS2_38"]
EARLY_TRAIN_FRAC = 0.30
EARLY_RANDOM_STATE = 42
EARLY_N_SPLITS = 5
EARLY_UNC_ALPHA = 0.12
EARLY_Y_LIM = (70, 102)
EARLY_ALIGN_X_AXES = True
EARLY_SPIKE_WINDOW = 9
EARLY_SPIKE_THRESH = 5.0
EARLY_SPIKE_PASSES = 2
EARLY_INTERP_METHOD = "linear"