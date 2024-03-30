# ------------------------------------------------------------------------
# Contains definition of LABELS, TRAFFIC LIGHT STATUS and more
#
# tl2la
# Copyright (c) 2023 Andreas Weber. All Rights Reserved.
# ------------------------------------------------------------------------

import enum

from interval import interval, inf

# Dataset sample rate of single snapshots
SAMPLE_FREQUENCY = 100e-3 # 10 Hz = 100ms

# Dataset size of preprocessed datasets: agent mapping, agent motion
DATASET_SIZE = 1000

# Traffic light states
class TL_STATUS(enum.Enum):
    RED = 0
    GREEN = 1
    YELLOW = 2
    YELLOW_RED = 3
    UNKNOWN = 4


# Possible manouvers of a lane depending on lane-connectivity 
class SCENE_TL_STATUS(enum.Enum):
    CONST_RED = 0
    CONST_GREEN = 1
    RED_GREEN = 2
    GREEN_RED = 3
    UNDEFINED = 4


# Possible manouvers of a lane depending on lane-connectivity 
class TURN(enum.Enum):
    LEFT = 0
    RIGHT = 1
    NORMAL = 2


# Possible state changes
STATUS_CHANGE = {
    "RED_TO_GREEN": 1,
    "GREEN_TO_RED": -1
}



############################################################
#     Parameters for pattern-based contribution method     #
############################################################
class DATASET_TYPE(enum.Enum):
    # contains all lane segments that belong to a common lane sequence of an intersection branch
    EXTENDED =  0 
    # contains only the first lane segment (sequence head lane id) that belong to a lane sequence of an intersection branch
    MINIMAL =  1
    # if you plan to use a model-based approach then dataset can be split in train and test junctions
    TRAIN =  2 
    TEST =  3
    UNDEFINED = 4
    

############################################################
#     Parameters for pattern-based contribution method     #
############################################################

class AGENT_VELOCITY_STATES(enum.Enum):
    # Agent velocity states
    
    # Velocity for standstil in m/s
    V_STANDSTIL = interval[0, 1]
    # Velocity for moving in m/s
    V_MOVING = interval[1, inf]


class AGENT_ACCELERATION_STATES(enum.Enum):
    # Agent acceleration states
    ACCELERATING = interval[1, inf]
    NO_ACCELERATION = interval[-1, 1]
    DECELERATION = interval[-inf, -1]


class AGENT_PATTERN(enum.Enum):
    # motion patterns are classified into 5 different classes
    STATONARY = 0
    MOVING = 1
    DECELERATION = 2
    ACCELERATION = 3
    UNDEFINED = 4


class THRESHOLDS(enum.Enum):
    # Maximum distance from stop line to a traffic agents current position to consider motion patterns
    T_DIST = 80 # meter
    
    # Lane Squence can be devided into a slow down zone and a stop zone (in meter)
    T_STOP_ZONE = 8 # meter
    T_SD_ZONE = 20 # meter
    
    # response time of a vehicle reacting to a green traffig singal change
    T_GREEN_REACTION = 3 # seconds
    
    # response time of a vehicle reacting to a red traffig singal change (some peoply might cross the intersection at red turning light)
    T_RED_REACTION = 1 # seconds
    
    # Traffic lights to consider from ego vehicles POV 
    T_TL_DIST = 40 # meter
    
    
############################################################
#             Parameters for rejection method              #
############################################################

class THRESHOLDS_CONSERVATIVE(enum.Enum):
    # Distance to the stop line to detect drive through
    STOP_LINE_DIST = 1 # meter
    # Velocity combined with the stop line dist to detect drive through
    V_DRIVE_THROUGH = 15 # kph
    # Velocity combined with the stop line dist to detect drive through on red-turn
    V_DRIVE_THROUGH_RIGHT_TURN = 25 # kph