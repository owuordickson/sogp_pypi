# -------- CONFIGURATION -------------

# GRAANK Hyperparameters
MIN_SUPPORT = 0.5

# T-GRAANK Hyperparameters
TARGET_COL = 1
MIN_REPRESENTATIVITY = 0.5
USE_CLUSTERING = 0

# Global Swarm Hyperparameters
MAX_ITERATIONS = 1
N_VAR = 1  # DO NOT CHANGE

# ACO-GRAANK Hyperparameters
EVAPORATION_FACTOR = 0.5

# GA-GRAANK Hyperparameters
N_POPULATION = 5
PC = 0.5
GAMMA = 1  # Cross-over
MU = 0.9  # Mutation
SIGMA = 0.9  # Mutation

# PSO-GRAANK Hyperparameters
VELOCITY = 0.9  # higher values helps to move to next number in search space
PERSONAL_COEFF = 0.01
GLOBAL_COEFF = 0.9
TARGET = 1
TARGET_ERROR = 1e-6
N_PARTICLES = 5

# PLS-GRAANK Hyperparameters
STEP_SIZE = 0.5

# CluGRAD Hyperparameters
ERASURE_PROBABILITY = 0.5  # determines the number of pairs to be ignored
SCORE_VECTOR_ITERATIONS = 10  # maximum iteration for score vector estimation
