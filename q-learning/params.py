"""params.py: Network protocol settings and parameters for Q-Learner."""

# --- Random number generation ---
PRNG_SEED = 1234567890

# --- Save and load ---
Q_NETWORK_LOAD_FILENAME = None
Q_NETWORK_SAVE_FILENAME = None

LABELING_NETWORK_FILE_NAME = 'saved-nns/best_encoder_bigdata'


# --- Q-Learner settings ---
MB_SIZE = 20
EXP_STORE_SIZE = 50000
PERCEPT_LENGTH = 25 # 64*64*3
N_ACTIONS = 3
ACTION_NAMES = ['left', 'straight', 'right']
STATE_STM = 4
GAMMA = 0.98
LEARNING_RATE = 0.0020
LEARNING_ITERATIONS_PER_STEP = 10
BURN_IN = 50

EPSILON_START = 1.00
EPSILON_END = 0.00
EPSILON_DECREASE_DURATION = 20*60*30

# Duration of random action (number of frames)
RANDOM_ACTION_DURATION = 20

Q_HIDDEN_NEURONS = 200


# --- Communication protocol settings ---
FRAME_COUNTER_POS = 0
FRAGMENT_ID_POS = 1
ACTION_POS = 2
REWARD_POS = 3
N_REWARD_BYTES = 4
IMG_DATA_POS = REWARD_POS + N_REWARD_BYTES
N_CHECKSUM_BYTES = 1
TIMEOUT_PERIOD = 0.5

FRAME_COUNTER_INC_STEP = 1
N_FRAGMENTS = 2
IMG_FRAGMENT_LENGTH = 32*64*3
IMAGE_WIDTH = 64


# --- Network settings ---
IN_IP = "0.0.0.0"
IN_PORT = 8888

REMOTE_HOST = "127.0.0.1"
REMOTE_PORT = 8889
