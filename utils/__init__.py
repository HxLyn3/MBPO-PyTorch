from .replay_buffer import ReplayBuffer
from .replay_buffer import ReplayBufferMBPC

BUFFER = {
    "vanilla": ReplayBuffer,
    "mbpc": ReplayBufferMBPC
}