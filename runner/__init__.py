# runner for sac
from .sac_trainer import SACTrainer
from .sac_tester import SACTester

# runner for mbpo
from .mbpo_trainer import MBPOTrainer
from .mbpo_tester import MBPOTester

RUNNER = {
    "sac-train": SACTrainer,
    "sac-test": SACTester,
    "mbpo-train": MBPOTrainer,
    "mbpo-test": MBPOTester
}