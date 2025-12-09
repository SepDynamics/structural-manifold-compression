import random
import os
import json

LOG_FILE = "synthetic_system.log"
GROUND_TRUTH_FILE = "ground_truth.json"

STABLE_MSG = "HEARTBEAT_OK_SYSTEM_NOMINAL_CYCLE_COMPLETE\n"


def main():
    print(f"Generating {LOG_FILE} with CANONICAL SIGNAL patterns...")

    with open(LOG_FILE, "wb") as f:
        ground_truth = []

        # 1. Stable: Pure Repetition (0 - 50KB)
        # 50,000 bytes ~ 1000 lines
        for i in range(1000):
            f.write(STABLE_MSG.encode("utf-8"))
            ground_truth.append({"index": i, "state": "STABLE"})

        # 2. Rupture: Pure Randomness (50KB - 70KB)
        # 20,000 bytes
        noise = os.urandom(20000)
        f.write(noise)
        # Map roughly to lines
        for i in range(1000, 1400):
            ground_truth.append({"index": i, "state": "RUPTURE"})

        # 3. Recovery: Pure Repetition (70KB - 100KB)
        for i in range(1400, 2000):
            f.write(STABLE_MSG.encode("utf-8"))
            ground_truth.append({"index": i, "state": "STABLE"})

    with open(GROUND_TRUTH_FILE, "w") as f:
        json.dump(ground_truth, f)

    print(f"Generated canonical test file.")


if __name__ == "__main__":
    main()
