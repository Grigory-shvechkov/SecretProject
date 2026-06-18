"""
Fish Position Simulator
------------------------
Generates random fake fish positions (in inches, within the real tank's
bounds) and POSTs them to the feed-fish API on a fixed interval.

This is meant for testing the end-to-end pipeline (Pi -> API -> Roblox)
without needing real sensor hardware wired up yet.

Run on the Raspberry Pi with:
    python3 fish_position_simulator.py
"""

import random
import time

import requests

# ===== CONFIG =====

API_URL = "https://feed-fish.onrender.com/newPos"

# Real-world tank dimensions in inches (must match what's used on the Roblox side)
REAL_TANK_SIZE = {"x": 24, "y": 12, "z": 12}
REAL_TANK_MIN = {"x": 0, "y": 0, "z": 0}

SEND_INTERVAL_SECONDS = 2.0  # how often to send a new position
REQUEST_TIMEOUT_SECONDS = 5.0


def random_position():
    """Generate a random (x, y, z) position within the real tank's bounds."""
    return {
        "x": round(REAL_TANK_MIN["x"] + random.random() * REAL_TANK_SIZE["x"], 2),
        "y": round(REAL_TANK_MIN["y"] + random.random() * REAL_TANK_SIZE["y"], 2),
        "z": round(REAL_TANK_MIN["z"] + random.random() * REAL_TANK_SIZE["z"], 2),
    }


def send_position(position):
    """POST a position dict like {'x': .., 'y': .., 'z': ..} to the API."""
    try:
        response = requests.post(
            API_URL,
            json=position,
            headers={"Content-Type": "application/json"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        print(f"Sent {position} -> {response.status_code} {response.json()}")
    except requests.exceptions.RequestException as error:
        print(f"Request failed for {position}: {error}")


def main():
    print("Starting fish position simulator. Press Ctrl+C to stop.")
    try:
        while True:
            position = random_position()
            send_position(position)
            time.sleep(SEND_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nStopped by user.")


if __name__ == "__main__":
    main()