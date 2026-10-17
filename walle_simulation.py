import json
import math
import random
from typing import Any, Dict, List, Optional, Tuple


CONFIG_FILE = "config.json"
TRACE_FILE = "action_trace.json"
Coordinate = Tuple[int, int]


def create_sample_config() -> Dict[str, Any]:
    """Create the sample configuration unless config.json already exists."""
    config = {
        "map_size": [10, 10],
        "walle_start": [0, 0],
        "eve_location": [8, 8],
        "clue_locations": [[2, 3], [5, 6]],
        "recharge_station": [0, 0],
        "resource_budget": {
            "battery": 100,
            "time": 50,
            "move_cost": 2,
            "scan_cost": 5,
            "recharge_rate": 50,
        },
        "initial_signal": {"direction": "NE", "strength": 0.2, "timestamp": 0},
    }

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8"):
            return config
    except FileNotFoundError:
        with open(CONFIG_FILE, "w", encoding="utf-8") as config_file:
            json.dump(config, config_file, indent=2)
        return config


class Environment:
    """Grid environment that applies actions and reports observations."""

    def __init__(self, config_path: str = CONFIG_FILE) -> None:
        try:
            with open(config_path, "r", encoding="utf-8") as config_file:
                self.config = json.load(config_file)
        except FileNotFoundError:
            create_sample_config()
            with open(config_path, "r", encoding="utf-8") as config_file:
                self.config = json.load(config_file)

        self.width, self.height = self.config["map_size"]
        self.position: Coordinate = tuple(self.config["walle_start"])  # type: ignore
        self.eve_location: Coordinate = tuple(self.config["eve_location"])  # type: ignore
        self.clue_locations = {
            tuple(location) for location in self.config["clue_locations"]
        }
        self.recharge_station: Coordinate = tuple(  # type: ignore
            self.config["recharge_station"]
        )
        self.battery = float(self.config["resource_budget"]["battery"])
        self.time_remaining = float(self.config["resource_budget"]["time"])
        self.max_battery = self.battery
        self.eve_found = False
        self.collected_clues: List[Coordinate] = []

    def _observation(
        self, status: str, message: str, strength: Optional[float] = None
    ) -> Dict[str, Any]:
        observation: Dict[str, Any] = {
            "status": status,
            "message": message,
            "position": list(self.position),
            "battery": round(self.battery, 2),
            "time_remaining": round(self.time_remaining, 2),
        }
        if strength is not None:
            observation["signal_strength"] = strength
        return observation

    def step(self, action: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Apply move, scan, or recharge and return a structured observation."""
        params = params or {}
        budget = self.config["resource_budget"]

        if action == "move":
            dx, dy = int(params.get("dx", 0)), int(params.get("dy", 0))
            if abs(dx) + abs(dy) != 1:
                return self._observation("INVALID", "A move must be one grid step.")
            if self.battery < budget["move_cost"] or self.time_remaining < 1:
                return self._observation("RESOURCE_EXHAUSTED", "Move cannot be afforded.")
            new_position = (self.position[0] + dx, self.position[1] + dy)
            if not (0 <= new_position[0] < self.width and 0 <= new_position[1] < self.height):
                return self._observation("BLOCKED", "Move would leave the map.")
            self.position = new_position
            self.battery -= budget["move_cost"]
            self.time_remaining -= 1
            return self._observation("MOVED", "WALL-E moved one grid cell.")

        if action == "scan":
            if self.battery < budget["scan_cost"] or self.time_remaining < 1:
                return self._observation("RESOURCE_EXHAUSTED", "Scan cannot be afforded.")
            self.battery -= budget["scan_cost"]
            self.time_remaining -= 1
            if self.position == self.eve_location:
                self.eve_found = True
                return self._observation("EVE_FOUND", "EVE's signal is confirmed.", 100.0)
            if self.position in self.clue_locations:
                if self.position not in self.collected_clues:
                    self.collected_clues.append(self.position)
                dx = self.eve_location[0] - self.position[0]
                dy = self.eve_location[1] - self.position[1]
                hint = [0 if dx == 0 else (1 if dx > 0 else -1),
                        0 if dy == 0 else (1 if dy > 0 else -1)]
                distance = math.dist(self.position, self.eve_location)
                strength = round(100 / (1 + distance), 2)
                return {
                    **self._observation("CLUE_FOUND", "A clue points toward EVE.", strength),
                    "direction_hint": hint,
                }
            distance = math.dist(self.position, self.eve_location)
            strength = round(100 / (1 + distance), 2)
            return self._observation("SIGNAL_DETECTED", "A faint signal was detected.", strength)

        if action == "recharge":
            if self.position != self.recharge_station:
                return self._observation("INVALID", "WALL-E is not at the recharge station.")
            if self.time_remaining < 1:
                return self._observation("RESOURCE_EXHAUSTED", "There is no time to recharge.")
            self.battery = min(self.max_battery, self.battery + budget["recharge_rate"])
            self.time_remaining -= 1
            return self._observation("RECHARGED", "Battery recharged.")

        return self._observation("INVALID", "Unknown action.")


class AutonomousAgent:
    """Belief-driven controller for WALL-E."""

    def __init__(self, environment: Environment) -> None:
        self.environment = environment
        self.position = environment.position
        self.beliefs: Dict[Coordinate, float] = {
            (x, y): 1.0 / (environment.width * environment.height)
            for x in range(environment.width)
            for y in range(environment.height)
        }
        self.visited: set = set()
        self.logs: List[Dict[str, Any]] = []
        self.step_number = 0
        self.last_reasoning = "Starting with a uniform belief map."
        self.random = random.Random(7)

    def _distance(self, first: Coordinate, second: Coordinate) -> int:
        return abs(first[0] - second[0]) + abs(first[1] - second[1])

    def _path_home_cost(self) -> float:
        return self._distance(self.position, self.environment.recharge_station) * self.environment.config[
            "resource_budget"
        ]["move_cost"]

    def _update_beliefs(self, observation: Dict[str, Any]) -> None:
        status = observation["status"]
        if status == "EVE_FOUND":
            self.beliefs[self.position] = 1.0
            return
        signal = observation.get("signal_strength")
        if signal is not None:
            for coordinate in self.beliefs:
                distance = math.dist(coordinate, self.position)
                expected = 100 / (1 + distance)
                similarity = 1 / (1 + abs(expected - signal))
                self.beliefs[coordinate] *= similarity
        if status == "CLUE_FOUND":
            hint = observation["direction_hint"]
            for coordinate in self.beliefs:
                if (coordinate[0] - self.position[0]) * hint[0] >= 0 and (
                    coordinate[1] - self.position[1]
                ) * hint[1] >= 0:
                    self.beliefs[coordinate] *= 2.5
        total = sum(self.beliefs.values())
        if total:
            self.beliefs = {coordinate: value / total for coordinate, value in self.beliefs.items()}

    def _next_target(self) -> Coordinate:
        unvisited_clues = [
            clue for clue in self.environment.clue_locations if clue not in self.visited
        ]
        if unvisited_clues:
            return min(unvisited_clues, key=lambda clue: self._distance(self.position, clue))
        candidates = [coordinate for coordinate in self.beliefs if coordinate not in self.visited]
        return max(candidates or list(self.beliefs), key=self.beliefs.get)

    def _move_toward(self, target: Coordinate) -> Tuple[int, int]:
        dx = target[0] - self.position[0]
        dy = target[1] - self.position[1]
        if dx:
            return (1 if dx > 0 else -1, 0)
        if dy:
            return (0, 1 if dy > 0 else -1)
        return (0, 0)

    def _log(self, action: str, observation: Dict[str, Any], reasoning: str) -> None:
        self.step_number += 1
        self.logs.append(
            {
                "step_number": self.step_number,
                "action_taken": action,
                "current_battery": round(self.environment.battery, 2),
                "current_time": round(self.environment.time_remaining, 2),
                "position": list(self.position),
                "observation": observation,
                "reasoning": reasoning,
            }
        )

    def _print_map(self) -> None:
        print("\nMap:")
        for y in range(self.environment.height - 1, -1, -1):
            row = []
            for x in range(self.environment.width):
                coordinate = (x, y)
                marker = "."
                if coordinate == self.environment.recharge_station:
                    marker = "R"
                if coordinate in self.environment.clue_locations:
                    marker = "C"
                if coordinate == self.position:
                    marker = "W"
                if self.environment.eve_found and coordinate == self.environment.eve_location:
                    marker = "E"
                row.append(marker)
            print(" ".join(row))

    def _act(self, action: str, params: Dict[str, Any], reasoning: str) -> Dict[str, Any]:
        observation = self.environment.step(action, params)
        self.position = self.environment.position
        if action == "scan":
            self.visited.add(self.position)
        self._update_beliefs(observation)
        self._log(action, observation, reasoning)
        if action == "move":
            self._print_map()
        return observation

    def run(self) -> None:
        self.visited.add(self.position)
        self._act("scan", {}, "Scan the starting cell for an initial signal.")
        while (
            not self.environment.eve_found
            and self.environment.time_remaining > 0
            and self.environment.battery > 0
        ):
            if self.environment.position == self.environment.recharge_station and (
                self.environment.battery <= self.environment.max_battery * 0.2
            ):
                self._act("recharge", {}, "Battery is at or below 20%; recharge before continuing.")
                continue

            if (
                self.environment.battery <= self.environment.max_battery * 0.2
                or self.environment.battery < self._path_home_cost() + 5
            ):
                if self.position == self.environment.recharge_station:
                    self._act("recharge", {}, "Recharge is required before the next search leg.")
                    continue
                target = self.environment.recharge_station
                reasoning = "Battery is low or insufficient for a safe return; head to recharge."
            else:
                target = self._next_target()
                reasoning = (
                    "Explore the nearest unvisited clue."
                    if target in self.environment.clue_locations
                    else "Move toward the highest-belief unvisited coordinate."
                )

            if self.position == target:
                observation = self._act("scan", {}, "Scan the selected target coordinate.")
                if observation["status"] == "EVE_FOUND":
                    break
                continue

            move = self._move_toward(target)
            observation = self._act(
                "move", {"dx": move[0], "dy": move[1]}, reasoning
            )
            if observation["status"] != "MOVED":
                break

        if self.position == self.environment.recharge_station and not self.environment.eve_found:
            self._act("scan", {}, "Scan the recharge station before ending the mission.")
        with open(TRACE_FILE, "w", encoding="utf-8") as trace_file:
            json.dump(self.logs, trace_file, indent=2)

    def report(self) -> None:
        if self.environment.eve_found:
            status = "Found"
            recommendation = "Transmit the confirmed coordinates and establish contact with EVE."
        elif self.environment.time_remaining <= 0:
            status = "Time Up"
            recommendation = "Resume from the strongest-belief region with a larger time budget."
        else:
            status = "Out of Battery"
            recommendation = "Recharge WALL-E and continue scanning the strongest-belief region."
        confidence = max(self.beliefs.values()) * 100
        print("\n" + "=" * 58)
        print("FINAL MISSION REPORT")
        print("=" * 58)
        print("Total Steps Taken:", self.step_number)
        print("Evidence Found (clues collected):", len(self.environment.collected_clues))
        print(
            "Inferred EVE Location & Final Status:",
            list(self.environment.eve_location),
            "|",
            status,
        )
        print("Confidence Level: {:.2f}%".format(confidence))
        print("Recommended Next Step:", recommendation)
        print("Action trace saved to:", TRACE_FILE)


def main() -> None:
    create_sample_config()
    environment = Environment()
    agent = AutonomousAgent(environment)
    agent.run()
    agent.report()


if __name__ == "__main__":
    main()
