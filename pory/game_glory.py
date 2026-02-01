import gym
import random
import numpy as np
import torch
from gym import spaces
import json


# terrain,               0
# resource,              1
# improvement,           2
# climate,               3
# border,                4
# improvement_owner,     5
# unit_owner,            6
# unit_type,             7
# unit_health,           8
# improvement_progress,  9
# Has Attacked          10
# Has Moved             11

class PolytopiaEnv(gym.Env):
    def __init__(self):
        super(PolytopiaEnv, self).__init__()
        self.filename = r"C:\Users\samth\Downloads\Maps.json"

        # Adjusted action space to include passing the turn (action code 121)
        self.action_space = spaces.Discrete(122)  # Actions from 0 to 120 are tiles, 121 is pass, 122-131 are tech tree
        self.observation_space = spaces.Box(
            low=np.array([[0] * 12], dtype=np.uint8).repeat(121, axis=0),
            high=np.array([[99] * 12], dtype=np.uint8).repeat(121, axis=0),
            dtype=np.uint8
        )
        self.current_observation = np.zeros((121, 12), dtype=np.uint8)
        self.p1_observation = np.zeros((121, 12), dtype=np.uint8)
        self.p2_observation = np.zeros((121, 12), dtype=np.uint8)
        self.zeros = np.zeros((1, 12), dtype=np.uint8)

        self.p1capital = (0, 0)
        self.p2capital = (0, 0)
        self.step_phase = 0
        self.previous_step = []
        self.turn = 1  # Keep track of whose turn it is
        self.p1stars = np.zeros((1, 12), dtype=np.uint8)
        self.p2stars = np.zeros((1, 12), dtype=np.uint8)
        self.tile_cities = [0] * 121

        self.start_units = {
            1: 1,  # Xin-xi - Warrior
            2: 1,  # Imperius - Warrior
            3: 1,  # Bardur - Warrior
            4: 4,  # Oumaji - Rider
            5: 1,  # Kickoo - Warrior
            6: 2,  # Hoodrick - Archer
            7: 1,  # Luxidoor - Warrior
            8: 8,  # Vengir - Swordsman
            9: 1,  # Zebasi - Warrior
            10: 1,  # Ai-Mo - Warrior
            11: 3,  # Quetzali - Defender
            12: 1,  # Yădakk - Warrior
            13: 19,  # Aquarion - Amphibian
            14: 1,  # Elyrion - Warrior
            15: 28,  # Polaris - Mooni
            16: 34  # Cymanti - Shaman
        }

        self.unit_stats = {  # health, attack, defense, movement, range
            1: (10, 2, 2, 1, 1),  # Warrior
            2: (10, 3, 1, 1, 2),  # Archer
            3: (15, 2, 4, 1, 1),  # Defender
            4: (10, 3, 1, 2, 1),  # Rider
            # Add other units as needed
        }

        with open(self.filename, 'r') as f:
            self.maps = json.load(f)

        # Initialize explored maps for both players
        self.p1_explored = np.zeros(121, dtype=bool)
        self.p2_explored = np.zeros(121, dtype=bool)

        # Glory mode parameters
        self.max_turns_per_player = 30
        self.total_turn_limit = self.max_turns_per_player * 2  # 30 turns each = 60 total turns
        self.turn_count = 0  # counts each player's turn ending
        self.game_ended = False
        self.final_p1_points = 0
        self.final_p2_points = 0

    def reset(self):
        # Select a random map
        selected_map = random.choice(self.maps)
        selected_map = self.maps[0]
        tiles = selected_map['tiles']

        self.current_observation = np.zeros((121, 12), dtype=np.uint8)
        self.p1_observation = np.zeros((121, 12), dtype=np.uint8)
        self.p2_observation = np.zeros((121, 12), dtype=np.uint8)
        self.units = []
        self.step_phase = 0
        self.previous_step = []
        self.turn = 1
        self.phase = 0
        self.tile_cities = [0] * 121
        self.game_ended = False
        self.final_p1_points = 0
        self.final_p2_points = 0
        self.turn_count = 0

        # Initialize explored maps
        self.p1_explored = np.zeros(121, dtype=bool)
        self.p2_explored = np.zeros(121, dtype=bool)

        # Initialize capital indices
        self.p1capitalindex = None
        self.p2capitalindex = None

        self.p1stars[0][0] = 2
        self.p2stars[0][0] = 2

        terrain_mapping = {
            'Ocean': 1,
            'Water': 2,
            'Field': 3,
            'Mountain': 4,
            'Forest': 5,
        }

        resource_mapping = {
            'Game': 1,
            'Fruit': 2,
            'Fish': 3,
            'Crop': 4,
            'Metal': 5,
            'Starfish': 6,
            'Spores': 7,
            'None': 0,
        }

        improvement_mapping = {
            'City': 1,
            'Ruin': 2,
            'Lighthouse': 3,
            'None': 0,
        }

        for i, tile in enumerate(tiles):
            x, y = eval(tile['coordinates'])
            terrain = terrain_mapping.get(tile['terrain'], 0)
            resource = resource_mapping.get(tile.get('resource'), 0)
            improvement = improvement_mapping.get(tile.get('improvement'), 0)
            if i == 0 or i == 10 or i == 120 or i == 110:
                improvement = 3
            climate = tile.get('climate', 0)
            if climate is None:
                climate = 0
            climate = int(climate)
            border = 0

            improvement_owner = tile.get('improvement owner')
            unit_owner = tile.get('unit owner')
            unit_type = tile.get('unit type')
            unit_health = tile.get('unit health')
            unit_veteran = tile.get('unit veteran')
            improvement_progress = tile.get('improvement progress')

            if improvement_owner is None or improvement_owner == 'null':
                improvement_owner = 0
            else:
                improvement_owner = int(improvement_owner)

            if unit_owner is None or unit_owner == 'null':
                unit_owner = 0
            else:
                unit_owner = int(unit_owner)

            if unit_type is None or unit_type == 'null':
                unit_type = 0
            else:
                unit_type = int(unit_type)

            if unit_health is None or unit_health == 'null':
                unit_health = 0
            else:
                unit_health = int(unit_health)

            if unit_veteran is None or unit_veteran == 'null':
                unit_veteran = 0
            else:
                unit_veteran = int(unit_veteran)

            if improvement_progress is None or improvement_progress == 'null':
                if improvement == 1:
                    if climate == 7:  # Just a logic from original code
                        improvement_progress = 2
                    else:
                        improvement_progress = 0
                else:
                    improvement_progress = 0
            else:
                improvement_progress = int(improvement_progress)

            self.current_observation[i] = [
                terrain,
                resource,
                improvement,
                climate,
                border,
                improvement_owner,
                unit_owner,
                unit_type,
                unit_health,
                improvement_progress,
                0,
                0
            ]

            if improvement == 1 and improvement_owner == 1:
                self.p1capital = (x, y)
                self.p1capitalindex = i
                starting_unit = self.start_units.get(climate, 1)
                unit_health = self.unit_stats[starting_unit][0]
                self.current_observation[i][6] = 1
                self.current_observation[i][7] = starting_unit
                self.current_observation[i][8] = unit_health
                self.units.append([i, 1, starting_unit, unit_health])
            elif improvement == 1 and improvement_owner == 2:
                self.p2capital = (x, y)
                self.p2capitalindex = i
                starting_unit = self.start_units.get(climate, 1)
                unit_health = self.unit_stats[starting_unit][0]
                self.current_observation[i][6] = 2
                self.current_observation[i][7] = starting_unit
                self.current_observation[i][8] = unit_health
                self.units.append([i, 2, starting_unit, unit_health])

            if self.current_observation[i][2] == 1:
                adj = self.get_adjacent_indices(i)
                for b in range(len(adj)):
                    if self.tile_cities[adj[b]] == 0:
                        self.tile_cities[adj[b]] = i

        if self.p1capitalindex is None or self.p2capitalindex is None:
            raise ValueError("Capitals for both players must be present in the map.")

        self.update_borders()
        self.initialize_explored_areas()

        self.start_turn()
        self.update_obs()
        return self.get_obs(self.turn)

    def step(self, action):
        if self.game_ended:
            # If game already ended, return done=True and no changes
            return self.get_obs(self.turn), 0, True, {}

        reward = 0
        done = False
        info = {}

        # Proceed only if game not ended
        if self.step_phase == 0:
            if action == 121:
                self.end_turn()
                # Check for game end condition
                if self.turn_count >= self.total_turn_limit:
                    self.end_game()
                    done = True
                    info = {
                        "turns": self.turn_count,
                        "p1_points": self.final_p1_points,
                        "p2_points": self.final_p2_points
                    }
                return self.get_obs(self.turn), reward, done, info

            self.selected_tile = action
            if not (0 <= self.selected_tile < 121):
                self.step_phase = 0
                return self.get_obs(self.turn), reward, done, info

            tile = self.current_observation[self.selected_tile]
            unit_owner = tile[6]
            resource = tile[1]
            border = tile[4]
            has_attacked = tile[10]
            has_moved = tile[11]
            improvement_owner = tile[5]
            improvement = tile[2]

            valid = False
            if unit_owner == self.turn:
                # If the unit hasn't moved or if it can still attack
                if has_moved == 0 or (has_attacked == 0 and self.can_attack_adjacent(self.selected_tile) != []):
                    valid = True
            if (resource != 0 or tile[0] == 5) and border == self.turn:
                valid = True
            if improvement_owner == self.turn and improvement == 1:
                valid = True

            if valid:
                self.step_phase = 1
            else:
                self.step_phase = 0
            return self.get_obs(self.turn), reward, done, info

        elif self.step_phase == 1:
            self.selected_action = action
            tile = self.current_observation[self.selected_tile]
            unit_owner = tile[6]
            unit_type = tile[7]
            unit_health = tile[8]
            resource = tile[1]
            has_moved = tile[11]
            has_attacked = tile[10]
            border = tile[4]
            improvement = tile[2]
            improvement_owner = tile[5]

            valid = False
            if self.selected_action == 1:
                if unit_owner == self.turn:
                    if has_moved == 0:
                        valid = True
                    elif has_attacked == 0 and self.can_attack_adjacent(self.selected_tile) != []:
                        valid = True
            elif self.selected_action == 2:
                if resource != 0 and border == self.turn:
                    valid = True
            elif self.selected_action == 3:
                valid = True
            elif self.selected_action in [4, 5]:
                if improvement_owner == self.turn and improvement == 1:
                    valid = True

            if valid:
                if self.selected_action == 2:
                    success = self.harvest_resource(self.selected_tile)
                    self.step_phase = 0
                    return self.get_obs(self.turn), reward, done, info
                elif self.selected_action == 1:
                    self.step_phase = 2
                    return self.get_obs(self.turn), reward, done, info
                elif self.selected_action == 3:
                    success = self.capture_village(self.selected_tile)
                    self.step_phase = 0
                    return self.get_obs(self.turn), reward, done, info
                elif self.selected_action == 4:
                    success = self.train_unit(self.selected_tile, 1)
                    self.step_phase = 0
                    return self.get_obs(self.turn), reward, done, info
                elif self.selected_action == 5:
                    success = self.train_unit(self.selected_tile, 4)
                    self.step_phase = 0
                    return self.get_obs(self.turn), reward, done, info
            else:
                self.step_phase = 0
                return self.get_obs(self.turn), reward, done, info

        elif self.step_phase == 2:
            target_tile_index = action
            if not (0 <= target_tile_index < 121):
                self.step_phase = 0
                return self.get_obs(self.turn), reward, done, info

            start_tile_index = self.selected_tile
            start_tile = self.current_observation[start_tile_index]
            has_moved = start_tile[11]
            has_attacked = start_tile[10]

            if has_moved == 0:
                success = self.move_unit(target_tile_index, start_tile_index)
                if success:
                    self.step_phase = 0
                    return self.get_obs(self.turn), reward, done, info
            if has_attacked == 0:
                success = self.attack(target_tile_index, start_tile_index)
                if success:
                    self.step_phase = 0
                    return self.get_obs(self.turn), reward, done, info

            return self.get_obs(self.turn), reward, done, info
        else:
            self.step_phase = 0
            return self.get_obs(self.turn), reward, done, info

    def initialize_explored_areas(self):
        for player in [1, 2]:
            capital_index = self.p1capitalindex if player == 1 else self.p2capitalindex
            explored_map = self.p1_explored if player == 1 else self.p2_explored
            row, col = divmod(capital_index, 11)
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    r = row + dr
                    c = col + dc
                    if 0 <= r < 11 and 0 <= c < 11:
                        idx = r * 11 + c
                        explored_map[idx] = True

    def update_borders(self):
        for i in range(len(self.current_observation)):
            improvement_owner = self.current_observation[i][5]
            improvement = self.current_observation[i][2]
            if improvement_owner in [1, 2] and improvement == 1:
                adjacent_indices = self.get_adjacent_indices(i)
                for idx in adjacent_indices:
                    self.current_observation[idx][4] = improvement_owner

    def get_obs(self, player=0):
        if player == 1:
            return np.concatenate((self.p1_observation, self.p1stars, self.zeros), axis=0)
        elif player == 2:
            return np.concatenate((self.p2_observation, self.p2stars, self.zeros), axis=0)
        elif player == 0:
            return np.concatenate((self.current_observation, self.p1stars, self.p2stars), axis=0)

    def reset_unit_actions(self, player):
        for i in range(len(self.current_observation)):
            tile = self.current_observation[i]
            if tile[6] == player:
                tile[10] = 0
                tile[11] = 0

    def harvest_resource(self, tile_index):
        tile = self.current_observation[tile_index]
        resource = tile[1]
        if self.turn == 1:
            stars = self.p1stars[0][0]
        elif self.turn == 2:
            stars = self.p2stars[0][0]
        if resource in [1, 2, 3] and stars >= 2:
            self.current_observation[tile_index][1] = 0
            if self.turn == 1:
                self.p1stars[0][0] -= 2
            elif self.turn == 2:
                self.p2stars[0][0] -= 2
            city = self.tile_cities[tile_index]
            self.current_observation[city][9] += 1
            self.check_upgrade(city)
            self.update_obs()
            return True
        elif resource in [4, 5, 7] and stars >= 5:
            if self.current_observation[tile_index][1] == 4:
                self.current_observation[tile_index][2] = 4
            elif self.current_observation[tile_index][1] == 5:
                self.current_observation[tile_index][2] = 5
            self.current_observation[tile_index][1] = 0
            if self.turn == 1:
                self.p1stars[0][0] -= 5
            elif self.turn == 2:
                self.p2stars[0][0] -= 5
            city = self.tile_cities[tile_index]
            for i in range(2):
                self.current_observation[city][9] += 1
                self.check_upgrade(city)
            self.update_obs()
            return True
        else:
            return False

    def move_unit(self, target, start):
        start_tile = self.current_observation[start]
        unit_owner = start_tile[6]
        unit_type = start_tile[7]
        unit_health = start_tile[8]

        if unit_owner != self.turn:
            return False

        if start_tile[11] == 1:
            return False

        unit_stats = self.unit_stats.get(unit_type)
        if unit_stats is None:
            return False
        movement_range = unit_stats[3]

        path = self.find_path(start, target, movement_range)
        if not path:
            return False

        for idx in path[1:]:
            if self.current_observation[idx][6] != 0:
                return False

            self.current_observation[idx][6] = unit_owner
            self.current_observation[idx][7] = unit_type
            self.current_observation[idx][8] = unit_health
            self.current_observation[idx][10] = start_tile[10]
            self.current_observation[idx][11] = 1

            for unit in self.units:
                if unit[0] == start:
                    unit[0] = idx
                    break

            self.current_observation[start][6] = 0
            self.current_observation[start][7] = 0
            self.current_observation[start][8] = 0
            self.current_observation[start][10] = 0
            self.current_observation[start][11] = 0

            start = idx

        self.update_obs()
        return True

    def find_path(self, start, goal, max_distance):
        from collections import deque

        visited = set()
        queue = deque()
        queue.append((start, [start]))
        visited.add(start)

        # Determine which explored map to use
        explored_map = self.p1_explored if self.turn == 1 else self.p2_explored

        while queue:
            current, path = queue.popleft()

            # Check if we've reached the goal and are within max_distance
            if current == goal and len(path) - 1 <= max_distance:
                return path

            # If path length exceeds max_distance, skip further exploration
            if len(path) - 1 >= max_distance:
                continue

            # Explore neighbors
            for neighbor in self.get_adjacent_indices(current):
                if neighbor not in visited and explored_map[neighbor]:
                    tile = self.current_observation[neighbor]
                    terrain = tile[0]
                    unit_owner_on_tile = tile[6]

                    # Skip tiles that are impassable:
                    # Example: terrain 1 (Ocean), 2 (Water) might be considered impassable
                    if terrain in [1, 2]:
                        continue

                    # Skip tiles occupied by any unit
                    if unit_owner_on_tile != 0:
                        continue

                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None  # No path found within movement range

    def get_adjacent_indices(self, index):
        row, col = divmod(index, 11)
        adjacent = []
        directions = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1), (0, 1),
            (1, -1), (1, 0), (1, 1)
        ]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < 11 and 0 <= c < 11:
                adjacent.append(r * 11 + c)
        return adjacent

    def update_obs(self):
        for player in [1, 2]:
            explored_map = self.p1_explored if player == 1 else self.p2_explored
            observation = np.zeros_like(self.current_observation)
            for idx in range(121):
                if explored_map[idx]:
                    observation[idx] = self.current_observation[idx]
            for unit in self.units:
                if unit[1] == player:
                    idx = unit[0]
                    unit_reveal_indices = self.get_revealed_indices(idx)
                    for reveal_idx in unit_reveal_indices:
                        if not explored_map[reveal_idx]:
                            explored_map[reveal_idx] = True
                        observation[reveal_idx] = self.current_observation[reveal_idx]
            if player == 1:
                self.p1_observation = observation
            else:
                self.p2_observation = observation

    def get_revealed_indices(self, center_index):
        row, col = divmod(center_index, 11)
        revealed_indices = []
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                r = row + dr
                c = col + dc
                if 0 <= r < 11 and 0 <= c < 11:
                    idx = r * 11 + c
                    revealed_indices.append(idx)
        return revealed_indices

    def end_turn(self):
        # Heal units that didn't move
        for i in range(len(self.current_observation)):
            tile = self.current_observation[i]
            if tile[6] == self.turn:
                if tile[11] == 0:
                    unit_type = tile[7]
                    max_health = self.unit_stats[unit_type][0]
                    heal_amount = 2
                    if tile[4] == self.turn:
                        heal_amount = 4
                    tile[8] = min(tile[8] + heal_amount, max_health)

        # Switch turn
        self.turn = 3 - self.turn
        self.step_phase = 0
        self.update_obs()
        self.start_turn()

        # Each end_turn call increments turn_count by 1
        self.turn_count += 1

    def start_turn(self):
        self.collect_stars()
        self.reset_unit_actions(self.turn)
        self.update_obs()

    def check_upgrade(self, index):
        # Check city improvement progress for potential upgrades
        # Not fully defined, but let's leave as is. No changes required for scoring here.
        if self.current_observation[index][9] == 5:
            if self.current_observation[index][5] == 1:
                self.p1stars[0] += 5
            elif self.current_observation[index][5] == 2:
                self.p2stars[0] += 5
        if self.current_observation[index][9] == 9:
            self.current_observation[index][9] += 3

    def calculate_city_stp(self, population):
        stars_per_turn = 1
        threshold = 2
        increment = 2
        while population >= threshold:
            stars_per_turn += 1
            increment += 1
            threshold += increment
        if population > 1:
            stars_per_turn += 1
        return stars_per_turn

    def collect_stars(self):
        for i in range(len(self.current_observation)):
            if self.current_observation[i][2] == 1 and self.current_observation[i][5] == self.turn:
                if self.turn == 1:
                    self.p1stars[0][0] += self.calculate_city_stp(self.current_observation[i][9])
                elif self.turn == 2:
                    self.p2stars[0][0] += self.calculate_city_stp(self.current_observation[i][9])
        if self.turn == 1:
            self.p1stars[0][0] += 1
        elif self.turn == 2:
            self.p2stars[0][0] += 1

    def can_attack_adjacent(self, index):
        adjacent_indices = self.get_adjacent_indices(index)
        current_player = self.turn
        attackable_units = []
        for adj_index in adjacent_indices:
            adj_tile = self.current_observation[adj_index]
            adj_unit_owner = adj_tile[6]
            if adj_unit_owner != 0 and adj_unit_owner != current_player:
                attackable_units.append(adj_index)
        return attackable_units

    def attack(self, target_tile_index, start_tile_index):
        attacker_tile = self.current_observation[start_tile_index]
        defender_tile = self.current_observation[target_tile_index]

        if attacker_tile[6] != self.turn:
            return False
        if defender_tile[6] == 0 or defender_tile[6] == self.turn:
            return False
        if attacker_tile[10] == 1:
            return False

        attacker_type = attacker_tile[7]
        defender_type = defender_tile[7]

        attacker_stats = self.unit_stats.get(attacker_type)
        defender_stats = self.unit_stats.get(defender_type)

        if attacker_stats is None or defender_stats is None:
            return False

        attacker_max_health = attacker_stats[0]
        defender_max_health = defender_stats[0]
        attacker_health = attacker_tile[8]
        defender_health = defender_tile[8]

        attacker_attack = attacker_stats[1]
        attacker_defense = attacker_stats[2]

        defender_attack = defender_stats[1]
        defender_defense = defender_stats[2]

        defense_bonus = self.get_defense_bonus(target_tile_index)

        attackForce = attacker_attack * (attacker_health / attacker_max_health)
        defenseForce = defender_defense * (defender_health / defender_max_health) * defense_bonus

        totalDamage = attackForce + defenseForce

        attackResult = np.ceil((attackForce / totalDamage) * attacker_attack * 4.5)
        defenseResult = np.floor((defenseForce / totalDamage) * defender_defense * 4.5)

        defender_health -= attackResult
        defender_health = max(defender_health, 0)

        if defender_health == 0:
            defender_tile[6] = 0
            defender_tile[7] = 0
            defender_tile[8] = 0
            defender_tile[10] = 0
            defender_tile[11] = 0
            self.units = [unit for unit in self.units if unit[0] != target_tile_index]
        else:
            defender_tile[8] = defender_health

        if defender_health > 0:
            attacker_health -= defenseResult
            attacker_health = max(attacker_health, 0)
            if attacker_health == 0:
                attacker_tile[6] = 0
                attacker_tile[7] = 0
                attacker_tile[8] = 0
                attacker_tile[10] = 0
                attacker_tile[11] = 0
                self.units = [unit for unit in self.units if unit[0] != start_tile_index]
            else:
                attacker_tile[8] = attacker_health

        attacker_tile[10] = 1
        self.update_obs()
        return True

    def get_defense_bonus(self, tile_index):
        tile = self.current_observation[tile_index]
        terrain = tile[0]
        improvement = tile[2]
        border = tile[4]
        unit_owner = tile[6]

        standard_defense_terrains = [4, 5]
        if terrain in standard_defense_terrains:
            return 1.5
        if improvement == 1 and border == unit_owner:
            return 4
        return 1

    def train_unit(self, tile_index, unit):
        if self.current_observation[tile_index][5] == self.turn and self.current_observation[tile_index][2] == 1:
            if self.turn == 1:
                stars = self.p1stars[0][0]
            elif self.turn == 2:
                stars = self.p2stars[0][0]
            if unit == 1:
                if stars < 2:
                    return False
                else:
                    self.current_observation[tile_index][6] = self.turn
                    self.current_observation[tile_index][7] = 1
                    self.current_observation[tile_index][8] = 10
                    self.current_observation[tile_index][10] = 1
                    self.current_observation[tile_index][11] = 1
                    self.update_obs()
                    return True
            elif unit == 4:
                if stars < 3:
                    return False
                else:
                    self.current_observation[tile_index][6] = self.turn
                    self.current_observation[tile_index][7] = 4
                    self.current_observation[tile_index][8] = 10
                    self.current_observation[tile_index][10] = 1
                    self.current_observation[tile_index][11] = 1
                    self.update_obs()
                    return True
            else:
                return False
        else:
            return False

    def capture_village(self, tile_index):
        if self.current_observation[tile_index][2] == 1 and self.current_observation[tile_index][5] == 0 and \
                self.current_observation[tile_index][11] == 0 and self.current_observation[tile_index][6] == self.turn:
            self.current_observation[tile_index][5] = self.turn
            self.current_observation[tile_index][11] = 1
            self.update_borders()
            adj = self.get_adjacent_indices(tile_index)
            for b in range(len(adj)):
                if self.tile_cities[adj[b]] == 0:
                    self.tile_cities[adj[b]] = tile_index
            self.update_obs()
            return True
        else:
            return False

    def get_mask(self):
        mask = [0]*121
        if self.turn == 1:
            tiles = self.p1_observation
            stars = self.p1stars[0][0]
        elif self.turn == 2:
            tiles = self.p2_observation
            stars = self.p2stars[0][0]

        if self.step_phase == 0:
            for i in range(len(tiles)):
                tile = tiles[i]
                if tile[0] != 0:
                    if tile[4] == self.turn and tile[1] != 0:
                        if stars >= 5:
                            mask[i] = 1
                        elif tile[1] in [1,2,3] and stars >= 2:
                            mask[i] = 1
                    if tile[6] == self.turn and (tile[11] == 0 or (tile[10] == 0 and self.find_valid_attacks(i) != [])):
                        mask[i] = 1
                    if tile[2] == 1 and tile[5] == self.turn and tile[6] == 0:
                        mask[i] = 1
        if self.step_phase == 1:
            mask = [0,0,0,0,0,0]
            tile = tiles[self.selected_tile]
            if tile[1] != 0:
                if stars >= 5:
                    mask[2] = 1
                elif stars >= 2 and tile[1] in [1,2,3]:
                    mask[2] = 1
            if tile[6] == self.turn and (tile[10] == 0 or (tile[10] == 0 and self.find_valid_attacks(self.selected_tile) != [])):
                mask[1] = 1
            if tile[6] == 0 and tile[2] == 1 and tile[11] == 0:
                mask[3] = 1
            if tile[2] == 1 and tile[5] == self.turn and tile[6] == 0:
                if stars >= 2:
                    mask[4] = 1
                if stars >= 3:
                    mask[5] = 1
        if self.step_phase == 2:
            unit_type = tiles[self.selected_tile][7]
            for i in range(len(tiles)):
                tile = tiles[i]
                if tile[0] != 0:
                    # Calculate Chebyshev distance
                    distance = max(
                        abs((i - 1) // 11 - (self.selected_tile - 1) // 11),
                        abs((i - 1) % 11 - (self.selected_tile - 1) % 11)
                    )
                    if tile[6] == 0 and self.find_path(self.selected_tile, i, self.unit_stats.get(unit_type)[3]) != None:
                        mask[i] = 1
                    elif tile[6] != 0 and tile[6] != self.turn and distance <= self.unit_stats.get(unit_type)[4]:
                        mask[i] = 1
        return mask

    def find_valid_attacks(self, index):
        from collections import deque

        tile = self.current_observation[index]
        unit_owner = tile[6]
        unit_type = tile[7]

        if unit_owner != self.turn:
            return []

        unit_stats = self.unit_stats.get(unit_type)
        if unit_stats is None:
            return []

        movement_range = unit_stats[3]
        explored_map = self.p1_explored if self.turn == 1 else self.p2_explored

        reachable_tiles = set()
        queue = deque()
        queue.append((index, 0))
        visited = set()
        visited.add(index)

        while queue:
            current_index, depth = queue.popleft()
            if depth > movement_range:
                continue
            reachable_tiles.add(current_index)
            neighbors = self.get_adjacent_indices(current_index)
            for neighbor in neighbors:
                if neighbor not in visited and explored_map[neighbor]:
                    tile = self.current_observation[neighbor]
                    terrain = tile[0]
                    if terrain in [1, 2]:
                        continue
                    if tile[6] != 0:
                        continue
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        attackable_units = set()
        for tile_index in reachable_tiles:
            adjacents = self.get_adjacent_indices(tile_index)
            for adj_index in adjacents:
                adj_tile = self.current_observation[adj_index]
                adj_unit_owner = adj_tile[6]
                if adj_unit_owner != 0 and adj_unit_owner != self.turn:
                    attackable_units.add(adj_index)

        return list(attackable_units)

    def encode_state(self, state):
        # Provided as-is from original code
        feature_lens = [6, 8, 6, 17, 3, 3, 3, 5, 1, 1, 2, 2]
        state_tensor = torch.zeros((sum(feature_lens), 11, 11))

        for i in range(len(state)):
            x = i % 11
            y = i // 11
            state_tensor[state[i][0], x, y] = 1
            state_tensor[feature_lens[0:1] + state[i][1], x, y] = 1
            state_tensor[sum(feature_lens[0:2]) + state[i][2], x, y] = 1
            state_tensor[sum(feature_lens[0:3]) + state[i][3], x, y] = 1
            state_tensor[sum(feature_lens[0:4]) + state[i][4], x, y] = 1
            state_tensor[sum(feature_lens[0:5]) + state[i][5], x, y] = 1
            state_tensor[sum(feature_lens[0:6]) + state[i][6], x, y] = 1
            state_tensor[sum(feature_lens[0:7]) + state[i][7], x, y] = 1
            state_tensor[sum(feature_lens[0:8]), x, y] = state[i][8]
            state_tensor[sum(feature_lens[0:9]), x, y] = state[i][9]
            state_tensor[sum(feature_lens[0:10]) + state[i][10], x, y] = 1
            state_tensor[sum(feature_lens[0:11]) + state[i][11], x, y] = 1

        return state_tensor

    def end_game(self):
        # Calculate final points for both players
        p1_points = self.calculate_points(1)
        p2_points = self.calculate_points(2)
        self.final_p1_points = p1_points
        self.final_p2_points = p2_points
        self.game_ended = True

    def calculate_points(self, player):
        # Points:
        # - 1000 points per city
        # - 250 points per upgraded city (improvement_progress > 0)
        # - 50 points per explored tile
        # - 100 points per unit

        points = 0
        explored_map = self.p1_explored if player == 1 else self.p2_explored

        # Cities and upgraded cities
        for i in range(len(self.current_observation)):
            tile = self.current_observation[i]
            if tile[2] == 1 and tile[5] == player:
                # City owned by player
                points += 1000
                if tile[9] > 0:
                    # Consider any improvement_progress > 0 as upgraded
                    points += 250

        # Explored tiles
        explored_count = np.sum(explored_map)
        points += explored_count * 50

        # Units
        unit_count = 0
        for i in range(len(self.current_observation)):
            tile = self.current_observation[i]
            if tile[6] == player:  # unit_owner
                unit_count += 1
        points += unit_count * 100

        return points

    def get_turn_and_points(self):
        # Returns how many turns have passed and how many points p1 and p2 have.
        # This should be called after the game ends.
        if not self.game_ended:
            # If game not ended yet, calculate temporary points or return 0
            p1_points = self.calculate_points(1)
            p2_points = self.calculate_points(2)
            return self.turn_count, p1_points, p2_points
        else:
            return True, self.final_p1_points, self.final_p2_points
