import gym
import random
import numpy as np
import torch
from gym import spaces
import json

#terrain,               0
#resource,              1
#improvement,           2
#climate,               3
#border,                4
#improvement_owner,     5
#unit_owner,            6
#unit_type,             7
#unit_health,           8
#improvement_progress,  9
# Has Attacked          10
# Has Moved             11

class PolytopiaEnv(gym.Env):
    def __init__(self):
        super(PolytopiaEnv, self).__init__()
        self.filename = r"C:\Users\samth\Downloads\Maps.json"

        # Adjusted action space to include passing the turn (action code 121)
        self.action_space = spaces.Discrete(122)  # Actions from 0 to 120 are tiles, 121 is pass, 122-131 are tech tree
        self.observation_space = spaces.Box(
            low=np.array([[0]*12], dtype=np.uint8).repeat(121, axis=0),
            high=np.array([[99]*12], dtype=np.uint8).repeat(121, axis=0),
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
        # Load unit stats and starting units
        self.start_units = {
            1: 1,   # Xin-xi - Warrior
            2: 1,   # Imperius - Warrior
            3: 1,   # Bardur - Warrior
            4: 4,   # Oumaji - Rider
            5: 1,   # Kickoo - Warrior
            6: 2,   # Hoodrick - Archer
            7: 1,   # Luxidoor - Warrior
            8: 8,   # Vengir - Swordsman
            9: 1,   # Zebasi - Warrior
            10: 1,  # Ai-Mo - Warrior
            11: 3,  # Quetzali - Defender
            12: 1,  # Yădakk - Warrior
            13: 19, # Aquarion - Amphibian
            14: 1,  # Elyrion - Warrior
            15: 28, # Polaris - Mooni
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

        # Initialize explored maps
        self.p1_explored = np.zeros(121, dtype=bool)
        self.p2_explored = np.zeros(121, dtype=bool)

        # Initialize capital indices
        self.p1capitalindex = None
        self.p2capitalindex = None

        self.p1stars[0][0] = 2
        self.p2stars[0][0] = 2
        # Map terrain/resource/improvement names to numeric values
        terrain_mapping = {
            'Ocean': 1,
            'Water': 2,
            'Field': 3,
            'Mountain': 4,
            'Forest': 5,
            # Add other terrains as needed
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
            # Add other resources as needed
        }

        improvement_mapping = {
            'City': 1,
            'Ruin': 2,
            'Lighthouse': 3,
            'None': 0,
            # Capitals will be handled separately if needed
        }

        # Process each tile
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
            border = 0  # Will be updated later

            # Corrected key names
            improvement_owner = tile.get('improvement owner')
            unit_owner = tile.get('unit owner')
            unit_type = tile.get('unit type')
            unit_health = tile.get('unit health')
            unit_veteran = tile.get('unit veteran')
            improvement_progress = tile.get('improvement progress')

            # Convert 'null' and 'None' strings to appropriate values
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

            # Handle improvement progress
            if improvement_progress is None or improvement_progress == 'null':
                if improvement == 1:  # If it's a city
                    if climate == 7:  # Assuming climate 7 is Luxidor
                        improvement_progress = 2
                    else:
                        improvement_progress = 0
                else:
                    improvement_progress = 0
            else:
                improvement_progress = int(improvement_progress)
            # Set the observation
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
                0,  # Has Attacked
                0   # Has Moved
            ]

            # Handle capitals and starting units
            if improvement == 1 and improvement_owner == 1:
                self.p1capital = (x, y)
                self.p1capitalindex = i
                starting_unit = self.start_units.get(climate, 1)
                unit_health = self.unit_stats[starting_unit][0]
                self.current_observation[i][6] = 1  # Unit owner
                self.current_observation[i][7] = starting_unit
                self.current_observation[i][8] = unit_health
                self.units.append([i, 1, starting_unit, unit_health])
            elif improvement == 1 and improvement_owner == 2:
                self.p2capital = (x, y)
                self.p2capitalindex = i
                starting_unit = self.start_units.get(climate, 1)
                unit_health = self.unit_stats[starting_unit][0]
                self.current_observation[i][6] = 2  # Unit owner
                self.current_observation[i][7] = starting_unit
                self.current_observation[i][8] = unit_health
                self.units.append([i, 2, starting_unit, unit_health])

            if self.current_observation[i][2] == 1:
                adj = self.get_adjacent_indices(i)
                for b in range(len(adj)):
                    if self.tile_cities[adj[b]] == 0:
                        self.tile_cities[adj[b]] = i
                    else:
                        pass
        if self.p1capitalindex is None or self.p2capitalindex is None:
            raise ValueError("Capitals for both players must be present in the map.")



        # Update borders
        self.update_borders()

        # Initialize explored areas around the capitals
        self.initialize_explored_areas()

        #start with 4 stars
        # Update observations for each player
        self.start_turn()
        self.update_obs()
        return self.get_obs(self.turn)

    def step(self, action):
        # action is an integer input
        # self.step_phase keeps track of the current phase
        # self.turn keeps track of the current player

        reward = 0  # Default reward is zero

        if self.step_phase == 0:
            if action == 121:
                # Player chooses to pass the turn
                self.end_turn()
                return True, reward  # Action succeeded

            # Phase 1: Select tile to act upon
            self.selected_tile = action  # tile index
            if not (0 <= self.selected_tile < 121):
                # Invalid tile index
                print("Failure: Invalid tile index selected.")
                self.step_phase = 0
                return False, reward  # Action failed

            tile = self.current_observation[self.selected_tile]
            unit_owner = tile[6]
            unit_type = tile[7]
            unit_health = tile[8]
            resource = tile[1]
            terrain = tile[0]
            border = tile[4]
            has_attacked = tile[10]
            has_moved = tile[11]
            improvement_owner = tile[5]
            improvement = tile[2]
            # Determine if tile selection is valid
            valid = False
            if unit_owner == self.turn:
                if has_moved == 0:
                    valid = True
                elif has_attacked == 0 and self.can_attack_adjacent(self.selected_tile) != []:
                    valid = True
                else:
                    valid = False
            if (resource != 0 or terrain == 5) and border == self.turn:
                valid = True
            if improvement_owner == self.turn and improvement == 1:
                valid = True
            if valid:
                self.step_phase = 1
                return True, reward  # Action accepted
            else:
                print("Failure: Tile selection invalid, no valid unit or action found.")
                self.step_phase = 0  # Reset to phase 1
                return False, reward  # Action failed

        elif self.step_phase == 1:
            # Phase 2: Pick action type (1: move, 2: harvest)
            self.selected_action = action  # action code
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
                    elif has_attacked == 0:
                        attackable_units = self.can_attack_adjacent(self.selected_tile)
                        if attackable_units != []:
                            valid = True
                        else:
                            print("Failure: No attackable units adjacent.")
                            valid = False
                    else:
                        print("Failure: The unit has already moved and cannot attack.")
                        valid = False
            elif self.selected_action == 2:
                if resource != 0 and border == self.turn:
                    valid = True
                else:
                    print("Failure: No resource available to harvest or invalid border.")
            elif self.selected_action == 3:
                valid = True
            elif self.selected_action in [4,5]:
                if improvement_owner == self.turn and improvement == 1:
                    valid = True
                else:
                    print("Failure: Cannot train unit on tile without owned city")
            if valid:
                if self.selected_action == 2:
                    success = self.harvest_resource(self.selected_tile)
                    if success:
                        self.update_obs()
                        self.step_phase = 0  # Reset to phase 1 for next action
                        return True, reward  # Action succeeded
                    else:
                        print("Failure: Resource harvesting failed.")
                        self.step_phase = 0
                        return False, reward  # Action failed
                elif self.selected_action == 1:
                    self.step_phase = 2
                    return True, reward  # Action accepted
                elif self.selected_action == 3:
                    success = self.capture_village(self.selected_tile)
                    if success:
                        self.step_phase = 0
                        return True, reward
                elif self.selected_action == 4: #train warrior
                    success = self.train_unit(self.selected_tile, 1)
                    if success:
                        self.step_phase = 0
                        return True, reward #trained unit success
                    else:
                        print("Failure: train warrior failed")
                        self.step_phase = 0
                        return False, reward #failed to train unit
                elif self.selected_action == 5: #train rider
                    success = self.train_unit(self.selected_tile, 4)
                    if success:
                        self.step_phase = 0
                        return True, reward #trained unit success
                    else:
                        print("Failure: train rider failed")
                        self.step_phase = 0
                        return False, reward #failed to train unit
            else:
                print("Failure: Invalid action selected.")
                self.step_phase = 0  # Reset to phase 1
                return False, reward  # Action failed

        elif self.step_phase == 2:
            # Phase 3: Pick tile to act upon (only if it's a moving action)
            target_tile_index = action  # tile index to move to
            if not (0 <= target_tile_index < 121):
                print("Failure: Invalid target tile index.")
                self.step_phase = 0
                return False, reward  # Action failed

            start_tile_index = self.selected_tile
            tile = self.current_observation[start_tile_index]
            unit_owner = tile[6]
            unit_type = tile[7]
            unit_health = tile[8]
            has_moved = tile[11]
            has_attacked = tile[10]
            unit_stats = self.unit_stats.get(unit_type)
            if unit_stats is None:
                print("Failure: Invalid unit type.")
                self.step_phase = 0
                return False, reward  # Action failed
            movement_range = unit_stats[3]  # Movement range
            print(has_moved, has_attacked)
            if has_moved == 0:
                success = self.move_unit(target_tile_index, start_tile_index)
                if success:
                    print("moved")
                    self.update_obs()
                    self.step_phase = 0  # Reset to phase 1 for next action
                    return True, reward  # Action succeeded
            if has_attacked == 0:
                success = self.attack(target_tile_index, start_tile_index)
                if success:
                    print("attacked")
                    self.update_obs()
                    self.step_phase = 0  # Reset to phase 1 for next action
                    return True, reward  # Action succeeded
            print("move and attack failed")
            return False, reward
        else:
            print("Failure: Invalid phase detected.")
            self.step_phase = 0
            return False, reward  # Action failed

    def initialize_explored_areas(self):
        # Reveal a 5x5 area around each capital (tiles up to 2 units away)
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
        # Update border values for tiles adjacent to player cities
        for i in range(len(self.current_observation)):
            improvement_owner = self.current_observation[i][5]
            improvement = self.current_observation[i][2]
            if improvement_owner in [1, 2] and improvement == 1:
                adjacent_indices = self.get_adjacent_indices(i)
                for idx in adjacent_indices:
                    self.current_observation[idx][4] = improvement_owner  # Set border

    def get_obs(self, player=0):
        if player == 1:
            return np.concatenate((self.p1_observation, self.p1stars, self.zeros), axis=0)
        elif player == 2:
            return np.concatenate((self.p2_observation, self.p2stars, self.zeros), axis=0)
        elif player == 0:
            return np.concatenate((self.current_observation, self.p1stars,self.p2stars), axis=0)

    def reset_unit_actions(self, player):
        # Reset Has Attacked and Has Moved flags for the player's units
        for i in range(len(self.current_observation)):
            tile = self.current_observation[i]
            if tile[6] == player:
                tile[10] = 0  # Has Attacked
                tile[11] = 0  # Has Moved

    def harvest_resource(self, tile_index):
        # Implement resource harvesting logic here
        tile = self.current_observation[tile_index]
        resource = tile[1]
        if self.turn == 1:
            stars = self.p1stars[0][0]
        elif self.turn == 2:
            stars = self.p2stars[0][0]
        if resource in [1,2,3] and stars >= 2:
            self.current_observation[tile_index][1] = 0
            if self.turn == 1:
                self.p1stars[0][0] -= 2
            elif self.turn == 2:
                self.p2stars[0][0] -= 2
            city = self.tile_cities[tile_index]
            self.current_observation[self.tile_cities[tile_index]][9] += 1
            self.check_upgrade(city)
            return True
        elif resource in [4,5,7] and stars >= 5:
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
            return True
        else:
            return False

    def move_unit(self, target, start):
        # Implement pathfinding and move unit step by step
        # Check if start tile has a unit belonging to the player
        start_tile = self.current_observation[start]
        unit_owner = start_tile[6]
        unit_type = start_tile[7]
        unit_health = start_tile[8]

        if unit_owner != self.turn:
            return False  # Can't move a unit that doesn't belong to the player

        if start_tile[11] == 1:
            # Unit has already moved
            return False  # Cannot move unit

        # Check movement range
        unit_stats = self.unit_stats.get(unit_type)
        if unit_stats is None:
            return False  # Invalid unit type
        movement_range = unit_stats[3]

        # Find path using BFS, ensuring tiles are explored
        path = self.find_path(start, target, movement_range)
        if not path:
            return False  # No valid path found within movement range

        # Move the unit along the path
        for idx in path[1:]:  # Exclude the starting tile
            # Check if the tile is occupied by another unit
            if self.current_observation[idx][6] != 0:
                return False  # Cannot move into occupied tile

            # Move unit to the next tile
            self.current_observation[idx][6] = unit_owner
            self.current_observation[idx][7] = unit_type
            self.current_observation[idx][8] = unit_health
            self.current_observation[idx][10] = start_tile[10]  # Has Attacked remains the same
            self.current_observation[idx][11] = 1  # Set Has Moved to 1

            # Update units list
            for unit in self.units:
                if unit[0] == start:
                    unit[0] = idx
                    break

            # Clear the previous tile
            self.current_observation[start][6] = 0
            self.current_observation[start][7] = 0
            self.current_observation[start][8] = 0
            self.current_observation[start][10] = 0
            self.current_observation[start][11] = 0

            # Update start index for next iteration
            start = idx

        return True

    def find_path(self, start, goal, max_distance):
        # BFS for shortest path within movement range, allowing diagonal movement
        from collections import deque

        visited = set()
        queue = deque()
        queue.append((start, [start]))
        visited.add(start)

        # Determine which explored map to use
        explored_map = self.p1_explored if self.turn == 1 else self.p2_explored

        while queue:
            current, path = queue.popleft()

            if current == goal and len(path) - 1 <= max_distance:
                return path

            if len(path) - 1 >= max_distance:
                continue  # Exceeded movement range

            for neighbor in self.get_adjacent_indices(current):
                if neighbor not in visited and explored_map[neighbor]:
                    # Check if terrain is passable
                    tile = self.current_observation[neighbor]
                    terrain = tile[0]
                    # Add logic for impassable terrain if needed
                    # For now, assume all terrains are passable
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))

        return None  # No path found within movement range

    def get_adjacent_indices(self, index):
        # Get indices of adjacent tiles (including diagonals)
        row, col = divmod(index, 11)
        adjacent = []
        directions = [
            (-1, -1), (-1, 0), (-1, 1),  # Upper row
            (0, -1),          (0, 1),    # Same row
            (1, -1),  (1, 0),  (1, 1)    # Lower row
        ]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < 11 and 0 <= c < 11:
                adjacent.append(r * 11 + c)
        return adjacent

    def update_obs(self):
        # Update observations for both players
        for player in [1, 2]:
            # Determine which explored map and observation to use
            explored_map = self.p1_explored if player == 1 else self.p2_explored
            observation = np.zeros_like(self.current_observation)

            # Include all previously explored tiles
            for idx in range(121):
                if explored_map[idx]:
                    observation[idx] = self.current_observation[idx]

            # Reveal tiles around units owned by the player
            for unit in self.units:
                if unit[1] == player:
                    idx = unit[0]
                    unit_reveal_indices = self.get_revealed_indices(idx)
                    for reveal_idx in unit_reveal_indices:
                        if not explored_map[reveal_idx]:
                            explored_map[reveal_idx] = True  # Permanently reveal
                            observation[reveal_idx] = self.current_observation[reveal_idx]
                        else:
                            observation[reveal_idx] = self.current_observation[reveal_idx]

            # Assign to the player's observation
            if player == 1:
                self.p1_observation = observation
            else:
                self.p2_observation = observation

    def get_revealed_indices(self, center_index):
        # Reveal adjacent tiles (including diagonals)
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
            if tile[6] == self.turn:  # Unit owner is current player
                if tile[11] == 0:  # Has Moved is 0
                    unit_type = tile[7]
                    max_health = self.unit_stats[unit_type][0]
                    heal_amount = 2
                    if tile[4] == self.turn:  # Inside own city border
                        heal_amount = 4
                    tile[8] = min(tile[8] + heal_amount, max_health)
        # Switch turn to other player
        self.turn = 3 - self.turn  # Switch turn between 1 and 2
        self.step_phase = 0
        self.update_obs()
        # Start the next player's turn
        self.start_turn()

    def start_turn(self):
        # Collect stars at the start of the turn
        self.collect_stars()

        # Reset Has Attacked and Has Moved for all units of the current player
        for i in range(len(self.current_observation)):
            tile = self.current_observation[i]
            if tile[6] == self.turn:  # Check if the unit belongs to the current player
                tile[10] = 0  # Reset Has Attacked
                tile[11] = 0  # Reset Has Moved

        # Reset Has Attacked and Has Moved for all units of the current player
        self.reset_unit_actions(self.turn)
        self.update_obs()

    def check_upgrade(self, index):
        if self.current_observation[index][9] == 5:
            print("got stars")
            if self.current_observation[index][5] == 1:
                self.p1stars[0] += 5
            elif self.current_observation[index][5] == 2:
                self.p2stars[0] += 5
        if self.current_observation[index][9] == 9:
            print("got pop")
            self.current_observation[index][9] += 3

    def calculate_city_stp(self, population):
        stars_per_turn = 1  # Start with 1 STP
        threshold = 2  # Initial threshold to gain +1 STP
        increment = 2  # Initial increment (2, 3, 4, ...)

        while population >= threshold:
            stars_per_turn += 1
            increment += 1  # Increase the increment for the next threshold
            threshold += increment  # Update the threshold with the new increment
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
        """
        Checks if there are any attackable enemy units adjacent to the tile at the given index.
        Returns a list of indices of the attackable units if found, otherwise returns an empty list.
        """
        # Get the list of adjacent indices
        adjacent_indices = self.get_adjacent_indices(index)

        # Current player
        current_player = self.turn

        # List to store indices of attackable units
        attackable_units = []

        # Iterate over adjacent tiles
        for adj_index in adjacent_indices:
            adj_tile = self.current_observation[adj_index]
            adj_unit_owner = adj_tile[6]  # unit_owner

            # Check if there is an enemy unit
            if adj_unit_owner != 0 and adj_unit_owner != current_player:
                attackable_units.append(adj_index)  # Append the index of the attackable unit

        # Return the list of attackable units
        return attackable_units

    def attack(self, target_tile_index, start_tile_index):
        """
        Executes an attack from the unit at start_tile_index to the unit at target_tile_index.
        Updates unit health, handles unit deaths, and applies the damage formula as per the game's rules.
        Returns True if the attack was successful, False otherwise.
        """
        # Get attacker and defender tiles
        attacker_tile = self.current_observation[start_tile_index]
        defender_tile = self.current_observation[target_tile_index]

        # Check if attacker unit belongs to the current player
        if attacker_tile[6] != self.turn:
            return False  # Cannot attack with a unit that doesn't belong to the player

        # Check if defender unit exists and belongs to the enemy
        if defender_tile[6] == 0 or defender_tile[6] == self.turn:
            return False  # No enemy unit to attack

        # Check if the attacker has already attacked
        if attacker_tile[10] == 1:
            return False  # Unit has already attacked this turn

        # Retrieve attacker and defender stats
        attacker_type = attacker_tile[7]
        defender_type = defender_tile[7]

        attacker_stats = self.unit_stats.get(attacker_type)
        defender_stats = self.unit_stats.get(defender_type)

        if attacker_stats is None or defender_stats is None:
            return False  # Invalid unit types

        # Get max health from unit stats
        attacker_max_health = attacker_stats[0]
        defender_max_health = defender_stats[0]

        # Get current health
        attacker_health = attacker_tile[8]
        defender_health = defender_tile[8]

        # Get attack and defense values
        attacker_attack = attacker_stats[1]
        attacker_defense = attacker_stats[2]

        defender_attack = defender_stats[1]
        defender_defense = defender_stats[2]

        # Calculate defense bonus for defender
        defense_bonus = self.get_defense_bonus(target_tile_index)

        # Calculate attackForce and defenseForce
        attackForce = attacker_attack * (attacker_health / attacker_max_health)
        defenseForce = defender_defense * (defender_health / defender_max_health) * defense_bonus

        totalDamage = attackForce + defenseForce

        # Calculate attackResult and defenseResult
        attackResult = np.ceil((attackForce / totalDamage) * attacker_attack * 4.5)
        defenseResult = np.floor((defenseForce / totalDamage) * defender_defense * 4.5)

        # Apply damage to defender
        defender_health -= attackResult
        defender_health = max(defender_health, 0)  # Ensure health doesn't go below 0

        # Update defender's health or remove unit if dead
        if defender_health == 0:
            # Remove defender unit
            defender_tile[6] = 0  # Unit owner
            defender_tile[7] = 0  # Unit type
            defender_tile[8] = 0  # Unit health
            defender_tile[10] = 0  # Has Attacked
            defender_tile[11] = 0  # Has Moved

            # Remove from units list
            self.units = [unit for unit in self.units if unit[0] != target_tile_index]
        else:
            # Update defender's health
            defender_tile[8] = defender_health

        # Apply retaliation damage to attacker if defender is still alive
        if defender_health > 0:
            attacker_health -= defenseResult
            attacker_health = max(attacker_health, 0)  # Ensure health doesn't go below 0

            # Update attacker's health or remove unit if dead
            if attacker_health == 0:
                # Remove attacker unit
                attacker_tile[6] = 0  # Unit owner
                attacker_tile[7] = 0  # Unit type
                attacker_tile[8] = 0  # Unit health
                attacker_tile[10] = 0  # Has Attacked
                attacker_tile[11] = 0  # Has Moved

                # Remove from units list
                self.units = [unit for unit in self.units if unit[0] != start_tile_index]
            else:
                # Update attacker's health
                attacker_tile[8] = attacker_health

        # Mark attacker as having attacked
        attacker_tile[10] = 1  # Has Attacked

        # Update observations
        self.update_obs()

        return True

    def get_defense_bonus(self, tile_index):
        """
        Calculates the defense bonus for the unit on the given tile.
        Returns 1 (no bonus), 1.5 (standard bonus), or 4 (city wall bonus).
        """
        tile = self.current_observation[tile_index]
        terrain = tile[0]
        improvement = tile[2]
        border = tile[4]
        unit_owner = tile[6]

        # Standard defense bonus tiles (e.g., forests, mountains)
        standard_defense_terrains = [4, 5]  # Mountains and Forests

        if terrain in standard_defense_terrains:
            return 1.5  # Standard defense bonus

        # City wall bonus
        if improvement == 1 and border == unit_owner:
            # Assuming city walls exist, for now we consider all cities have walls
            return 4  # City wall defense bonus

        # No defense bonus
        return 1

    def train_unit(self, tile_index, unit):
        # 1: warrior
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
                    return True
            else:
                return False


        else:
            return False

    def capture_village(self, tile_index):
        if self.current_observation[tile_index][2] == 1 and self.current_observation[tile_index][5] == 0 and self.current_observation[tile_index][11] == 0 and self.current_observation[tile_index][6] == self.turn:
            self.current_observation[tile_index][5] = self.turn
            self.current_observation[tile_index][11] = 1
            self.update_borders()
            adj = self.get_adjacent_indices(tile_index)
            for b in range(len(adj)):
                if self.tile_cities[adj[b]] == 0:
                    self.tile_cities[adj[b]] = tile_index
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
            mask[121] = 1
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
            if tile[6] == self.turn and tile[2] == 1 and tile[11] == 0:
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
                    if tile[6] == 0 and distance <= self.unit_stats.get(unit_type)[3]:
                        mask[i] = 1
                    elif tile[6] != 0 and tile[6] != self.turn and distance <= self.unit_stats.get(unit_type)[4]:
                        mask[i] = 1
        return mask

    def find_valid_attacks(self, index):
        """
        Given a tile index with a unit, returns a list of indices of enemy units that can be attacked by that unit.
        Considers the unit's movement range and the possibility of attacking adjacent units after moving.
        """
        from collections import deque

        # Get the unit on the given tile
        tile = self.current_observation[index]
        unit_owner = tile[6]
        unit_type = tile[7]
        unit_health = tile[8]
        has_moved = tile[11]
        has_attacked = tile[10]

        # Check if there's a unit belonging to the current player
        if unit_owner != self.turn:
            return []  # No unit belonging to the player on this tile

        # Get unit stats
        unit_stats = self.unit_stats.get(unit_type)
        if unit_stats is None:
            return []  # Invalid unit type

        movement_range = unit_stats[3]

        # Set to store reachable tiles within movement range
        reachable_tiles = set()

        # BFS queue: (tile_index, depth)
        queue = deque()
        queue.append((index, 0))

        visited = set()
        visited.add(index)

        # Determine which explored map to use
        explored_map = self.p1_explored if self.turn == 1 else self.p2_explored

        # Collect all reachable tiles within movement range
        while queue:
            current_index, depth = queue.popleft()
            if depth > movement_range:
                continue  # Exceeded movement range

            reachable_tiles.add(current_index)

            # Get adjacent indices
            neighbors = self.get_adjacent_indices(current_index)
            for neighbor in neighbors:
                if neighbor not in visited:
                    # Check if the tile is passable (terrain is passable, no units)
                    tile = self.current_observation[neighbor]
                    terrain = tile[0]
                    unit_owner_neighbor = tile[6]
                    # For simplicity, assume terrain types 1 (Ocean) and 2 (Water) are impassable
                    if terrain in [1, 2]:
                        continue  # Can't pass through water
                    if unit_owner_neighbor != 0:
                        continue  # Tile occupied by a unit
                    if not explored_map[neighbor]:
                        continue  # Tile not explored
                    visited.add(neighbor)
                    queue.append((neighbor, depth + 1))

        # Set to store indices of enemy units that can be attacked
        attackable_units = set()

        # For each reachable tile, get adjacent tiles and check for enemy units
        for tile_index in reachable_tiles:
            adjacents = self.get_adjacent_indices(tile_index)
            for adj_index in adjacents:
                adj_tile = self.current_observation[adj_index]
                adj_unit_owner = adj_tile[6]
                if adj_unit_owner != 0 and adj_unit_owner != self.turn:
                    attackable_units.add(adj_index)

        # Return list of indices of enemy units that can be attacked
        return list(attackable_units)

    def encode_state(self, state):

            #  0         1         2           3      4        5           6           7          8           9           10          11
            # terrain  resource improvement climate  border  imp owner  unit owner   unit type   health   imp progress  has atckd    has mvd


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
