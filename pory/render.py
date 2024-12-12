import pygame

class Render:
    def __init__(self):
        pygame.init()
        self.TILE_SIZE = 64
        self.GRID_SIZE = 11
        self.SCREEN_SIZE = self.TILE_SIZE * self.GRID_SIZE
        self.screen = pygame.display.set_mode((self.SCREEN_SIZE, self.SCREEN_SIZE))
        pygame.display.set_caption("Polytopia Environment")
        self.running = True

        # Terrain colors
        self.terrain_colors = {
            1: (0, 0, 255),     # Ocean - Blue
            2: (0, 128, 255),   # Water - Light Blue
            3: (34, 139, 34),   # Field - Green
            4: (139, 69, 19),   # Mountain - Brown
            5: (0, 100, 0),     # Forest - Dark Green
            0: (255, 255, 255)  # None - White
        }

        # Load images
        self.resource_images = {
            1: pygame.image.load('images/game.png'),
            2: pygame.image.load('images/fruit.png'),
            3: pygame.image.load('images/fish.png'),
            4: pygame.image.load('images/crop.png'),
            5: pygame.image.load('images/metal.png'),
            6: pygame.image.load('images/starfish.png'),
            7: pygame.image.load('images/spores.png'),
        }

        self.improvement_images = {
            1: pygame.image.load('images/city.png'),
            2: pygame.image.load('images/ruin.png'),
            3: pygame.image.load('images/lighthouse.png'),
            4: pygame.image.load('images/Farm.png'),
            5: pygame.image.load('images/Mine.png'),
        }

        self.unit_images = {
            1: pygame.image.load('images/Warrior.png'),
            2: pygame.image.load('images/Archer.png'),
            3: pygame.image.load('images/Defender.png'),
            4: pygame.image.load('images/Rider.png'),
            # Add other unit images as needed
        }

    def get_city_level_and_progress(self, improvement_progress):
        # Initialize level requirements
        level_requirements = []
        cumulative = 0
        increment = 2
        while cumulative <= improvement_progress:
            cumulative += increment
            level_requirements.append(cumulative)
            increment += 1

        # Determine current level
        current_level = 0
        for idx, req in enumerate(level_requirements):
            if improvement_progress >= req:
                current_level = idx + 1
            else:
                break

        # Calculate progress towards the next level
        if current_level < len(level_requirements):
            prev_req = level_requirements[current_level - 1] if current_level > 0 else 0
            next_req = level_requirements[current_level]
            progress_towards_next = improvement_progress - prev_req
            total_progress_needed = next_req - prev_req
        else:
            # Max level reached
            progress_towards_next = 0
            total_progress_needed = 0

        return current_level, progress_towards_next, total_progress_needed

    def render(self, obs):
        # Process Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        # Define the width for the player stats column
        stats_width = 200
        screen_width = self.GRID_SIZE * self.TILE_SIZE + stats_width

        # Ensure the screen is set up with the extra width for stats
        self.screen = pygame.display.set_mode((screen_width, self.GRID_SIZE * self.TILE_SIZE))

        self.screen.fill((255, 255, 255))  # Fill background with white

        font = pygame.font.Font(None, 24)

        # Draw the grid
        for row in range(self.GRID_SIZE):
            for col in range(self.GRID_SIZE):
                tile_index = (self.GRID_SIZE - 1 - row) * self.GRID_SIZE + col

                # Ensure the tile index is within bounds
                if tile_index >= len(obs):
                    continue

                tile = obs[tile_index]

                # Unpack tile data
                terrain, resource, improvement, climate, border, improvement_owner, unit_owner, \
                    unit_type, unit_health, improvement_progress, has_attacked, has_moved = tile

                # Draw terrain
                color = self.terrain_colors.get(terrain, (255, 255, 255))
                pygame.draw.rect(self.screen, color,
                                 pygame.Rect(col * self.TILE_SIZE, row * self.TILE_SIZE, self.TILE_SIZE,
                                             self.TILE_SIZE))

                # Draw climate number
                text = font.render(str(climate), True, (0, 0, 0))
                self.screen.blit(text, (col * self.TILE_SIZE + 5, row * self.TILE_SIZE + 5))

                # Draw resource image
                if resource in self.resource_images:
                    resource_img = pygame.transform.scale(self.resource_images[resource], (32, 32))
                    self.screen.blit(resource_img,
                                     (col * self.TILE_SIZE + self.TILE_SIZE - 37, row * self.TILE_SIZE + 5))

                # Draw improvement image
                if improvement in self.improvement_images:
                    improvement_img = pygame.transform.scale(self.improvement_images[improvement], (32, 32))
                    self.screen.blit(improvement_img, (
                        col * self.TILE_SIZE + self.TILE_SIZE - 37, row * self.TILE_SIZE + self.TILE_SIZE - 37))

                # Handle city progression and upgrades
                if improvement == 1:
                    current_level, progress_towards_next, total_progress_needed = self.get_city_level_and_progress(improvement_progress)

                    # Draw level number above the city image
                    level_text = font.render(str(current_level), True, (0, 0, 0))
                    level_text_rect = level_text.get_rect(center=(col * self.TILE_SIZE + self.TILE_SIZE // 2, row * self.TILE_SIZE + 10))
                    self.screen.blit(level_text, level_text_rect)

                    # Draw progress bars representing progress towards the next level
                    if total_progress_needed > 0:
                        num_bars = total_progress_needed
                        filled_bars = progress_towards_next
                        bar_width = (self.TILE_SIZE - 10) // num_bars
                        bar_height = 5
                        bar_x = col * self.TILE_SIZE + 5
                        bar_y = row * self.TILE_SIZE + self.TILE_SIZE - bar_height - 5

                        # Draw the empty bars
                        for i in range(num_bars):
                            pygame.draw.rect(self.screen, (192, 192, 192),
                                             (bar_x + i * bar_width, bar_y, bar_width - 2, bar_height))

                        # Draw the filled bars
                        for i in range(int(filled_bars)):
                            pygame.draw.rect(self.screen, (0, 255, 0),
                                             (bar_x + i * bar_width, bar_y, bar_width - 2, bar_height))
                    else:
                        # Max level reached, no progress bar needed
                        pass

                # Draw border
                if border == 1:
                    border_color = (0, 0, 255)
                elif border == 2:
                    border_color = (255, 0, 0)
                else:
                    border_color = None

                if border_color:
                    pygame.draw.rect(self.screen, border_color,
                                     pygame.Rect(col * self.TILE_SIZE, row * self.TILE_SIZE, self.TILE_SIZE,
                                                 self.TILE_SIZE), 1)

                # Draw unit
                if unit_type > 0 and unit_type in self.unit_images:
                    unit_img = pygame.transform.scale(self.unit_images[unit_type], (32, 32))

                    # Check if unit has moved and can't attack
                    if has_moved == 1:
                        adjacent_enemy = False
                        adjacent_indices = self.get_adjacent_indices(tile_index)
                        for adj_idx in adjacent_indices:
                            adj_tile = obs[adj_idx]
                            if adj_tile[6] != 0 and adj_tile[6] != unit_owner:
                                adjacent_enemy = True
                                break
                        if not adjacent_enemy:
                            unit_img.fill((128, 128, 128), special_flags=pygame.BLEND_MULT)

                    self.screen.blit(unit_img,
                                     (col * self.TILE_SIZE + 5, row * self.TILE_SIZE + self.TILE_SIZE - 37))

                    # Draw unit health
                    health_text = font.render(str(unit_health), True, (255, 0, 0))
                    self.screen.blit(health_text,
                                     (col * self.TILE_SIZE + 20, row * self.TILE_SIZE + self.TILE_SIZE - 15))

                    # Draw unit owner
                    owner_text = font.render(str(unit_owner), True, (0, 0, 255))
                    self.screen.blit(owner_text,
                                     (col * self.TILE_SIZE + 5, row * self.TILE_SIZE + self.TILE_SIZE - 15))

        # Draw the player stats in the reserved column area
        stats_x = self.GRID_SIZE * self.TILE_SIZE + 10  # Start of the stats column
        y_offset = 10
        player_stats = {
            "Player 1 Stars": obs[121][0],
            "Player 2 Stars": obs[122][0]
        }
        for stat_name, stat_value in player_stats.items():
            stat_text = font.render(f"{stat_name}: {stat_value}", True, (0, 0, 0))
            self.screen.blit(stat_text, (stats_x, y_offset))
            y_offset += 30  # Increase y_offset for the next stat line

        pygame.display.flip()

    def get_adjacent_indices(self, index):
        row, col = divmod(index, 11)
        adjacent = []
        directions = [
            (-1, 0), (0, -1), (0, 1), (1, 0)  # Up, Left, Right, Down
        ]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < 11 and 0 <= c < 11:
                adjacent.append(r * 11 + c)
        return adjacent

    def close(self):
        pygame.quit()
import pygame

class Render:
    def __init__(self):
        pygame.init()
        self.TILE_SIZE = 64
        self.GRID_SIZE = 11
        self.SCREEN_SIZE = self.TILE_SIZE * self.GRID_SIZE
        self.screen = pygame.display.set_mode((self.SCREEN_SIZE, self.SCREEN_SIZE))
        pygame.display.set_caption("Polytopia Environment")
        self.running = True

        # Terrain colors
        self.terrain_colors = {
            1: (0, 0, 255),     # Ocean - Blue
            2: (0, 128, 255),   # Water - Light Blue
            3: (34, 139, 34),   # Field - Green
            4: (139, 69, 19),   # Mountain - Brown
            5: (0, 100, 0),     # Forest - Dark Green
            0: (255, 255, 255)  # None - White
        }

        # Load images
        self.resource_images = {
            1: pygame.image.load('images/game.png'),
            2: pygame.image.load('images/fruit.png'),
            3: pygame.image.load('images/fish.png'),
            4: pygame.image.load('images/crop.png'),
            5: pygame.image.load('images/metal.png'),
            6: pygame.image.load('images/starfish.png'),
            7: pygame.image.load('images/spores.png'),
        }

        self.improvement_images = {
            1: pygame.image.load('images/city.png'),
            2: pygame.image.load('images/ruin.png'),
            3: pygame.image.load('images/lighthouse.png'),
            4: pygame.image.load('images/Farm.png'),
            5: pygame.image.load('images/Mine.png'),
        }

        self.unit_images = {
            1: pygame.image.load('images/Warrior.png'),
            2: pygame.image.load('images/Archer.png'),
            3: pygame.image.load('images/Defender.png'),
            4: pygame.image.load('images/Rider.png'),
            # Add other unit images as needed
        }

    def get_city_level_and_progress(self, improvement_progress):
        # Initialize level requirements
        level_requirements = []
        cumulative = 0
        increment = 2
        while cumulative <= improvement_progress:
            cumulative += increment
            level_requirements.append(cumulative)
            increment += 1

        # Determine current level
        current_level = 0
        for idx, req in enumerate(level_requirements):
            if improvement_progress >= req:
                current_level = idx + 1
            else:
                break

        # Calculate progress towards the next level
        if current_level < len(level_requirements):
            prev_req = level_requirements[current_level - 1] if current_level > 0 else 0
            next_req = level_requirements[current_level]
            progress_towards_next = improvement_progress - prev_req
            total_progress_needed = next_req - prev_req
        else:
            # Max level reached
            progress_towards_next = 0
            total_progress_needed = 0

        return current_level, progress_towards_next, total_progress_needed

    def render(self, obs):
        # Process Pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False

        # Define the width for the player stats column
        stats_width = 200
        screen_width = self.GRID_SIZE * self.TILE_SIZE + stats_width

        # Ensure the screen is set up with the extra width for stats
        self.screen = pygame.display.set_mode((screen_width, self.GRID_SIZE * self.TILE_SIZE))

        self.screen.fill((255, 255, 255))  # Fill background with white

        font = pygame.font.Font(None, 24)
        smallfont = pygame.font.Font(None, 12)

        # Draw the grid
        for row in range(self.GRID_SIZE):
            for col in range(self.GRID_SIZE):
                tile_index = (self.GRID_SIZE - 1 - row) * self.GRID_SIZE + col

                # Ensure the tile index is within bounds
                if tile_index >= len(obs):
                    continue

                tile = obs[tile_index]

                # Unpack tile data
                terrain, resource, improvement, climate, border, improvement_owner, unit_owner, \
                    unit_type, unit_health, improvement_progress, has_attacked, has_moved = tile

                # Draw terrain
                color = self.terrain_colors.get(terrain, (255, 255, 255))
                pygame.draw.rect(self.screen, color,
                                 pygame.Rect(col * self.TILE_SIZE, row * self.TILE_SIZE, self.TILE_SIZE,
                                             self.TILE_SIZE))

                # Draw climate number
                text = font.render(str(climate), True, (0, 0, 0))
                self.screen.blit(text, (col * self.TILE_SIZE + 5, row * self.TILE_SIZE + 5))

                text = smallfont.render(str(tile_index), True, (220,220,220))
                self.screen.blit(text, (col * self.TILE_SIZE + 25, row * self.TILE_SIZE + 2))

                # Draw resource image
                if resource in self.resource_images:
                    resource_img = pygame.transform.scale(self.resource_images[resource], (32, 32))
                    self.screen.blit(resource_img,
                                     (col * self.TILE_SIZE + self.TILE_SIZE - 37, row * self.TILE_SIZE + 5))

                # Draw improvement image
                if improvement in self.improvement_images:
                    improvement_img = pygame.transform.scale(self.improvement_images[improvement], (32, 32))
                    self.screen.blit(improvement_img, (
                        col * self.TILE_SIZE + self.TILE_SIZE - 37, row * self.TILE_SIZE + self.TILE_SIZE - 37))

                # Handle city progression and upgrades
                if improvement == 1:
                    current_level, progress_towards_next, total_progress_needed = self.get_city_level_and_progress(improvement_progress)

                    # Draw level number above the city image
                    level_text = font.render(str(current_level), True, (0, 0, 0))
                    level_text_rect = level_text.get_rect(center=(col * self.TILE_SIZE + self.TILE_SIZE // 2, row * self.TILE_SIZE + 10))
                    self.screen.blit(level_text, level_text_rect)

                    # Draw progress bars representing progress towards the next level
                    if total_progress_needed > 0:
                        num_bars = total_progress_needed
                        filled_bars = progress_towards_next
                        bar_width = (self.TILE_SIZE - 10) // num_bars
                        bar_height = 5
                        bar_x = col * self.TILE_SIZE + 5
                        bar_y = row * self.TILE_SIZE + self.TILE_SIZE - bar_height - 5

                        # Draw the empty bars
                        for i in range(num_bars):
                            pygame.draw.rect(self.screen, (192, 192, 192),
                                             (bar_x + i * bar_width, bar_y, bar_width - 2, bar_height))

                        # Draw the filled bars
                        for i in range(int(filled_bars)):
                            pygame.draw.rect(self.screen, (0, 255, 0),
                                             (bar_x + i * bar_width, bar_y, bar_width - 2, bar_height))
                    else:
                        # Max level reached, no progress bar needed
                        pass

                # Draw border
                if border == 1:
                    border_color = (0, 0, 255)
                elif border == 2:
                    border_color = (255, 0, 0)
                else:
                    border_color = None

                if border_color:
                    pygame.draw.rect(self.screen, border_color,
                                     pygame.Rect(col * self.TILE_SIZE, row * self.TILE_SIZE, self.TILE_SIZE,
                                                 self.TILE_SIZE), 1)

                # Draw unit
                if unit_type > 0 and unit_type in self.unit_images:
                    unit_img = pygame.transform.scale(self.unit_images[unit_type], (32, 32))

                    # Check if unit has moved and can't attack
                    if has_moved == 1:
                        adjacent_enemy = False
                        adjacent_indices = self.get_adjacent_indices(tile_index)
                        for adj_idx in adjacent_indices:
                            adj_tile = obs[adj_idx]
                            if adj_tile[6] != 0 and adj_tile[6] != unit_owner:
                                adjacent_enemy = True
                                break
                        if not adjacent_enemy:
                            unit_img.fill((128, 128, 128), special_flags=pygame.BLEND_MULT)

                    self.screen.blit(unit_img,
                                     (col * self.TILE_SIZE + 5, row * self.TILE_SIZE + self.TILE_SIZE - 37))

                    # Draw unit health
                    health_text = font.render(str(unit_health), True, (255, 0, 0))
                    self.screen.blit(health_text,
                                     (col * self.TILE_SIZE + 20, row * self.TILE_SIZE + self.TILE_SIZE - 15))

                    # Draw unit owner
                    owner_text = font.render(str(unit_owner), True, (0, 0, 255))
                    self.screen.blit(owner_text,
                                     (col * self.TILE_SIZE + 5, row * self.TILE_SIZE + self.TILE_SIZE - 15))

        # Draw the player stats in the reserved column area
        stats_x = self.GRID_SIZE * self.TILE_SIZE + 10  # Start of the stats column
        y_offset = 10
        player_stats = {
            "Player 1 Stars": obs[121][0],
            "Player 2 Stars": obs[122][0]
        }
        for stat_name, stat_value in player_stats.items():
            stat_text = font.render(f"{stat_name}: {stat_value}", True, (0, 0, 0))
            self.screen.blit(stat_text, (stats_x, y_offset))
            y_offset += 30  # Increase y_offset for the next stat line

        pygame.display.flip()

    def get_adjacent_indices(self, index):
        row, col = divmod(index, 11)
        adjacent = []
        directions = [
            (-1, 0), (0, -1), (0, 1), (1, 0)  # Up, Left, Right, Down
        ]
        for dr, dc in directions:
            r, c = row + dr, col + dc
            if 0 <= r < 11 and 0 <= c < 11:
                adjacent.append(r * 11 + c)
        return adjacent

    def close(self):
        pygame.quit()
