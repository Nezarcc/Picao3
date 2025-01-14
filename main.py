import pygame
import random
from typing import List, Tuple, Set, Dict
from collections import deque
from dataclasses import dataclass
from enum import Enum
import numpy as np
from pygame import Surface, SRCALPHA
import math

class Colors(Enum):
    BLACK = (0, 0, 0)
    WHITE = (255, 255, 255)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    GRAY = (150, 150, 150)
    YELLOW = (255, 255, 0)
    BLUE = (0, 0, 255)

@dataclass(slots=True, frozen=True)
class Position:
    x: int
    y: int
    
    def __add__(self, other):
        return Position(self.x + other.x, self.y + other.y)
    
    def to_tuple(self):
        return (self.x, self.y)

class MainMenu:
    def __init__(self, screen):
        self.screen = screen
        screen_width, screen_height = self.screen.get_size()
        button_width, button_height = 200, 50
        self.font_large = pygame.font.Font(None, 100)
        self.font = pygame.font.Font(None, 36)
        self.buttons = {
            "Play": pygame.Rect((screen_width - button_width) // 2, screen_height // 2 - 50, button_width, button_height),
            "Quit": pygame.Rect((screen_width - button_width) // 2, screen_height // 2 + 50, button_width, button_height)
        }

    def draw(self):
        self.screen.fill(Colors.WHITE.value)
        title = self.font_large.render("Picão 3", True, Colors.BLACK.value)
        self.screen.blit(title, (self.screen.get_width() // 2 - title.get_width() // 2, 425))

        for text, rect in self.buttons.items():
            pygame.draw.rect(self.screen, Colors.GRAY.value, rect)
            label = self.font.render(text, True, Colors.BLACK.value)
            self.screen.blit(label, (rect.x + (rect.width - label.get_width()) // 2, rect.y + (rect.height - label.get_height()) // 2))
        
        pygame.display.flip()

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                if self.buttons["Play"].collidepoint(mouse_pos):
                    return "play"
                elif self.buttons["Quit"].collidepoint(mouse_pos):
                    return "quit"
        return "menu"

class PauseMenu:
    def __init__(self, screen, font):
        self.screen = screen
        self.font = font
        screen_width, screen_height = self.screen.get_size()
        button_width, button_height = 200, 50
        self.buttons = {
            "Continue": pygame.Rect((screen_width - button_width) // 2, screen_height // 2 - 100, button_width, button_height),
            "Main Menu": pygame.Rect((screen_width - button_width) // 2, screen_height // 2, button_width, button_height),
            "Quit": pygame.Rect((screen_width - button_width) // 2, screen_height // 2 + 100, button_width, button_height)
        }

    def draw(self):
        self.screen.fill(Colors.WHITE.value)
        title = self.font.render("Paused", True, Colors.BLACK.value)
        self.screen.blit(title, (self.screen.get_width() // 2 - title.get_width() // 2, 100))

        for text, rect in self.buttons.items():
            pygame.draw.rect(self.screen, Colors.GRAY.value, rect)
            label = self.font.render(text, True, Colors.BLACK.value)
            self.screen.blit(label, (rect.x + (rect.width - label.get_width()) // 2, rect.y + (rect.height - label.get_height()) // 2))
        
        pygame.display.flip()

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return "quit"
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                if self.buttons["Continue"].collidepoint(mouse_pos):
                    return "play"
                elif self.buttons["Main Menu"].collidepoint(mouse_pos):
                    return "menu"
                elif self.buttons["Quit"].collidepoint(mouse_pos):
                    return "quit"
        return "paused"

class MazeGame:
    DIRECTIONS = [Position(0, -1), Position(1, 0), Position(0, 1), Position(-1, 0)]
    GENERATION_DIRECTIONS = [Position(0, -2), Position(2, 0), Position(0, 2), Position(-2, 0)]
    
    def __init__(self, initial_size: int = 5, cell_size: int = 40):
        pygame.init()
        self.cell_size = cell_size
        self.level = 1
        self.score = 0
        self.smoothing_factor = 0.08
        self.zoom_factor = 1
        
        self.maze_size = max(3, initial_size + (1 - initial_size % 2))
        self.screen_size = 1200
        
        self.screen = pygame.display.set_mode((self.screen_size, self.screen_size))
        pygame.display.set_caption("Picão 3")
        self.font = pygame.font.Font(None, 36)
        
        self._init_surfaces()
        self.maze = np.full((self.maze_size, self.maze_size), '#', dtype=str)
        self.initialize_new_maze()
        self.running = True
        self.clock = pygame.time.Clock()
        
        self.state = "menu"
        self.main_menu = MainMenu(self.screen)
        self.pause_menu = PauseMenu(self.screen, self.font)
        self.blue_cube_pos = None

    def _init_surfaces(self):
        self.cell_surfaces = {}
        for color in Colors:
            surface = Surface((self.cell_size, self.cell_size), SRCALPHA)
            surface.fill(color.value)
            self.cell_surfaces[color] = surface
        
        self.hud_surface = Surface((200, 80), SRCALPHA)
        self.hud_surface.fill(Colors.GRAY.value)
        self.hud_surface.set_alpha(180)
        
        self.text_templates = {
            'level': self.font.render("Level: ", True, Colors.BLACK.value),
            'score': self.font.render("Score: ", True, Colors.BLACK.value),
            'size': self.font.render("Size: ", True, Colors.BLACK.value)
        }

    def draw_player_with_compass(self, x: int, y: int):
        # Calculate direction to goal
        dx = self.end_pos.x - self.player_pos.x
        dy = self.end_pos.y - self.player_pos.y
        angle = math.atan2(dy, dx)
        
        # Create player surface with compass
        player_surface = Surface((self.cell_size, self.cell_size), SRCALPHA)
        
        # Draw player
        pygame.draw.rect(player_surface, Colors.RED.value, 
                        (0, 0, self.cell_size, self.cell_size))
        
        # Draw direction arrow
        center = self.cell_size // 2
        arrow_length = self.cell_size // 2
        end_x = center + arrow_length * math.cos(angle)
        end_y = center + arrow_length * math.sin(angle)
        
        # Draw arrow
        pygame.draw.line(player_surface, Colors.YELLOW.value,
                        (center, center),
                        (end_x, end_y), 3)
        
        # Draw arrow head
        head_length = 10
        head_angle = 0.5  # ~30 degrees in radians
        
        angle1 = angle + math.pi + head_angle
        angle2 = angle + math.pi - head_angle
        
        head1_x = end_x + head_length * math.cos(angle1)
        head1_y = end_y + head_length * math.sin(angle1)
        head2_x = end_x + head_length * math.cos(angle2)
        head2_y = end_y + head_length * math.sin(angle2)
        
        pygame.draw.line(player_surface, Colors.YELLOW.value,
                        (end_x, end_y), (head1_x, head1_y), 3)
        pygame.draw.line(player_surface, Colors.YELLOW.value,
                        (end_x, end_y), (head2_x, head2_y), 3)
        
        self.screen.blit(player_surface, (x, y))

    def initialize_new_maze(self):
        self.maze.fill('#')
        self.set_random_start_end()
        self.player_pos = Position(self.start_pos.x, self.start_pos.y)
        self.camera_pos = Position(self.start_pos.x, self.start_pos.y)
        self.blue_cube_pos = Position(self.start_pos.x, self.start_pos.y)
        self.generate_maze()
        self.blue_cube_path = self.find_shortest_path(self.start_pos, self.end_pos)

    def set_random_start_end(self):
        corners = [(1, 1), (1, self.maze_size - 2),
                  (self.maze_size - 2, 1), (self.maze_size - 2, self.maze_size - 2)]
        start_idx, end_idx = random.sample(range(4), 2)
        self.start_pos = Position(*corners[start_idx])
        self.end_pos = Position(*corners[end_idx])

    @staticmethod
    @np.vectorize
    def _is_wall(cell):
        return cell == '#'

    def generate_maze(self):
        self.maze.fill('#')
        self.maze[self.start_pos.y, self.start_pos.x] = '.'
        self.maze[self.end_pos.y, self.end_pos.x] = '.'
        
        if not self._generate_paths(self.start_pos) or not self._verify_path():
            self.generate_maze()

    def _generate_paths(self, start: Position) -> bool:
        stack = [start]
        visited = {start.to_tuple()}
        max_iterations = self.maze_size * self.maze_size
        
        for _ in range(max_iterations):
            if not stack:
                break
                
            current = stack[-1]
            neighbors = []
            
            for direction in self.GENERATION_DIRECTIONS:
                next_pos = current + direction
                if (1 <= next_pos.x < self.maze_size - 1 and 
                    1 <= next_pos.y < self.maze_size - 1 and 
                    next_pos.to_tuple() not in visited):
                    
                    # Vectorized corridor check
                    corridor_positions = [(next_pos + d).to_tuple() for d in self.DIRECTIONS]
                    if sum(self.maze[y, x] == '.' for x, y in corridor_positions) <= 1:
                        neighbors.append(next_pos)
            
            if neighbors:
                next_pos = random.choice(neighbors)
                mid_x = current.x + (next_pos.x - current.x) // 2
                mid_y = current.y + (next_pos.y - current.y) // 2
                self.maze[mid_y, mid_x] = '.'
                self.maze[next_pos.y, next_pos.x] = '.'
                stack.append(next_pos)
                visited.add(next_pos.to_tuple())
            else:
                stack.pop()
                
        return len(visited) > 0

    def _verify_path(self) -> bool:
        queue = deque([(self.start_pos.y, self.start_pos.x)])
        visited = np.zeros((self.maze_size, self.maze_size), dtype=bool)
        visited[self.start_pos.y, self.start_pos.x] = True
        
        while queue:
            y, x = queue.popleft()
            if (y, x) == (self.end_pos.y, self.end_pos.x):
                return True
            
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if (0 <= ny < self.maze_size and 
                    0 <= nx < self.maze_size and 
                    self.maze[ny, nx] == '.' and 
                    not visited[ny, nx]):
                    queue.append((ny, nx))
                    visited[ny, nx] = True
        return False

    def find_shortest_path(self, start: Position, end: Position) -> List[Position]:
        queue = deque([start])
        visited = {start.to_tuple(): None}
        
        while queue:
            current = queue.popleft()
            if current == end:
                break
            
            for direction in self.DIRECTIONS:
                neighbor = current + direction
                if (0 <= neighbor.x < self.maze_size and
                    0 <= neighbor.y < self.maze_size and
                    self.maze[neighbor.y, neighbor.x] == '.' and
                    neighbor.to_tuple() not in visited):
                    queue.append(neighbor)
                    visited[neighbor.to_tuple()] = current
        
        # Reconstruct the path
        path = []
        current = end
        while current:
            path.append(current)
            current = visited[current.to_tuple()]
        path.reverse()
        return path

    def draw_maze(self):
        self.screen.fill(Colors.WHITE.value)
        
        delta_x = self.player_pos.x - self.camera_pos.x
        delta_y = self.player_pos.y - self.camera_pos.y
        self.camera_pos = Position(
            self.camera_pos.x + delta_x * self.smoothing_factor,
            self.camera_pos.y + delta_y * self.smoothing_factor
        )
        
        offset_x = self.screen_size // 2 - int(self.camera_pos.x * self.cell_size)
        offset_y = self.screen_size // 2 - int(self.camera_pos.y * self.cell_size)
        
        min_x = max(0, int((0 - offset_x) / self.cell_size))
        max_x = min(self.maze_size, int((self.screen_size - offset_x) / self.cell_size) + 1)
        min_y = max(0, int((0 - offset_y) / self.cell_size))
        max_y = min(self.maze_size, int((self.screen_size - offset_y) / self.cell_size) + 1)
        
        wall_positions = np.argwhere(self.maze[min_y:max_y, min_x:max_x] == '#')
        for y, x in wall_positions:
            cell_x = (x + min_x) * self.cell_size + offset_x
            cell_y = (y + min_y) * self.cell_size + offset_y
            self.screen.blit(self.cell_surfaces[Colors.BLACK], (cell_x, cell_y))
        
        goal_x = self.end_pos.x * self.cell_size + offset_x
        goal_y = self.end_pos.y * self.cell_size + offset_y
        self.screen.blit(self.cell_surfaces[Colors.GREEN], (goal_x, goal_y))
        
        player_x = self.player_pos.x * self.cell_size + offset_x
        player_y = self.player_pos.y * self.cell_size + offset_y
        self.draw_player_with_compass(player_x, player_y)
        
        blue_cube_x = self.blue_cube_pos.x * self.cell_size + offset_x
        blue_cube_y = self.blue_cube_pos.y * self.cell_size + offset_y
        self.screen.blit(self.cell_surfaces[Colors.BLUE], (blue_cube_x, blue_cube_y))
        
        self.draw_hud()
        pygame.display.flip()

    def draw_hud(self):
        self.screen.blit(self.hud_surface, (10, 10))
        
        # Use pre-rendered templates
        y = 15
        for key, template in self.text_templates.items():
            self.screen.blit(template, (20, y))
            value = getattr(self, key) if hasattr(self, key) else f"{self.maze_size}x{self.maze_size}"
            value_text = self.font.render(str(value), True, Colors.BLACK.value)
            self.screen.blit(value_text, (20 + template.get_width(), y))
            y += 30

    def handle_input(self):
        keys = pygame.key.get_pressed()
        dx = keys[pygame.K_d] - keys[pygame.K_a]
        dy = keys[pygame.K_s] - keys[pygame.K_w]
        
        if dx or dy:
            new_pos = Position(self.player_pos.x + dx, self.player_pos.y + dy)
            if (0 <= new_pos.x < self.maze_size and
                0 <= new_pos.y < self.maze_size and
                self.maze[new_pos.y, new_pos.x] == '.'):
                self.player_pos = new_pos
                
                if (new_pos.x, new_pos.y) == self.end_pos.to_tuple():
                    self.start_fade_to_white()
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                self.state = "paused"

    def start_fade_to_white(self):
        self.state = "fading_to_white"
        self.fade_alpha = 0

    def fade_to_white(self):
        self.fade_alpha += 5
        fade_surface = Surface((self.screen_size, self.screen_size))
        fade_surface.fill(Colors.WHITE.value)
        fade_surface.set_alpha(self.fade_alpha)
        self.screen.blit(fade_surface, (0, 0))
        pygame.display.flip()
        if self.fade_alpha >= 255:
            self.state = "transition_to_next_level"
            self.fade_alpha = 0

    def transition_to_next_level(self):
        pygame.time.wait(100)  # Delay before starting the new level
        self.level += 1
        self.score += self.maze_size * 10
        
        if self.level % 2 == 0:
            self.maze_size += 2
            self.maze = np.full((self.maze_size, self.maze_size), '#', dtype=str)
        
        self.initialize_new_maze()
        self.state = "play"

    def move_blue_cube(self):
        if self.blue_cube_path:
            self.blue_cube_pos = self.blue_cube_path.pop(0)

    def run(self):
        while self.running:
            if self.state == "menu":
                self.state = self.main_menu.handle_input()
                self.main_menu.draw()
            elif self.state == "play":
                self.handle_input()
                self.move_blue_cube()
                self.draw_maze()
                self.clock.tick(60)
            elif self.state == "paused":
                self.state = self.pause_menu.handle_input()
                self.pause_menu.draw()
            elif self.state == "fading_to_white":
                self.fade_to_white()
            elif self.state == "transition_to_next_level":
                self.transition_to_next_level()
            elif self.state == "quit":
                self.running = False
        
        pygame.quit()

if __name__ == "__main__":
    game = MazeGame()
    game.run()
