import pygame
import random
import numpy as np
import math
from enum import Enum
from dataclasses import dataclass
from collections import deque

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
    def __add__(self, other): return Position(self.x + other.x, self.y + other.y)
    def to_tuple(self): return (self.x, self.y)

class Menu:
    def __init__(self, screen, options, title="Menu", font_size=36, title_size=100):
        self.screen = screen
        self.font = pygame.font.Font(None, font_size)
        self.font_large = pygame.font.Font(None, title_size)
        screen_width, screen_height = screen.get_size()
        button_width, button_height = 200, 50
        self.buttons = {opt: pygame.Rect((screen_width - button_width) // 2, screen_height // 2 - (len(options) - 1) * button_height // 2 + i * button_height, button_width, button_height) for i, opt in enumerate(options)}
        self.title = title

    def draw(self):
        self.screen.fill(Colors.WHITE.value)
        title_text = self.font_large.render(self.title, True, Colors.BLACK.value)
        self.screen.blit(title_text, (self.screen.get_width() // 2 - title_text.get_width() // 2, 100))
        for text, rect in self.buttons.items():
            pygame.draw.rect(self.screen, Colors.GRAY.value, rect)
            label = self.font.render(text, True, Colors.BLACK.value)
            self.screen.blit(label, (rect.x + (rect.width - label.get_width()) // 2, rect.y + (rect.height - label.get_height()) // 2))
        pygame.display.flip()

    def handle_input(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT: return "quit"
            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_pos = pygame.mouse.get_pos()
                for text, rect in self.buttons.items():
                    if rect.collidepoint(mouse_pos): return text.lower()
        return "menu"

class MazeGame:
    DIRS = [Position(0, -1), Position(1, 0), Position(0, 1), Position(-1, 0)]
    GEN_DIRS = [Position(0, -2), Position(2, 0), Position(0, 2), Position(-2, 0)]

    def __init__(self, initial_size=5, cell_size=40, screen_size=1200):
        pygame.init()
        self.cell_size, self.screen_size = cell_size, screen_size
        self.level, self.score, self.smooth, self.zoom = 1, 0, 0.08, 1
        self.maze_size = max(3, initial_size + (1 - initial_size % 2))
        self.screen = pygame.display.set_mode((screen_size, screen_size))
        pygame.display.set_caption("Picão 3")
        self.font = pygame.font.Font(None, 36)
        self.cell_surfaces = {color: pygame.Surface((cell_size, cell_size), pygame.SRCALPHA).fill(color.value) for color in Colors}
        self.hud = pygame.Surface((200, 80), pygame.SRCALPHA); self.hud.fill(Colors.GRAY.value); self.hud.set_alpha(180)
        self.texts = {k: self.font.render(f"{k.capitalize()}: ", True, Colors.BLACK.value) for k in ["level", "score", "size"]}
        self.maze = np.full((self.maze_size, self.maze_size), '#', dtype=str)
        self.new_maze(); self.running, self.clock, self.state = True, pygame.time.Clock(), "menu"
        self.main_menu, self.pause_menu = Menu(self.screen, ["Play", "Quit"], "Picão 3"), Menu(self.screen, ["Continue", "Main Menu", "Quit"], "Paused")
        self.blue_cube_pos = None

    def new_maze(self):
        self.maze.fill('#'); self.start_end(); self.player = Position(self.start.x, self.start.y); self.cam = Position(self.start.x, self.start.y); self.blue_cube_pos = Position(self.start.x, self.start.y); self.gen_maze(); self.blue_path = self.find_path(self.start, self.end)

    def start_end(self):
        corners = [(1, 1), (1, self.maze_size - 2), (self.maze_size - 2, 1), (self.maze_size - 2, self.maze_size - 2)]
        self.start, self.end = map(Position, random.sample(corners, 2))

    def gen_maze(self):
        self.maze.fill('#'); self.maze[self.start.y, self.start.x] = '.'; self.maze[self.end.y, self.end.x] = '.'
        if not self._gen_paths(self.start) or not self._verify(): self.gen_maze()

    def _gen_paths(self, start):
        stack, visited, max_iters = [start], {start.to_tuple()}, self.maze_size * self.maze_size
        for _ in range(max_iters):
            if not stack: break
            curr, neighbors = stack[-1], []
            for d in self.GEN_DIRS:
                nxt = curr + d
                if 1 <= nxt.x < self.maze_size - 1 and 1 <= nxt.y < self.maze_size - 1 and nxt.to_tuple() not in visited:
                    if sum(self.maze[y, x] == '.' for x, y in [(nxt + dd).to_tuple() for dd in self.DIRS]) <= 1: neighbors.append(nxt)
            if neighbors:
                nxt = random.choice(neighbors); mid = Position(curr.x + (nxt.x - curr.x) // 2, curr.y + (nxt.y - curr.y) // 2)
                self.maze[mid.y, mid.x] = '.'; self.maze[nxt.y, nxt.x] = '.'; stack.append(nxt); visited.add(nxt.to_tuple())
            else: stack.pop()
        return len(visited) > 0

    def _verify(self):
        queue, visited = deque([(self.start.y, self.start.x)]), np.zeros((self.maze_size, self.maze_size), dtype=bool)
        visited[self.start.y, self.start.x] = True
        while queue:
            y, x = queue.popleft();
            if (y, x) == (self.end.y, self.end.x): return True
            for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < self.maze_size and 0 <= nx < self.maze_size and self.maze[ny, nx] == '.' and not visited[ny, nx]:
                    queue.append((ny, nx)); visited[ny, nx] = True
        return False

    def find_path(self, start, end):
        queue, visited = deque([start]), {start.to_tuple(): None}
        while queue:
            curr = queue.popleft();
            if curr == end: break
            for d in self.DIRS:
                nxt = curr + d
                if 0 <= nxt.x < self.maze_size and 0 <= nxt.y < self.maze_size and self.maze[nxt.y, nxt.x] == '.' and nxt.to_tuple() not in visited:
                    queue.append(nxt); visited[nxt.to_tuple()] = curr
        path, curr = [], end
        while curr: path.append(curr); curr = visited[curr.to_tuple()]
        return path[::-1]

    def draw_player(self, x, y):
        dx, dy = self.end.x - self.player.x, self.end.y - self.player.y; angle = math.atan2(dy, dx)
        p = pygame.Surface((self.cell_size, self.cell_size), pygame.SRCALPHA); pygame.draw.rect(p, Colors.RED.value, (0, 0, self.cell_size, self.cell_size))
        center, arrow_len = self.cell_size // 2, self.cell_size // 2; ex, ey = center + arrow_len * math.cos(angle), center + arrow_len * math.sin(angle)
        pygame.draw.line(p, Colors.YELLOW.value, (center, center), (ex, ey), 3)
        h_len, h_angle = 10, 0.5; a1, a2 = angle + math.pi + h_angle, angle + math.pi - h_angle
        h1x, h1y, h2x, h2y = ex + h_len * math.cos(a1), ey + h_len * math.sin(a1), ex + h_len * math.cos(a2), ey + h_len * math.sin(a2)
        pygame.draw.line(p, Colors.YELLOW.value, (ex, ey), (h1x, h1y), 3); pygame.draw.line(p, Colors.YELLOW.value, (ex, ey), (h2x, h2y), 3)
        self.screen.blit(p, (x, y))

    def draw(self):
        self.screen.fill(Colors.WHITE.value); dx, dy = self.player.x - self.cam.x, self.player.y - self.cam.y
        self.cam = Position(self.cam.x + dx * self.smooth, self.cam.y + dy * self.smooth); off_x, off_y = self.screen_size // 2 - int(self.cam.x * self.cell_size), self.screen_size // 2 - int(self.cam.y * self.cell_size)
        min_x, max_x = max(0, int((0 - off_x) / self.cell_size)), min(self.maze_size, int((self.screen_size - off_x) / self.cell_size) + 1)
        min_y, max_y = max(0, int((0 - off_y) / self.cell_size)), min(self.maze_size, int((self.screen_size - off_y) / self.cell_size) + 1)
        walls = np.argwhere(self.maze[min_y:max_y, min_x:max_x] == '#')
        for y, x in walls: self.screen.blit(self.cell_surfaces[Colors.BLACK], ((x + min_x) * self.cell_size + off_x, (y + min_y) * self.cell_size + off_y))
        self.screen.blit(self.cell_surfaces[Colors.GREEN], (self.end.x * self.cell_size + off_x, self.end.y * self.cell_size + off_y))
        self.draw_player(self.player.x * self.cell_size + off_x, self.player.y * self.cell_size + off_y)
        self.screen.blit(self.cell_surfaces[Colors.BLUE], (self.blue_cube_pos.x * self.cell_size + off_x, self.blue_cube_pos.y * self.cell_size + off_y))
        self.screen.blit(self.hud, (10, 10)); y_pos = 15
        for k, v in self.texts.items(): self.screen.blit(v, (20, y_pos)); self.screen.blit(self.font.render(str(getattr(self, k) if hasattr(self, k) else f"{self.maze_size}x{self.maze_size}"), True, Colors.BLACK.value), (20 + v.get_width(), y_pos)); y_pos += 30
        pygame.display.flip()

    def handle_input(self):
        keys = pygame.key.get_pressed(); dx, dy = keys[pygame.K_d] - keys[pygame.K_a], keys[pygame.K_s] - keys[pygame.K_w]
        if dx or dy:
            nxt = Position(self.player.x + dx, self.player.y + dy)
            if 0 <= nxt.x < self.maze_size and 0 <= nxt.y < self.maze_size and self.maze[nxt.y, nxt.x] == '.': self.player = nxt
            if (nxt.x, nxt.y) == self.end.to_tuple(): self.fade()
        for e in pygame.event.get():
            if e.type == pygame.QUIT: self.running = False
            if e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE: self.state = "paused"

    def fade(self): self.state, self.fade_alpha = "fade", 0

    def fade_to_white(self):
        self.fade_alpha += 5; s = pygame.Surface((self.screen_size, self.screen_size)); s.fill(Colors.WHITE.value); s.set_alpha(self.fade_alpha); self.screen.blit(s, (0, 0)); pygame.display.flip()
        if self.fade_alpha >= 255: self.state, self.fade_alpha = "next", 0

    def next_level(self):
        pygame.time.wait(100); self.level += 1; self.score += self.maze_size * 10
        if self.level % 2 == 0: self.maze_size += 2; self.maze = np.full((self.maze_size, self.maze_size), '#', dtype=str)
        self.new_maze(); self.state = "play"

    def move_blue(self):
        if self.blue_path: self.blue_cube_pos = self.blue_path.pop(0)

    def run(self):
        while self.running:
            if self.state == "menu": self.state = self.main_menu.handle_input(); self.main_menu.draw()
            elif self.state == "play": self.handle_input(); self.move_blue(); self.draw(); self.clock.tick(60)
            elif self.state == "paused": self.state = self.pause_menu.handle_input(); self.pause_menu.draw()
            elif self.state == "fade": self.fade_to_white()
            elif self.state == "next": self.next_level()
            elif self.state == "quit": self.running = False
        pygame.quit()

if __name__ == "__main__": MazeGame().run()
