import pygame
import numpy as np
import random
import gymnasium as gym
from gymnasium import spaces


class GridMazeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 2}

    def __init__(self, grid_size=5, goal_pos=[(4, 4)], mines=[(1, 1), (2, 3)], cell_size=100, render_mode="human",rnd=True):

        super().__init__()
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.width = self.grid_size * self.cell_size
        self.height = self.grid_size * self.cell_size
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(4)  # 0: Up, 1: Down, 2: Left, 3: Right
        self.observation_space = spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32)
        self.rnd = rnd
        if rnd:
            self.goal_pos = [ (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)) ]
            self.mines = set()
            for _ in range(len(mines)):
                while True:
                    mine = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
                    if mine not in self.goal_pos:
                        self.mines.add(mine)
                        break
            while True: 
                pos = np.array([random.randint(0, self.grid_size-1),
                                random.randint(0, self.grid_size-1)])
                if tuple(pos) not in self.goal_pos and tuple(pos) not in self.mines:
                    self.agent_pos = pos
                    break
        
        else:
            self.goal_pos = goal_pos
            self.mines = set(mines)  
            self.agent_pos = np.array([0, 0])  # will be set in reset()


        # PyGame setup
        self.screen = None
        self.clock = None
        self.running = False

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.rnd:
            while True: 
                pos = np.array([random.randint(0, self.grid_size-1),
                                random.randint(0, self.grid_size-1)])
                if tuple(pos) not in self.goal_pos and tuple(pos) not in self.mines:
                    self.agent_pos = pos
                    break
        else:
            self.agent_pos = np.array([0, 0])
        return self.agent_pos, {} #(observation, info)

    def step(self, action):
        # Define perpendicular actions for each action
        perpendicular_actions = {
            0: [2, 3], 
            1: [2, 3], 
            2: [0, 1],  
            3: [0, 1]   
        }
        
        prob = random.random()
        if prob < 0.7:  
            actual_action = action
        else:  
            actual_action = random.choice(perpendicular_actions[action])
            
        # Move the agent based on the actual action
        x, y = int(self.agent_pos[0]), int(self.agent_pos[1])
        if actual_action == 0:  
            y = min(y + 1, self.grid_size - 1)
        elif actual_action == 1: 
            y = max(y - 1, 0)
        elif actual_action == 2:  
            x = max(x - 1, 0)
        elif actual_action == 3: 
            x = min(x + 1, self.grid_size - 1)

        self.agent_pos = np.array([x, y])

        # Check for goal
        if tuple(self.agent_pos) in self.goal_pos:
            reward = 10
            done = True

        # Check for mine
        elif tuple(self.agent_pos) in self.mines:
            reward = -10
            done = True
        else:
            reward = -1
            done = False

        return self.agent_pos, reward, done, False, {} #(observation, reward, terminated, truncated [due to time limit], info)


    def render(self,action=None):
        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                self.screen = pygame.display.set_mode((self.width, self.height))
                pygame.display.set_caption("Grid Maze")
            else: 
                self.screen = pygame.Surface((self.width, self.height))
            self.clock = pygame.time.Clock()
            self.running = True

        # Fill background
        self.screen.fill((255, 255, 255))  # white background

        # Draw grid lines
        for x in range(0, self.width, self.cell_size):
            pygame.draw.line(self.screen, (0, 0, 0), (x, 0), (x, self.height))
        for y in range(0, self.height, self.cell_size):
            pygame.draw.line(self.screen, (0, 0, 0), (0, y), (self.width, y))
        # Helper to convert (x,y) to screen rect (top-left coords)
        def to_screen_rect(pos):
            sx = pos[0] * self.cell_size
            # flip y because screen y=0 is top
            sy = (self.grid_size - 1 - pos[1]) * self.cell_size
            return pygame.Rect(sx, sy, self.cell_size, self.cell_size)
        # Draw goal
        for goal in self.goal_pos:
            goal_rect = to_screen_rect(goal)
            pygame.draw.rect(self.screen, (0, 255, 0), goal_rect)  # green

        # Draw mines
        for mine in self.mines:
            mine_rect = to_screen_rect(mine)
            pygame.draw.rect(self.screen, (255, 0, 0), mine_rect)  # red

        # Draw agent as circle
        ax, ay = int(self.agent_pos[0]), int(self.agent_pos[1])
        center_x = ax * self.cell_size + self.cell_size // 2
        center_y = (self.grid_size - 1 - ay) * self.cell_size + self.cell_size // 2
        radius = self.cell_size // 3
        pygame.draw.circle(self.screen, (0, 0, 255), (center_x, center_y), radius)  # blue
      
        # Handle render modes
        if self.render_mode == "human":
            pygame.display.get_surface().blit(self.screen, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            # pygame.time.delay(100)
        elif self.render_mode == "rgb_array":
            # return RGB array for video recording
            return np.transpose(pygame.surfarray.array3d(self.screen), (1, 0, 2))


    def get_states(self):
        """Get all possible states in the environment"""
        states = []
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                #if (i, j) not in self.mines:  # Exclude mine positions
                states.append((i, j))
        return states

    def get_actions(self, state):
        """Get all possible actions"""
        return list(range(4))  # 0: Up, 1: Down, 2: Left, 3: Right

    def get_transitions(self, state, action):
        """Get transition probabilities and next states for a given state-action pair"""
        transitions = {}
        state = list(state)

        # Define perpendicular actions for each action
        perpendicular_actions = {
            0: [2, 3],  
            1: [2, 3], 
            2: [0, 1],  
            3: [0, 1]   
        }

        # Helper function to get next state given an action
        def get_next_state(current_state, act):
            x, y = current_state.copy()
            if act == 0:  # Up
                y = min(y + 1, self.grid_size - 1)
            elif act == 1:  # Down
                y = max(y - 1, 0)
            elif act == 2:  # Left
                x = max(x - 1, 0)
            elif act == 3:  # Right
                x = min(x + 1, self.grid_size - 1)
            return (x, y)

        # Main intended direction (70% probability)
        intended_next_state = get_next_state(state, action)
        transitions[intended_next_state] = transitions.get(intended_next_state, 0) + 0.7

        # Perpendicular directions (15% probability each)
        for perp_action in perpendicular_actions[action]:
            perp_next_state = get_next_state(state, perp_action)
            transitions[perp_next_state] = transitions.get(perp_next_state, 0) + 0.15


        return transitions

    def get_reward(self, state, action, next_state):
        """Get reward for a state-action-next_state transition"""
        if next_state in self.goal_pos:
            return 10
        elif next_state in self.mines:
            return -10
        return -1

    def is_terminal(self, state):
        """Check if a state is terminal"""
        return state in self.goal_pos or state in self.mines

    def close(self):
        if self.screen is not None:
            pygame.quit()
            self.screen = None
            self.clock = None
            self.running = False