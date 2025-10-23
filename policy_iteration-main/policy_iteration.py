from gymnasium.wrappers import RecordVideo
import random
from grid_maze import GridMazeEnv




class Policy:
    def __init__(self):
        self.policy = {}

    def select_action(self, st, acts):
        return self.policy.get(st, random.choice(acts))

    def update(self, st, act):
        self.policy[st] = act

class ValueFunction:
    def __init__(self):
        self._store = {}

    def get_value(self, st):
        return self._store.get(st, 0.0)

    def add(self, st, val):
        self._store[st] = val

class PolicyIteration:
    def __init__(self, environment):
        self.environment = environment
        self.policy = Policy()
        self.discount = 0.8
        for state in self.environment.get_states():
            if not self.environment.is_terminal(state):
                self.policy.update(state, random.choice(self.environment.get_actions(state)))
                
    def print_policy(self):
        grid_n = self.environment.grid_size
        display_grid = [['' for _ in range(grid_n)] for _ in range(grid_n)]
        action_map = {0: '↑', 1: '↓', 2: '←', 3: '→'}
        
        for st in self.environment.get_states():
            x, y = st
            if self.environment.is_terminal(st):
                display_grid[y][x] = ' G ' if st in self.environment.goal_pos else ' M '
            else:
                act = self.policy.select_action(st, self.environment.get_actions(st))
                display_grid[y][x] = f' {action_map.get(act, "?")} '
        
        # Column header
        header = '    ' + ' '.join(f'{i:^3}' for i in range(grid_n))
        print(header)
        print('   ' + '----' * grid_n)
        for row_idx, row in enumerate(display_grid[::-1]):
            print(f'{grid_n-1-row_idx:>2} |' + ' '.join(f'{cell:^3}' for cell in row))
        print()

    def policy_evaluation(self, val_func, tolerance=0.0001):
        
        while True:
            max_delta = 0
            for st in self.environment.get_states():
                if self.environment.is_terminal(st):
                    continue
                
                old_val = val_func.get_value(st)
                chosen_act = self.policy.select_action(st, self.environment.get_actions(st))
                
                new_val = 0
                for next_st, prob in self.environment.get_transitions(st, chosen_act).items():
                    r = self.environment.get_reward(st, chosen_act, next_st)
                    new_val += prob * (r + self.discount * val_func.get_value(next_st))
                
                val_func.add(st, new_val)
                max_delta = max(max_delta, abs(old_val - new_val))
            
            if max_delta < tolerance:
                break
        
        return val_func

    def policy_improvement(self, val_func):
        stable_flag = True
        
        for st in self.environment.get_states():
            if self.environment.is_terminal(st):
                continue
                
            prev_act = self.policy.select_action(st, self.environment.get_actions(st))
            
            best_val = float('-inf')
            best_act = None
            
            for act in self.environment.get_actions(st):
                act_val = 0
                for next_st, prob in self.environment.get_transitions(st, act).items():
                    r = self.environment.get_reward(st, act, next_st)
                    act_val += prob * (r + self.discount * val_func.get_value(next_st))
                
                if act_val > best_val:
                    best_val = act_val
                    best_act = act
            
            self.policy.update(st, best_act)
            
            if best_act != prev_act:
                stable_flag = False
        
        return stable_flag

    def solve(self, max_iters=20000, tolerance=0.0001):
        val_func = ValueFunction()
        for s in self.environment.get_states():
            val_func.add(s, 0.0)
        for iteration in range(max_iters):
            # Policy Evaluation
            val_func = self.policy_evaluation(val_func, tolerance)
            
            # Policy Improvement
            converged = self.policy_improvement(val_func)
            
            if converged:
                print(f"\nPolicy converged after {iteration+1} iterations.")
                print(f"Discount: {self.discount}, States: {len(self.environment.get_states())}")
                print("Optimal policy (grid):")
                self.print_policy()
                return self.policy
        
        print(f"\nMaximum iterations reached ({max_iters}). Returning current policy.")
        print("Current policy (grid):")
        self.print_policy()
        return self.policy

def run_policy_iteration():
    # Create environment
    env_instance = GridMazeEnv(grid_size=5, goal_pos=[(4,4)], mines=[(1,3), (3,2)], 
                      rnd=True, render_mode="rgb_array")
    
    # Test the optimal policy
    recorder = RecordVideo(env_instance, video_folder="videos", episode_trigger=lambda e: True, fps=2)
    
    observation, _ = recorder.reset()
    finished = False
    cum_reward = 0
    step_no = 0
    # Create and run policy iteration
    pi_agent = PolicyIteration(env_instance)
    optimal_policy = pi_agent.solve()
    
    action_names = {0: "Up", 1: "Down", 2: "Left", 3: "Right"}
    print("Executing policy...")
    print(f"{'Step':>4} | {'State':>9} | {'Action':>6} | {'Reward':>7} | {'Cum.':>7}")
    print("-" *  50)
    
    while not finished:
        # normalize observation to plain Python ints for clean printing & dict keys
        state = tuple(int(x) for x in observation)
        action_choice = optimal_policy.select_action(state, env_instance.get_actions(state))
        observation, reward, finished, _, _ = recorder.step(action_choice)
        step_no += 1
        taken = action_names.get(action_choice, "?")
        cum_reward += reward
        print(f"{step_no:4d} | {state!s:>9} | {taken:>6} | {reward:7.2f} | {cum_reward:7.2f}")
        recorder.render()
    
    print("\nEpisode finished.")
    print(f"Steps: {step_no}, Total reward: {cum_reward:.2f}")
    recorder.close()

if __name__ == "__main__":
    run_policy_iteration()
