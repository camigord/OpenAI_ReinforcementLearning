import numpy as np
import gym
import cv2

class GameScreen:
    def __init__(self, conf):
        self.screens = np.zeros([conf['pool_frame_size'],conf['screen_height'],conf['screen_width']],dtype=np.float32)
        
    def add(self, screen):
        self.screens[:-1] = self.screens[1:]
        self.screens[-1] = screen
        
    def get(self):        
        return np.amax(np.transpose(self.screens,(1,2,0)),axis=2)
    
class GymEnvironment():
    def __init__(self,name,conf):
        self.env = gym.make(name)            # Initialize Gym environment
        self.game_screen = GameScreen(conf)
        self.screen_width = conf['screen_width']         
        self.screen_height = conf['screen_height']
        self.random_start = conf['random_start']
        self.action_repeat = conf['action_repeat']
        self.pool_frame_size = conf['pool_frame_size']
        self.display = False
        
    def set_display(self,value=False):
        self.display = value
        
    def new_game(self):
        self._observation = self.env.reset()
        
        for i in range(self.pool_frame_size):
            self.game_screen.add(self.observation)
            
        self.render()        
        return self.screen
    
    def new_random_game(self):  # Starts a random new game doing 
        _ = self.new_game()
        terminal = False
        for _ in xrange(random.randint(0,self.random_start-1)):
            self._observation, reward, terminal, _ = self.env.step(0)
            self.game_screen.add(self.observation)
        
        self.render()
        return self.screen, 0, terminal
        
    def execute_action(self,action,is_training=True):
        # This function execute the selected action for 'action_repeat' number of times and returns the cumulative reward
        # and final state
        cum_reward = 0
        start_lives = self.num_lives
        
        for _ in xrange(self.action_repeat):
            self._observation, reward, terminal, _ = self.env.step(action)
            self.game_screen.add(self.observation)
            cum_reward += reward
            
            if is_training and start_lives > self.num_lives:
                terminal = True
                
            if terminal:
                break
                
        self.render()
        
        return self.screen, cum_reward, terminal
    
    @property
    def screen(self):
        return self.game_screen.get()
        
    @property
    def action_size(self):
        return self.env.action_space.n        # Number of available actions

    @property
    def num_lives(self):
        return self.env.ale.lives()
    
    @property
    def observation(self):     # Method to resize the screen provided by gym to the desired values
        return cv2.resize(cv2.cvtColor(self._observation,cv2.COLOR_RGB2GRAY)/255.,(self.screen_width,self.screen_height))
    
    def render(self):    # Renders the environment only if display == True
        if self.display:
            self.env.render()  