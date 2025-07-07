import numpy as np
from matplotlib import pyplot as plt
from matplotlib import animation
from matplotlib.animation import FuncAnimation

# Environmen hyperparam:
ARENA_SIDE_LENGTH = 300
STEPS             = 1000
MAX_SPEED         = 5e-1
BASE_SPEED        = 8e-2

# Swarm hyperparameters:
NUMBER_OF_ROBOTS  = 40
NEIGHTBORS_SPACE = np.inf
SAFE_SPACE = 7

# Make the environment toroidal 
def wrap(z):    
    return z % ARENA_SIDE_LENGTH

class agent:
    def __init__(self, id, x, y, vx, vy):
        self.id = id
        self.x = x
        self.y = y
        self.vx = vx
        self.vy = vy
        self.vx0 = vx
        self.vy0 = vy
        self.yaw = np.arctan2(vy,vx)

        self.N = None 
        if id == 0: self.id_follow = 0
        else:       self.id_follow = id - 1

    def position(self):
        return np.array([self.x, self.y])

    def set_position(self, new_x, new_y, avoid = False):
        if avoid:
            x,y = self.avoid_collision(new_x,new_y)
        else:
            x,y = [new_x, new_y]
        if np.linalg.norm([self.vx, self.vy]) > BASE_SPEED/5:
            self.yaw = np.arctan2(self.vy,self.vx)

        self.x = x
        self.y = y
      
    def neightbors(self):
        return self.N

    def set_neightbors(self,neightbors):
        self.N = neightbors

    def forward(self):
        x_next = wrap(self.x + self.vx0)
        y_next = wrap(self.y + self.vy0)
        self.set_position(x_next, y_next)
        return x_next, y_next
    
    def stop(self):
        self.vx = self.x
        self.vy = self.y
        self.set_position(self.x, self.y)
        return self.x, self.y

    def avoid_collision(self,x,y):
        new_x = np.copy(x)
        new_y = np.copy(y)

        if new_x != self.x and new_y != self.y:
            for n in self.N:
                distance = np.linalg.norm(self.position() - n.position())
                if distance < SAFE_SPACE:
                    # Calculate the adjustment vector
                    adjustment = self.position() - n.position()
                    adjustment /= distance  # Normalize to unit vector
                    adjustment *= (SAFE_SPACE - distance) / 2  # Scale by the amount to adjust
                    new_x += adjustment[0]
                    new_y += adjustment[1]
        
        return new_x, new_y
    
    def lider(self, ):
        global mouse_x, mouse_y

        [x,y] = self.position()

        if mouse_y == None or mouse_x == None:
            point=[ARENA_SIDE_LENGTH/2,ARENA_SIDE_LENGTH/2]
        else:
            point=[mouse_x,mouse_y]

        delta_x = point[0] - x
        delta_y = point[1] - y
        
        x_next = wrap(x + BASE_SPEED*delta_x)
        y_next = wrap(y + BASE_SPEED*delta_y)
        
        self.vx = BASE_SPEED*delta_x
        self.vy = BASE_SPEED*delta_y

        self.set_position(x_next,y_next)
        return x_next, y_next
    
    def follower(self):
        global d, phi

        # Self-position:
        [x,y] = self.position()

        # Agent to follow position:
        [n_x, n_y] = self.N[0].position()

        # Compute offset:
        angle = phi + self.id*(2*np.pi/(NUMBER_OF_ROBOTS-1))
        offset = [d*np.cos(angle), d*np.sin(angle)]

        # Control law:
        delta_x = (n_x - offset[0]) - x
        delta_y = (n_y - offset[1]) - y
        
        x_next = wrap(x + BASE_SPEED*delta_x)
        y_next = wrap(y + BASE_SPEED*delta_y)
        
        self.vx = BASE_SPEED*delta_x
        self.vy = BASE_SPEED*delta_y

        self.set_position(x_next,y_next)
        return x_next, y_next
    
    def chain(self):
        global d, phi

        # Self-position:
        [x,y] = self.position()

        # Find agent to follow:
        [n_x, n_y] = [None, None]
        for n in self.N:
            if n.id == self.id_follow:
                [n_x, n_y] = n.position()

        # Compute offset:
        angle = self.N[self.id_follow].yaw
        offset = [d*np.cos(angle), d*np.sin(angle)]

        # Control law:
        delta_x = (n_x - offset[0]) - x
        delta_y = (n_y - offset[1]) - y
        
        x_next = wrap(x + BASE_SPEED*delta_x)
        y_next = wrap(y + BASE_SPEED*delta_y)
        
        self.vx = BASE_SPEED*delta_x
        self.vy = BASE_SPEED*delta_y

        self.set_position(x_next,y_next)
        return x_next, y_next
    

    def cluster(self,state):
        # Difference position neightbors-agent 
        delta_x = 0
        delta_y = 0

        [x,y] = self.position()
        
        # Control law:
        for n in self.neightbors():
            [n_x, n_y] = state[n.id]
            delta_x += n_x - x
            delta_y += n_y - y
        
        # Norm:
        delta_x = delta_x/np.linalg.norm([delta_x,delta_y])
        delta_y = delta_y/np.linalg.norm([delta_x,delta_y])

        x_next = wrap(x + MAX_SPEED*delta_x)
        y_next = wrap(y + MAX_SPEED*delta_y)
        
        self.vx = MAX_SPEED*delta_x
        self.vy = MAX_SPEED*delta_y

        self.set_position(x_next,y_next, avoid = True)
        return x_next, y_next
    
    
    
class SwarmNetwork():

    def __init__(self):
        # Set random intial point:
        x_0 = np.random.uniform(low=0, high=ARENA_SIDE_LENGTH, size=(NUMBER_OF_ROBOTS,))
        y_0 = np.random.uniform(low=0, high=ARENA_SIDE_LENGTH, size=(NUMBER_OF_ROBOTS,))

        # Velocities random:
        vx = np.random.uniform(low=-MAX_SPEED, high=MAX_SPEED, size=(NUMBER_OF_ROBOTS,))
        vy = np.random.uniform(low=-MAX_SPEED, high=MAX_SPEED, size=(NUMBER_OF_ROBOTS,))

        # Agents:
        self.index = np.arange(NUMBER_OF_ROBOTS)
        self.agents = [agent(i,x_0[i],y_0[i],vx[i],vy[i]) for i in range(NUMBER_OF_ROBOTS)] # List of agents (own self-position)

        # Adjacency and Laplacian matrix:
        self.Adj = np.zeros((NUMBER_OF_ROBOTS,NUMBER_OF_ROBOTS))
        self.L = np.zeros((NUMBER_OF_ROBOTS,NUMBER_OF_ROBOTS))

        # Set initial topology:
        self.update_Topology()


    def state(self):
        return np.array([agent.position() for agent in self.agents])

    def update_position(self,delta_x,delta_y):
        for i,agent in enumerate(self.agents):
            agent.set_position(delta_x[i],delta_y[i])


    def one_step(self, mode = "random"):
        global d, phi

        x = []
        y = []
        for agent in self.agents:
            if mode == "shape":
                if agent.id == 0:
                    _x,_y = agent.lider()
                else:
                    _x,_y = agent.follower()
            elif mode == "chain":
                if agent.id == 0:
                    _x,_y = agent.lider()
                else:
                    _x,_y = agent.chain()
            elif mode == "random":
                _x,_y = agent.forward()
            elif mode == "cluster":
                _x,_y = agent.cluster(self.state())
            else:   # Do nothing
                _x,_y = agent.stop()

            x.append(_x)
            y.append(_y)

        # Update all agent position at once to avoid troubles in the algorithm 
        # Each agent has made its decision individually
        #self.update_position(x,y)
        self.update_Topology()

    def update_Topology(self):
        
        state = self.state()
        neightbors = [] # List of list of neightbors
        # For every agent in the swarm
        for agent in self.agents:

            # Check distance to every other agent
            dist_neighbors = np.linalg.norm(agent.position() - state,axis=1)
            # Select closest agents:
            sort_id = self.index[np.argsort(dist_neighbors)[1:]]
            neightbors_id = []
            for idx in sort_id:
                if dist_neighbors[idx] < NEIGHTBORS_SPACE:
                    neightbors_id.append(idx)

            neightbors.append(neightbors_id)

        # Save list of agents as every agent neightbors:
        for i,agent in enumerate(self.agents):
            temp_neightbor = []
            for other_agent in self.agents:
                if other_agent.id in neightbors[i]: 
                    temp_neightbor.append(other_agent)

            # Update agent's neightbour:
            agent.set_neightbors(temp_neightbor)

        #TODO: Check unconected subgraph
            

            
        # Double neightbour correlation:    To be neightbors, 2 agents must be neightbors respectively
        '''for agent in self.agents:
            c_neight = agent.neightbors()

            for j,n in enumerate(c_neight):
                if agent.id not in self.agents[n].neightbors():
                    c_neight[j] = agent.id    # Self-reference, agent connected with itself
                else:
                    self.Adj[agent.id, n] = 1
                    self.Adj[n, agent.id] = 1 
            
            # Update agent's neightbor:
            agent.set_neightbors(c_neight)'''

d = 15
phi = 0

phi_increment_value = 0  # Variable to hold the current increment value for phi
left_button_pressed = False  # Flag to track if the left mouse button is pressed
right_button_pressed = False  # Flag to track if the right mouse button is pressed

mouse_x = 0
mouse_y = 0

def toggle_mode(event):
    global mode, previous_mode, d, phi, deltaY, deltaX

    if event.key == 'up':
        mode = "random"
    elif event.key == 'down':
        mode = "cluster"
    elif event.key == 'right':
        mode = "shape"
        d = 40
    elif event.key == 'left':
        mode = "chain"
        d = 3
    elif event.key == ' ':
        if mode == "stop":
            mode = previous_mode
        else:
            previous_mode = mode
            mode = "stop"

def on_scroll(event):
    global d
    if event.button == 'up':
        d += 1
    elif event.button == 'down':
        d -= 1
    print('d: ',d)
def on_click(event):
    global phi_increment_value, left_button_pressed, right_button_pressed
    if event.button == 3:  # Right mouse button
        right_button_pressed = True
        phi_increment_value = np.pi / 40  # Set the increment value for phi
    elif event.button == 1:  # Left mouse button
        left_button_pressed = True
        phi_increment_value = np.pi / 40  # Set the decrement value for phi

def on_release(event):
    global phi_increment_value, left_button_pressed, right_button_pressed
    if event.button == 3:  # Right mouse button
        right_button_pressed = False
        phi_increment_value = 0  # Reset the increment value for phi
    elif event.button == 1:  # Left mouse button
        left_button_pressed = False
        phi_increment_value = 0  # Reset the decrement value for phi

# Función para rastrear la posición del ratón
def on_move(event):
    global mouse_x, mouse_y
    if event.inaxes == ax:
        mouse_x, mouse_y = event.xdata, event.ydata
    else:
        mouse_x, mouse_y = [None, None]

    
    

################# PLOT ########################

# Set up the output (1024 x 768):
fig = plt.figure(figsize=(10, 10), dpi=100)
ax = plt.axes(xlim=(0, ARENA_SIDE_LENGTH), ylim=(0, ARENA_SIDE_LENGTH))
ax_map = plt.axes([0.05, 0.05, 0.9, 0.3])  # Adjust position for map
points, = ax.plot([], [], 'bo', lw=0, )
#map_plot = ax_map.imshow(map_image, cmap='binary')
ax_map.axis('off')

# Create swarm:
net = SwarmNetwork()
mode = "stop"
previous_mode = "random"


def init():
    points.set_data([], [])
    return points,

def animate(i):
    global phi
    if left_button_pressed:
        phi -= phi_increment_value  # Decrement phi if the left mouse button is pressed
    elif right_button_pressed:
        phi += phi_increment_value  # Increment phi if the right mouse button is pressed


    net.one_step(mode)
    
    # Get points
    p = net.state()
    x = p[:, 0]
    y = p[:, 1]

    points.set_data(x, y)
    
    print('Step ', i + 1, '/', STEPS, end='\r')

    return points,

# Attach events to the figure
fig.canvas.mpl_connect('scroll_event', on_scroll)
fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('button_release_event', on_release)
fig.canvas.mpl_connect('key_press_event', toggle_mode)
fig.canvas.mpl_connect('motion_notify_event', on_move)
anim = FuncAnimation(fig, animate, init_func=init,
                               frames=STEPS, interval=1, blit=True)

videowriter = animation.FFMpegWriter(fps=60)
#anim.save("..\output.mp4", writer=videowriter)
plt.show()