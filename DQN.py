from os import system
import random
import time
import gym
from keras.engine.saving import load_model
import matplotlib.pyplot as plt
import numpy as np
import keras
from collections import deque
from keras.models import Sequential
from keras.layers import Input
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.image as mpimg
from IPython import display
import os
import QTable

#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = ""
MODELS_FOLDER = 'experiments/DQN_Agent/models/'

class DQN_Agent():
    EPS_START = 1
    EPS_MIN = 0.1
    EPS_DECAY = 0.995
    def __init__(self, env, ID, lr=0.001, activation_fn='linear', loss_fn='mse', filename=None, use_target_network=False, state_size=None):
        self.env = env
        self.ID = ID
        self.lr = lr
        self.activation_fn = activation_fn
        self.loss_fn = loss_fn
        self.epsilon = self.EPS_START
        self.use_target_network = use_target_network
        self.memory = deque(maxlen=2000)        
        if state_size:
            self.state_size = state_size
        elif env.observation_space.shape == ():
            self.state_size = env.observation_space.n
        else:
            self.state_size = env.observation_space.shape
        if filename:
            from keras.models import load_model
            self.agent = load_model(filename)
        else:
            use_bias = True
            activation = 'relu'
            self.agent = Sequential()
            if state_size:
                self.agent.add(Dense(25, input_shape=(state_size, ), use_bias=use_bias, activation=activation))
            elif env.observation_space.shape == ():
                self.agent.add(Dense(25, input_shape=(1, ), activation=activation))
            else:
                self.agent.add(Dense(25, input_shape=self.state_size, activation=activation))
            self.agent.add(Dense(25, activation='relu'))
            self.agent.add(Dense(env.action_space.n, activation=activation_fn))        
            #self.agent.compile(loss=loss_fn, optimizer=keras.optimizers.SGD(learning_rate=lr), metrics=['accuracy'])
            self.agent.compile(loss=loss_fn, optimizer=keras.optimizers.Adam(self.lr))
            
            if self.use_target_network:
                self.target = keras.models.clone_model(self.agent)
    def train(self, gamma=0.99, epochs=1000, batchSize = 50, file=None):
        results = []
        self.epsilon = self.EPS_START
        for e in range(epochs):
            done = False
            state = self.env.reset()
            if type(self.state_size) != int:
                state = np.reshape(state, [1, self.state_size])
            r = 0
            for i in range(5000):                
                action = self.act(state)
                new_state, reward, done, _ = self.env.step(action)
                if type(self.state_size) != int:
                    new_state = np.reshape(new_state, [1, self.state_size])
                    reward = reward if not done else -10
                self.rember([state, action, reward, new_state, done])
                state = new_state
                if type(self.state_size) == int:
                    r += reward
                if done:
                    if type(self.state_size) == int:
                        results.append(r)
                    else:
                        results.append(i)
                    if e % round(epochs * 0.1) == 0:
                        print('episode: {}/{}, score: {}, epsilon: {}, reward: {}'.format(e, epochs, i, self.epsilon, r))
                    break
                
            if self.epsilon > self.EPS_MIN:
                self.epsilon *= self.EPS_DECAY
            self.replay(gamma, batchSize)
            if self.use_target_network and e % 50 == 0 and e != 0:
                self.target = keras.models.clone_model(self.agent)

        self.save()
        return results
    def act(self, state, train=True):
        if not train:
            if type(state) is not list:
                return np.argmax(self.agent.predict(np.array([state])))
            return np.argmax(self.agent.predict(state))

        if random.uniform(0,1) > self.epsilon:
            if type(state) is not list:
                return np.argmax(self.agent.predict(np.array([state])))
            return np.argmax(self.agent.predict(state))
        else:
            return env.action_space.sample()
    def rember(self, s):
        self.memory.append(s)
    def replay(self, gamma=0.99, batch_size=50):
        if len(self.memory) < batch_size:
            return
        data = random.sample(self.memory, batch_size)
        for s, a, r, s_new, done in data:
            target = r
            if not done:
                if self.use_target_network:
                    if type(s_new) is not list:
                        prediction = self.target.predict(np.array([s_new]))
                    else:
                        prediction = self.target.predict(s_new)
                else:
                    if type(s_new) is not list:
                        prediction = self.agent.predict(np.array([s_new]))
                    else:
                        prediction = self.agent.predict(s_new)
                target = r + gamma * np.amax(prediction[0])
            if type(s_new) is not list:
                T_s = self.agent.predict(np.array([s]))
            else:
                T_s = self.agent.predict(s)
            T_s[0][a] = target
            if type(s_new) is not list:
                self.agent.fit(np.array([s]), T_s, epochs=1, batch_size=1, verbose=0)
            else:
                self.agent.fit(s, T_s, epochs=1, batch_size=1, verbose=0)

    def save(self, filename=None):
        if filename == None:
            filename = MODELS_FOLDER + str(self.ID)
        self.agent.save(filename)

    def test(self, epochs=10, render=True):
        max_score = 0
        for e in range(epochs):
            state = np.array([self.env.reset()])
            done = False
            score = 0
            for i in range(300):
                if type(state) is not list:
                    action = np.argmax(self.agent.predict(np.array([state])))
                else:
                    action = np.argmax(self.agent.predict(state))
                new_state, reward, done, _ = self.env.step(action)
                new_state = np.array([new_state])
                if render:
                    self.env.render()
                state = new_state

                if done:
                    break
                score += 1
            if score > max_score:
                max_score = score
            print("Epoch:", e, "score", score)
        print("Max score", max_score)

def encode(state):
    s_new = np.zeros(16)
    s_new[state] = 1
    return np.array(s_new)
def plot(results, saveFolder=None, ID=0,xs=None):
    plt.title("Score per episode")
    if not xs:
        plt.plot(results)
    else:
        plt.plot(xs, results)
    if saveFolder:
        plt.savefig(saveFolder)
    else:
        plt.show()
    plt.clf()

def run():
    global env
    lr = 0.001
    gamma = 0.99
    ID = 1
    agent = DQN_Agent(env, ID, lr, 'linear')
    agent.agent.summary()
    r = []
    running = True
    while running:
        command = input("Agent: " + str(ID) + "\nChoose action: \n\t- e - run DQN experiments \n\t- q - quit \n\t- t - train \n\t- p - plot \n\t- c - create new agent \n\t- r - test agent \n\t- a - change agent \n")
        if command=='q':
            running = False
            break
        if command=='t':
            r.extend(agent.train(gamma, 100))
        if command == 'p' and r:
            plot(r)
        if command=='r':
            agent.test(5)
        if command=='c':
            agent = DQN_Agent(env, ID, lr, 'linear')
            r = []
        if command=='a':
            temp = input("Choose ID: \n")
            ID = int(temp)
            r = []
        if command=='s':
            plot(r, MODELS_FOLDER + str(ID), ID)
        if command=='e':
            run_DQN_experiments()
        if command=='l':
            import tkinter as tk
            from tkinter import filedialog
            file_path = filedialog.askopenfilename(initialdir=MODELS_FOLDER, title="Pasirinkite agento modeli")
            agent = DQN_Agent(env, ID=999, filename=file_path)
def run_DQN_experiments():
    global env
    global MODELS_FOLDER
    MODELS_FOLDER = 'experiments/DQN_Agent/CartPole/models/'
    experiment_ID_file = 'experiments/DQN_Agent/CartPole/exp_ID.txt'
    if os.path.exists(experiment_ID_file):
        with open(experiment_ID_file, 'r+') as f:
            ID = int(f.readline())
            f.close()
    else:
        ID = 0

    #ID = run_experiment(ID, 500, 0.01,  0.99, 'linear', 'mse')
    ID = run_experiment(ID, 3000, 0.001, 0.99, 'linear', 'mse')
    ID = run_experiment(ID, 3000, 0.001, 0.9, 'linear', 'mse')
    ID = run_experiment(ID, 3000, 0.01,  0.99, 'linear', 'mse')
    ID = run_experiment(ID, 3000, 0.01,  0.9, 'linear', 'mse')
    ID = run_experiment(ID, 3000, 0.1,   0.99, 'linear', 'mse')
    ID = run_experiment(ID, 3000, 0.1,   0.9,  'linear', 'mse')

    with open(experiment_ID_file, 'w') as f:
        f.write(str(ID)+'\n')
        f.close()
def run_DQN_FrozenLake_experiments():
    global env
    global MODELS_FOLDER
    MODELS_FOLDER = 'experiments/DQN_Agent/FrozenLake/models/'
    experiment_ID_file = 'experiments/DQN_Agent/FrozenLake/exp_ID.txt'
    if os.path.exists(experiment_ID_file):
        with open(experiment_ID_file, 'r+') as f:
            ID = int(f.readline())
            f.close()
    else:
        ID = 0

    #ID = run_experiment(ID, 500, 0.01,  0.99, 'linear', 'mse')
    #ID = run_DQN_FrozenLake_experiment(ID, 1000, 0.01,  0.9, 'linear', 'mse')
    #ID = run_DQN_FrozenLake_experiment(ID, 10000, 0.0001, 0.99, 'linear', 'mse')
    #ID = run_DQN_FrozenLake_experiment(ID, 10000, 0.0001, 0.9, 'linear', 'mse')
    ID = run_DQN_FrozenLake_experiment(ID, 10000, 0.01, 0.99, 'linear', 'mse')
    ID = run_DQN_FrozenLake_experiment(ID, 10000, 0.01, 0.9, 'linear', 'mse')
    #ID = run_DQN_FrozenLake_experiment(ID, 10000, 0.1, 0.99, 'linear', 'mse')
    #ID = run_DQN_FrozenLake_experiment(ID, 10000, 0.1, 0.9, 'linear', 'mse')
    #ID = run_DQN_FrozenLake_experiment(ID, 1000, 0.005,  0.9, 'linear', 'mse')
    #ID = run_DQN_FrozenLake_experiment(ID, 1000, 0.005,  0.99,  'linear', 'mse')

    with open(experiment_ID_file, 'w') as f:
        f.write(str(ID)+'\n')
        f.close() 
def run_DQN_FrozenLake_experiment(ID, epochs = 100, lr=0.01, gamma=0.99, activation='linear', loss='mse'):
    env = gym.make("FrozenLake-v0")
    experiments_folder = 'experiments'
    agent_folder    = 'DQN_Agent/FrozenLake'
    folder          = experiments_folder + '/' + agent_folder
    file            = 'FrozenLake'+str(ID)
    fullPath        = folder + '/' + file
    fullPathWithExt = fullPath + '.txt'

    if not os.path.exists(experiments_folder):
        os.mkdir(experiments_folder)
    if not os.path.exists(folder):
        os.mkdir(folder)

    print("Running experiment " + str(ID) + ":")
    agent = DQN_Agent(env, ID, lr, activation, loss, state_size=env.observation_space.n)

    result_x_size = 100
    batchSize = 32

    results = []
    agent.epsilon = agent.EPS_START
    success_count = 0
    t_max = 200
    for e in range(epochs):
        done = False
        state = env.reset()
        state = encode(state)
        for i in range(t_max):                
            action = agent.act(state)
            new_state, reward, done, _ = env.step(action)
            new_state = encode(new_state)
            agent.rember([state, action, reward, new_state, done])
            state = new_state
            if done or i == t_max-1:
                results.append(reward)
                if reward > 0:
                    success_count += 1
                    print('episode: {}/{}, score: {}, epsilon: {}, reward: {}'.format(e, epochs, i, agent.epsilon, reward)) 
                if e % round(epochs * 0.1) == 0:
                    print('episode: {}/{}, score: {}, epsilon: {}, reward: {}'.format(e, epochs, i, agent.epsilon, reward))
                break
        agent.replay(gamma, batchSize)
        if success_count > 20 and agent.epsilon > agent.EPS_MIN:
            agent.epsilon *= agent.EPS_DECAY
    print('success_count:',  str(success_count))
    print('lr:',  str(lr))
    print('gamma:',  str(gamma))
    agent.save()

    train_results = results
    rewards_per_x_episodes = np.split(np.array(train_results),epochs/result_x_size)
    count = result_x_size
    results = [] # average rewards per result_x_size episodes
    for r in rewards_per_x_episodes:
        results.append(sum(r/result_x_size))
        count += result_x_size
    plot(results, fullPath, ID, xs=range(0, epochs, result_x_size))

    with open(fullPathWithExt, 'w') as f:
        f.write("Experiment "     + str(ID)     + ':\n')
        f.write("Epochs: "        + str(epochs) + '\n')
        f.write("Learning rate: " + str(lr)     + '\n')
        f.write("Gamma: "         + str(gamma)  + '\n')
        f.write("Activation: "    + activation  + '\n')
        f.write("Model summary:\n")
        agent.agent.summary(print_fn=lambda s: f.write(s + '\n'))
        f.write("Final score: " + str(results[len(results)-1]) + '\n')
        print("Final score: " + str(results[len(results)-1]))
    env.close()
    return ID+1

def run_frozen_lake_experiments():
    global MODELS_FOLDER
    MODELS_FOLDER = 'experiments/Q_Table/FrozenLake/models/'
    experiment_ID_file = 'experiments/Q_Table/FrozenLake/exp_ID.txt'
    if os.path.exists(experiment_ID_file):
        with open(experiment_ID_file, 'r+') as f:
            ID = int(f.readline())
            f.close()
    else:
        ID = 0
    global env    
    env = gym.make("FrozenLake-v0")
    #ID = run_frozen_lake_experiment(ID, epochs=30000, lr=0.1,  gamma=0.99, result_x_size=1000)
    #ID = run_frozen_lake_experiment(ID, epochs=30000, lr=0.1,  gamma=0.9, result_x_size=1000)
    #ID = run_frozen_lake_experiment(ID, epochs=30000, lr=0.01, gamma=0.99, result_x_size=1000)
    #ID = run_frozen_lake_experiment(ID, epochs=30000, lr=0.01, gamma=0.9, result_x_size=1000)
    ID = run_frozen_lake_experiment(ID, epochs=30000, lr=0.001,  gamma=0.99, result_x_size=1000)
    #ID = run_frozen_lake_experiment(ID, epochs=30000, lr=0.001,  gamma=0.9, result_x_size=1000)
    with open(experiment_ID_file, 'w') as f:
        f.write(str(ID)+'\n')
        f.close()
def run_cartpole_experiments():
    global MODELS_FOLDER
    MODELS_FOLDER = 'experiments/Q_Table/CartPole/models/'
    experiment_ID_file = 'experiments/Q_Table/CartPole/exp_ID.txt'
    if os.path.exists(experiment_ID_file):
        with open(experiment_ID_file, 'r+') as f:
            ID = int(f.readline())
            f.close()
    else:
        ID = 0
    global env
    env = gym.make("CartPole-v0")
    ID = run_cartpole_experiment(ID, epochs=100000, lr=0.5,  gamma=0.9, result_x_size=1000)
    ID = run_cartpole_experiment(ID, epochs=100000, lr=0.3,  gamma=0.9, result_x_size=1000)
    #ID = run_cartpole_experiment(ID, epochs=30000, lr=0.1,  gamma=0.99, result_x_size=1000)
    #ID = run_cartpole_experiment(ID, epochs=30000, lr=0.01, gamma=0.99, result_x_size=1000)
    #ID = run_cartpole_experiment(ID, epochs=30000, lr=0.01, gamma=0.99, result_x_size=1000)
    #ID = run_cartpole_experiment(ID, epochs=30000, lr=0.1,  gamma=0.99, result_x_size=1000)
    with open(experiment_ID_file, 'w') as f:
        f.write(str(ID)+'\n')
        f.close()
def run_frozen_lake_experiment(ID, epochs=100, lr=0.01, gamma=0.99, result_x_size=100):
    global env
    def choose_action(table, state):
        if random.uniform(0, 1) > epsilon:
            action = np.argmax(table.getValue(state))
        else:
            action = env.action_space.sample()
        return action
    experiments_folder = 'experiments'
    agent_folder    = 'Q_Table/FrozenLake'
    folder          = experiments_folder + '/' + agent_folder
    file            = 'FrozenLake_'+str(ID)
    fullPath        = folder + '/' + file
    fullPathWithExt = fullPath + '.txt'

    if not os.path.exists(experiments_folder):
        os.mkdir(experiments_folder)
    if not os.path.exists(folder):
        os.mkdir(folder)

    print("Running experiment " + str(ID) + ":")
    epsilon = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.01
    exploration_decay_rate = 0.01
    model = [QTable.Input(0, 15, 1, 0)]
    table = QTable.QTable(env.action_space.n, model=model)
    rewards = []
    max_steps = 200
    for e in range(epochs):
        state = env.reset()
        done = False
        r = 0
        for s in range(max_steps):
            action = choose_action(table, state)
            new_state, reward, done, _ = env.step(action)
            v = table.getValue(state)
            q_new = table.getValue(state)[action] * (1-lr) + lr * (reward + gamma * np.max(table.getValue(new_state)))
            table.setValue(state, action, q_new)
            state = new_state
            r += reward
            if done:
                break
        epsilon = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*e)
        rewards.append(r)
    
    rewards_per_x_episodes = np.split(np.array(rewards),epochs/result_x_size)
    count = result_x_size

    results = [] # average rewards per result_x_size episodes
    for r in rewards_per_x_episodes:
        results.append(sum(r/result_x_size))
        count += result_x_size
    
    with open(fullPathWithExt, 'w') as f:
        f.write("Experiment "     + str(ID)        + ':\n')
        f.write("Epochs: "        + str(epochs)    + '\n')
        f.write("Learning rate: " + str(lr)        + '\n')
        f.write("Gamma: "         + str(gamma)     + '\n')
        f.write("Last Reward: "   + str(rewards[len(rewards)-1])     + '\n')

        plot(results, fullPath, ID, xs=[i for i in range(result_x_size, epochs+1, result_x_size)])
        f.write("Final score: " + str(results[len(results)-1]) + '\n')
        print("Final score: " + str(results[len(results)-1]))
    table.save(MODELS_FOLDER+str(ID))
    return ID+1

def run_cartpole_experiment(ID, epochs=100, lr=0.01, gamma=0.99, result_x_size=100):
    def choose_action(table, state):
        if random.uniform(0, 1) > epsilon:
            action = np.argmax(table.getValue(state))
        else:
            action = env.action_space.sample()
        return action
    global env
    experiments_folder = 'experiments'
    agent_folder    = 'Q_Table/CartPole'
    folder          = experiments_folder + '/' + agent_folder
    file            = 'CartPole_'+str(ID)
    fullPath        = folder + '/' + file
    fullPathWithExt = fullPath + '.txt'

    if not os.path.exists(experiments_folder):
        os.mkdir(experiments_folder)
    if not os.path.exists(folder):
        os.mkdir(folder)

    print("Running experiment " + str(ID) + ":")
    epsilon = 1
    max_exploration_rate = 1
    min_exploration_rate = 0.01
    exploration_decay_rate = 0.01
    model = [QTable.Input(-1, 1, 0.1, 4), 
             QTable.Input(-1, 1, 0.1, 4),
             QTable.Input(-1, 1, 0.1, 4),
             QTable.Input(-1, 1, 0.1, 4)]
    import DQTable as dqt
    table = dqt.DQTable(env.action_space.n, model=model)
    rewards = []
    max_steps = 500
    for e in range(epochs):
        state = env.reset()
        done = False
        for s in range(max_steps):
            action = choose_action(table, state)
            new_state, reward, done, _ = env.step(action)
            reward = reward  if not done else -10
            q_new = table.getValue(state)[action] * (1-lr) + lr * (reward + gamma * np.max(table.getValue(new_state)))
            table.setValue(state, action, q_new)
            #table.setValue(state, action, q_new, e < 10)
            state = new_state
            if done:
                rewards.append(s)
                if e % round(epochs * 0.1) == 0:
                    print('episode: {}/{}, score: {}'.format(e, epochs, s))
                break
        epsilon = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*e)
    
    rewards_per_x_episodes = np.split(np.array(rewards),epochs/result_x_size)
    count = result_x_size

    results = [] # average rewards per result_x_size episodes
    for r in rewards_per_x_episodes:
        results.append(sum(r/result_x_size))
        count += result_x_size
    
    with open(fullPathWithExt, 'w') as f:
        f.write("Experiment "     + str(ID)        + ':\n')
        f.write("Epochs: "        + str(epochs)    + '\n')
        f.write("Learning rate: " + str(lr)        + '\n')
        f.write("Gamma: "         + str(gamma)     + '\n')
        f.write("Last Reward: "   + str(rewards[len(rewards)-1])     + '\n')

        plot(results, fullPath, ID, xs=[i for i in range(result_x_size, epochs+1, result_x_size)])
        f.write("Final score: " + str(results[len(results)-1]) + '\n')
        print("Final score: " + str(results[len(results)-1]))
    table.save(MODELS_FOLDER+str(ID))
    return ID+1

def load_frozen_lake_agent(ID):
    global MODELS_FOLDER
    MODELS_FOLDER = 'experiments/Q_Table/FrozenLake/models/'
    return QTable.QTable.load(MODELS_FOLDER+str(ID))

def play_frozen_lake(table, n=1, verbose=0):
    # verbose 0 - no priniting, with visualization, 1 - only printing, 2 - print and plot graph, 3 - only plot graph
    env = gym.make("FrozenLake-v0")
    epsilon = 1
    rewards = []
    def choose_action(table, state):
        if random.uniform(0, 1) > epsilon:
            action = np.argmax(table.getValue(state))
        else:
            action = env.action_space.sample()
        return action
    for e in range(n):
        import time
        state = env.reset()
        done = False
        r = 0
        if verbose == 1 or verbose == 2:
            print("*****EPISODE ", e+1, "*****\n\n\n\n")
            time.sleep(1)
        for s in range(100):
            if verbose == 0:
                env.render()
            action = choose_action(table, state)
            new_state, reward, done, _ = env.step(action)
            state = new_state
            r += reward
            
            if done:
                if verbose == 1 or verbose == 2:
                    print("Your reward:", reward)
                    time.sleep(3)
                break
        rewards.append(r)
    env.close()
    if verbose == 3 or verbose == 2:
        plot(rewards)
    return rewards
    
def run_experiment(ID, epochs = 100, lr=0.01, gamma=0.99, activation='linear', loss='mse'):
    global env
    experiments_folder = 'experiments'
    agent_folder    = 'DQN_Agent/CartPole'
    folder          = experiments_folder + '/' + agent_folder
    file            = 'CartPole_'+str(ID)
    fullPath        = folder + '/' + file
    fullPathWithExt = fullPath + '.txt'

    if not os.path.exists(experiments_folder):
        os.mkdir(experiments_folder)
    if not os.path.exists(folder):
        os.mkdir(folder)

    print("Running experiment " + str(ID) + ":")
    print('lr:',  str(lr), 'gamma:',  str(gamma))
    agent = DQN_Agent(env, ID, lr, activation, loss)

    result_x_size = 50
    batchSize = 50

    results = []
    agent.epsilon = 1
    t_max = 500
    for e in range(epochs):
        done = False
        state = env.reset()
        for i in range(t_max):                
            action = agent.act(state)
            new_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            agent.rember([state, action, reward, new_state, done])
            state = new_state
            if done:
                results.append(i)
                if e % round(epochs * 0.02) == 0:
                    print('episode: {}/{}, score: {}, epsilon: {}, reward: {}'.format(e, epochs, i, agent.epsilon, reward))
                break
        agent.replay(gamma, batchSize)
        if agent.epsilon > agent.EPS_MIN:
            agent.epsilon *= agent.EPS_DECAY
    agent.save()

    train_results = results
    rewards_per_x_episodes = np.split(np.array(train_results),epochs/result_x_size)
    count = result_x_size
    results = [] # average rewards per result_x_size episodes
    for r in rewards_per_x_episodes:
        results.append(sum(r/result_x_size))
        count += result_x_size
    plot(results, fullPath, ID, xs=range(0, epochs, result_x_size))

    with open(fullPathWithExt, 'w') as f:
        f.write("Experiment "     + str(ID)     + ':\n')
        f.write("Epochs: "        + str(epochs) + '\n')
        f.write("Learning rate: " + str(lr)     + '\n')
        f.write("Gamma: "         + str(gamma)  + '\n')
        f.write("Activation: "    + activation  + '\n')
        f.write("Loss: "          + loss        + '\n\n')
        f.write("Model summary:\n")
        agent.agent.summary(print_fn=lambda s: f.write(s + '\n'))
        f.write("Final score: " + str(train_results[len(train_results)-1]) + '\n')
        print("Final score: " + str(train_results[len(train_results)-1]))

    return ID+1

if __name__ == "__main__":
    import sys
    global env
    #env = gym.make("CartPole-v0")
    env = gym.make("FrozenLake-v0")
   
    #agent = load_frozen_lake_agent(16)
    #play_frozen_lake(agent, n=1000, verbose=3)
    run_DQN_experiments()
    #run_frozen_lake_experiments()
    #run_cartpole_experiments()
    #run_cartpole_experiments()
    #run_DQN_FrozenLake_experiments()
    env.close()
    sys.exit(0)