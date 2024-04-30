import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from environment import Environment1
from data_loader import load_data, split_data
from plot import plot_loss_reward, plot_train_test_by_q
import os
import pandas as pd


# Dueling Double DQN

class Q_Network(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Q_Network, self).__init__(
            
        )
        self.state_value = nn.Linear(hidden_size//2, 1)
        self.advantage_value = nn.Linear(hidden_size//2, output_size)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size//2)
        self.fc4 = nn.Linear(hidden_size, hidden_size//2)
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        hs = F.relu(self.fc3(h))
        ha = F.relu(self.fc4(h))
        state_value = self.state_value(hs)
        advantage_value = self.advantage_value(ha)
        advantage_mean = (torch.sum(advantage_value, axis=1)/float(self.output_size)).reshape(-1, 1)
        q_value = torch.concat([state_value for _ in range(self.output_size)], axis=1) + (advantage_value - torch.concat([advantage_mean for _ in range(self.output_size)], axis=1))
        return q_value

    def reset(self):
        self.zero_grad()
""" >>> """

def train_dddqn(env, Q = None):

    """ <<< Double DQN -> Dueling Double DQN
    class Q_Network(chainer.Chain):

        def __init__(self, input_size, hidden_size, output_size):
            super(Q_Network, self).__init__(
                fc1 = nn.Linear(input_size, hidden_size),
                fc2 = nn.Linear(hidden_size, hidden_size),
                fc3 = nn.Linear(hidden_size, output_size)
            )

        def forward(self, x):
            h = nn.relu(self.fc1(x))
            h = nn.relu(self.fc2(h))
            y = self.fc3(h)
            return y

        def reset(self):
            self.zerograds()
    === """

    if Q is None:
        Q = Q_Network(input_size=env.history_t+1, hidden_size=100, output_size=3)
    Q_ast = copy.deepcopy(Q)
    optimizer = optim.Adam(Q.parameters(), lr=0.0001)

    epoch_num = 50
    step_max = len(env.data)-1
    memory_size = 200
    batch_size = 50
    epsilon = 1.0
    epsilon_decrease = 1e-3
    epsilon_min = 0.1
    start_reduce_epsilon = 200
    train_freq = 10
    update_q_freq = 20
    gamma = 0.97
    show_log_freq = 5

    memory = []
    total_step = 0
    total_rewards = []
    total_losses = []

    start = time.time()
    for epoch in range(epoch_num):

        pobs = env.reset()
        step = 0
        done = False
        total_reward = 0
        total_loss = 0

        while not done and step < step_max:

            # select act
            pact = np.random.randint(3)
            if np.random.rand() > epsilon:
                pact = Q(torch.tensor(pobs, dtype=torch.float32).reshape(1, -1))
                pact = np.argmax(pact.data)

            # act
            obs, reward, done = env.step(pact)

            # add memory
            memory.append((pobs, pact, reward, obs, done))
            if len(memory) > memory_size:
                memory.pop(0)

            # train or update q
            if len(memory) == memory_size:
                if total_step % train_freq == 0:
                    memory_ = np.asarray(memory, dtype=object)
                    shuffled_memory = np.random.permutation(memory_)
                    memory_idx = range(len(shuffled_memory))
                    for i in memory_idx[::batch_size]:
                        batch = np.array(shuffled_memory[i:i+batch_size])
                        b_pobs = np.array(batch[:, 0].tolist(), dtype=np.float32).reshape(batch_size, -1)
                        b_pact = np.array(batch[:, 1].tolist(), dtype=np.int32)
                        b_reward = np.array(batch[:, 2].tolist(), dtype=np.int32)
                        b_obs = np.array(batch[:, 3].tolist(), dtype=np.float32).reshape(batch_size, -1)
                        b_done = np.array(batch[:, 4].tolist(), dtype=np.bool)

                        q = Q(torch.tensor(b_pobs))
                        """ <<< DQN -> Double DQN
                        maxq = np.max(Q_ast(b_obs).data, axis=1)
                        === """
                        indices = np.argmax(q.data, axis=1)
                        maxqs = Q_ast(torch.tensor(b_obs)).data
                        """ >>> """
                        target = copy.deepcopy(q.data)
                        for j in range(batch_size):
                            """ <<< DQN -> Double DQN
                            target[j, b_pact[j]] = b_reward[j]+gamma*maxq[j]*(not b_done[j])
                            === """
                            target[j, b_pact[j]] = b_reward[j]+gamma*maxqs[j, indices[j]]*(not b_done[j])
                            """ >>> """
                        Q.reset()
                        loss = F.smooth_l1_loss(q, torch.tensor(target))
                        total_loss += loss.data
                        loss.backward()
                        optimizer.step()

                if total_step % update_q_freq == 0:
                    Q_ast = copy.deepcopy(Q)

            # epsilon
            if epsilon > epsilon_min and total_step > start_reduce_epsilon:
                epsilon -= epsilon_decrease

            # next step
            total_reward += reward
            pobs = obs
            step += 1
            total_step += 1

        total_rewards.append(total_reward)
        total_losses.append(total_loss)

        if (epoch+1) % show_log_freq == 0:
            log_reward = sum(total_rewards[((epoch+1)-show_log_freq):])/show_log_freq
            log_loss = sum(total_losses[((epoch+1)-show_log_freq):])/show_log_freq
            elapsed_time = time.time()-start
            print('\t'.join(map(str, [epoch+1, epsilon, total_step, log_reward, log_loss, elapsed_time])))
            start = time.time()
            
    return Q, total_losses, total_rewards


def train_folder(load_Q = False):
    Q = None
    data_path = 'src/data'
    output_data = []
    for i, filename in enumerate(os.listdir(data_path)):
        stock_name = filename.split(".")[0]
        model_path = 'src/models/dddqn/main/'
        output_data.append(stock_name)


        data = load_data(data_path + '/' + filename)
        date_split = pd.read_csv(data_path + '/' + filename)["Date"][int(len(data) * 0.9)]
        train, test = split_data(data, date_split)

        if not load_Q:
            Q, total_losses, total_rewards = train_dddqn(Environment1(train), Q)
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            torch.save(Q, model_path + 'network_data.pth')
            np.save(model_path + 'total_losses.npy', total_losses)
            np.save(model_path + 'total_rewards.npy', total_rewards)

        if load_Q:
            Q = torch.load(model_path + 'network_data.pth')
            total_losses = np.load(model_path + 'total_losses.npy')
            total_rewards = np.load(model_path + 'total_rewards.npy')

        if i == 10:
            break
    print(output_data)

    plot_loss_reward(total_losses, total_rewards)
    plot_train_test_by_q(Environment1(train), Environment1(test), Q, 'Dueling Double DQN', date_split, "src/plots/" + stock_name + "/dddqn.html")

def train_one(filename, load_Q = False):
    data_path = 'src/data'
    stock_name = filename.split(".")[0]
    model_path = 'src/models/dddqn/'


    data = load_data(data_path + '/' + filename)
    date_split = pd.read_csv(data_path + '/' + filename)["Date"][int(len(data) * 0.8)]
    train, test = split_data(data, date_split)

    if not load_Q:
        Q, total_losses, total_rewards = train_dddqn(Environment1(train))
        if not os.path.exists(model_path + stock_name):
            os.makedirs(model_path + stock_name)
        torch.save(Q, model_path + stock_name + '/network_data.pth')
        np.save(model_path + stock_name + '/total_losses.npy', total_losses)
        np.save(model_path + stock_name + '/total_rewards.npy', total_rewards)

    if load_Q:
        Q = torch.load(model_path + stock_name + '/network_data.pth')
        total_losses = np.load(model_path + stock_name + '/total_losses.npy')
        total_rewards = np.load(model_path + stock_name + '/total_rewards.npy')

    
    plot_loss_reward(total_losses, total_rewards)
    plot_train_test_by_q(Environment1(train), Environment1(test), Q, 'Dueling Double DQN', date_split, "src/plots/" + stock_name + "/dddqn.html")

if __name__ == '__main__':
    train_one('googl.us.txt', load_Q=True)


    # Q_Rand = Q_Network(input_size=91, hidden_size=100, output_size=3)
    # plot_train_test_by_q(Environment1(train), Environment1(test), Q_Rand, 'Random DQN', date_split, "src/plots/" + stock_name + "/random.html")

    # data_path = 'src/data'
    # filename = 'googl.us.txt'
    # stock_name = filename.split(".")[0]
    # model_path = 'src/models/dddqn/'


    # data = load_data(data_path + '/' + filename)
    # date_split = pd.read_csv(data_path + '/' + filename)["Date"][int(len(data) * 0.8)]
    # train, test = split_data(data, date_split)

    # Q = torch.load(model_path + 'network_data.pth')
    # total_losses = np.load(model_path + 'total_losses.npy')
    # total_rewards = np.load(model_path + 'total_rewards.npy')

    
    # plot_loss_reward(total_losses, total_rewards)
    # plot_train_test_by_q(Environment1(train), Environment1(test), Q, 'Dueling Double DQN', date_split, "src/plots/" + stock_name + "/dddqn.html")