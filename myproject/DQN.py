import heapq
import random
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from myproject.model import Net

MEMORY_CAPACITY = 2000  # replay memory的大小，越大越占用内存
MEMORY_WARMUP_SIZE = 200  # replay_memory 里需要预存一些经验数据，再从里面sample一个batch的经验让agent去learn
cloud_size=5            #边缘云容量
GAMMA = 0.9
LR = 0.001  # 学习率
EPSILON=0.9
TARGET_NETWORK_REPLACE_FREQ = 10  # How frequently target netowrk updates
BATCH_SIZE = 32
obs_dim=20
service_number=20

class DQN(object):
    def __init__(self,obs_dim,action_dim):
        self.eval_net, self.target_net = Net(obs_dim,action_dim),Net(obs_dim,action_dim)
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = np.zeros((MEMORY_CAPACITY, obs_dim * 2 + cloud_size+1))
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()

    def choose_action(self, x):        #choice返回的是具体的选择
        x = torch.unsqueeze(torch.FloatTensor(x), 0)  # add 1 dimension to input state x
        if np.random.uniform() < EPSILON:  # greedy
            actions_value = self.target_net.forward(x)        #这里返回的是20个action
            actions_numpy=actions_value[0].data.numpy()
            choice=heapq.nlargest(cloud_size, range(len(actions_numpy)), actions_numpy.take)
        else:  # random
            choice =random.sample(range(0, 20), 5)
        return choice

    def reward_eval_action(self,x):
        x = torch.unsqueeze(torch.FloatTensor(x), 0)  # add 1 dimension to input state x
        actions_value = self.target_net.forward(x)  # 这里返回的是20个action
        actions_numpy = actions_value[0].data.numpy()
        choice = heapq.nlargest(cloud_size, range(len(actions_numpy)), actions_numpy.take)
        return choice

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, r, s_))  # horizontally stack these vectors
        index = self.memory_counter % MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1

    def learn(self):
        if self.learn_step_counter % TARGET_NETWORK_REPLACE_FREQ == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)  # randomly select some data from buffer
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :obs_dim]))
        b_a = Variable(torch.LongTensor(b_memory[:, obs_dim:obs_dim + cloud_size]))
        b_r = Variable(torch.FloatTensor(b_memory[:, obs_dim + cloud_size:obs_dim + cloud_size+1]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -obs_dim:]))

        q_eval = self.eval_net(b_s).gather(1, b_a)  # (batch_size, 1) 根据 action选择除了 5个元素,将他们加起来
        q_eval=torch.sum(q_eval, dim=(1, ), keepdim=True)

        q_next = self.target_net(b_s_).detach()  # detach from computational graph, don't back propagate
        indices = q_next.topk(k=5, dim=1, largest=True, sorted=True)
        q_next=torch.sum(indices[0],dim=(1, ), keepdim=True)
        q_target = b_r + GAMMA * q_next.view(BATCH_SIZE, 1)  # (batch_size, 1)

        loss =self.loss_func(q_eval, q_target)    #均方差损失函数
        self.optimizer.zero_grad()  # reset the gradient to zero
        loss.backward()
        self.optimizer.step()  # execute back propagation for one step
        return loss

    def storemodel(self):
        torch.save(self.target_net, '../myproject/dqn.pt')

