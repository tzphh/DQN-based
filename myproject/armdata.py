import itertools

import numpy as np
class edgecaching():
    # 定义服务缓存常用的变量
    def __init__(self, arm_k, skc, wkc, ske, wke,acc,ace,datac,datae,demand_helper):
        # 多少个服务
        self.arm_k = arm_k
        # 服务的计算大小和数据量
        self.skc = skc # 定义每个任务在云端云端处理的大小 s所需计算资源，w所占的容量
        self.wkc = wkc
        self.ske = ske # 定义边缘云的大小
        self.wke = wke
        self.acc = acc #定义服务的准确度
        self.ace = ace
        self.datac=datac
        self.datae=datae
        self.demand_helper = demand_helper

        # 服务所带来的期望延时与能耗
        self.expected_delay = np.zeros(self.arm_k) # 每个服务带来的延时
        self.true_delay = np.zeros(self.arm_k)
        self.expected_energy = np.zeros(self.arm_k) # 每个服务带来的能耗
        self.true_energy = np.zeros(self.arm_k)

        # 服务缓存带来的奖励，此处也可以考虑延时与能耗的比
        self.delay = 0
        self.energy = 0
        self.reward = np.zeros(self.arm_k)

        # 对于step函数的定义
        self.estimated_delay = np.zeros(self.arm_k)
        self.estimated_energy = np.zeros(self.arm_k)
        self.estimated_accuracy = np.zeros(self.arm_k)
        self.estimated_reward = np.zeros(self.arm_k)

        self.counts = np.zeros(self.arm_k)
        self.value_delay = np.zeros(self.arm_k)
        self.value_energy = np.zeros(self.arm_k)
        self.value_accuracy = np.zeros(self.arm_k)
        self.value_reward=np.zeros(self.arm_k)

    #---------------------------------
    # 边缘服务数据的产生
    #--------------------------
    def generate_bandit_data(self, expected_demand, true_demand):
        self.expected_delay = np.zeros(self.arm_k) # 每个服务带来的延时
        self.true_delay = np.zeros(self.arm_k)
        self.expected_energy = np.zeros(self.arm_k) # 每个服务带来的能耗
        self.true_energy = np.zeros(self.arm_k)
        self.expected_accuracy=np.zeros(self.arm_k)
        self.true_accuracy = np.zeros(self.arm_k) #每个服务的准确度
        self.expected_reward=np.zeros(self.arm_k)
        self.true_reward=np.zeros(self.arm_k)
        # 计算边缘云缓存的延时
        for i in range(0, self.arm_k):
            # 计算均在云端
            for j in range(0, self.arm_k):
                self.expected_delay[i] += expected_demand[j] * (self.wkc[j] + self.datac[j])
                self.true_delay[i] += true_demand[j] * (self.wkc[j] + self.datac[j])
            # 第i个在边缘云上处理
            self.expected_delay[i] -= expected_demand[i] * (self.skc[i] + self.datac[i])
            self.true_delay[i] -= true_demand[i] * (self.skc[i] + self.datac[i])
            # 在边缘云上处理的延时
            self.expected_delay[i] += expected_demand[i] * (self.wke[i] + self.datae[i]/1.1)
            self.true_delay[i] += true_demand[i] * (self.wke[i] + self.datae[i]/1.1)

        # 计算边缘云缓存的能耗
        for i in range(0, self.arm_k):
            for j in range(0, self.arm_k):
                 self.expected_energy[i] +=expected_demand[i] * (self.wkc[i] + self.skc[i]*self.skc[i])
                 self.true_energy[i]  +=true_demand[i] * (self.wkc[i] + self.skc[i]*self.skc[i])
            self.expected_energy[i] -=expected_demand[i] * (self.wkc[i] + self.skc[i]*self.skc[i])
            self.true_energy[i]  -=true_demand[i] * (self.wkc[i] + self.skc[i]*self.skc[i])
            self.expected_energy[i] += expected_demand[i] *( self.wke[i] + self.ske[i] * self.ske[i])
            self.true_energy[i] += true_demand[i] * (self.wke[i] + self.ske[i] * self.ske[i])
        #计算边缘云能耗的准确度

        for i in range(0, self.arm_k):
            for j in range(0, self.arm_k):
                self.expected_accuracy[i]+=true_demand[i] * self.acc[i]
                self.true_accuracy[i]+=true_demand[i] * self.acc[i]

            self.expected_accuracy[i] -= true_demand[i] * self.acc[i]
            self.true_accuracy[i] -= true_demand[i] * self.acc[i]
            self.expected_accuracy[i] += true_demand[i] * self.ace[i]
            self.true_accuracy[i] += true_demand[i] * self.ace[i]

        self.expected_reward=self.expected_accuracy-self.expected_delay-self.expected_energy #3/16实验
        self.true_reward = self.true_accuracy - self.true_delay - self.true_energy  # 3/16实验
        return self.expected_reward,self.true_reward

    def setenvironment(self, state_dim):  # 初始化环境，state_dim表示有多少个用于训练的状态，即一个episode多少个训练
        demand_helper=np.zeros(self.arm_k)
        true_helper=np.random.rand(self.arm_k)
        self.true_demand = np.zeros([state_dim, self.arm_k])
        for i in range(0, self.arm_k):
            self.true_demand[:, i] = (np.random.poisson(true_helper[i] * 10000000,
                                                        state_dim) / 10000000)  # .reshape(state_dim,1)
        self.true_demand.astype(np.float)
        return self.true_demand

    #----------------------=-
    # 关于step函数的计算
    #-------------------------
    def step(self, true_demand, choice, previous_choice):
        # 当服务i缓存时带来的延时减少
        self.delay = 0 # 第episode 所消耗的延时
        self.energy = 0 # 第episode所消耗的能耗
        self.accuracy = 0 #第episode所改变的准确度

        # 以数组形式记录该次请求时的服务缓存选择
        self.cur_choice = np.zeros(self.arm_k)
        for i in choice:
            self.cur_choice[i] = 1
            for i in range(0, self.arm_k):
                self.delay += true_demand[i] * (self.wkc[i] + self.datac[i])
            for i in choice:
                self.delay -= true_demand[i] * (self.wkc[i] + self.datac[i])
                self.delay += true_demand[i] * (self.wke[i] + self.datae[i]/1.1+self.wke[i]* (1 if self.cur_choice[i] != previous_choice[i] else 0))

            # 当服务i缓存带来的能耗增加
            for i in range(0, self.arm_k):
                self.energy += true_demand[i] * (self.wkc[i] + self.skc[i]*self.skc[i])
            for i in choice:
                self.energy -= true_demand[i] * (self.wkc[i] + self.skc[i]*self.skc[i])
                self.energy += true_demand[i] * (self.wke[i]* (2 if self.cur_choice[i] != previous_choice[i] else 1) + self.ske[i] *self.ske[i])
                #这里用wke[e]来作为对应的替换能耗

            # 当服务缓存i时带来的准确度改变
            for i in range(0, self.arm_k):
                self.accuracy +=  true_demand[i] * self.acc[i]
            for i in choice:
                self.accuracy -=  true_demand[i] * self.acc[i]
                self.accuracy +=  true_demand[i] * self.ace[i]
        # 计算缓存所带来的奖励
        self.reward=self.accuracy-self.delay-self.energy #3/16实验

        return self.reward,self.delay, self.energy, self.accuracy

    #--------------------------------
    # 对于update函数，我们应该如何更新；
    #-------------------------------
    def update(self, true_demand, choice, previous_choice):
        # 以数组形式记录该次请求时的服务缓存选择
        self.cur_choice = np.zeros(self.arm_k)
        for i in choice:
            self.cur_choice[i] = 1
        self.delay_temp = 0
        self.energy_temp = 0
        self.accuracy_temp = 0

        for i in range(0, self.arm_k):
            self.delay_temp += true_demand[i] *(self.wkc[i] + self.datac[i])
            self.energy_temp += true_demand[i] *( self.wkc[i] + self.skc[i]*self.skc[i])
            self.accuracy_temp += true_demand[i] *self.acc[i]

            # 计算第i个臂带来的延时与能耗
        for i in choice:
            self.counts[i] += 1

            self.estimated_delay[i]=self.delay_temp
            self.estimated_delay[i] -= true_demand[i] * (self.wkc[i] + self.datac[i])
            self.estimated_delay[i] += true_demand[i] * (self.wke[i] + self.datae[i]/1.1+self.wke[i]* (1 if self.cur_choice[i] != previous_choice[i] else 0))
            self.value_delay[i]  = (self.estimated_delay[i] + self.value_delay[i]*(self.counts[i]-1)) / self.counts[i]

            self.estimated_energy[i]=self.energy_temp
            self.estimated_energy[i] -= true_demand[i] * (self.wkc[i] + self.skc[i] * self.skc[i])
            self.estimated_energy[i]+= true_demand[i] * (self.wke[i]* (2 if self.cur_choice[i] != previous_choice[i] else 1) + self.ske[i] *self.ske[i])
            self.value_energy[i] = (self.estimated_energy[i] + self.value_energy[i]*(self.counts[i]-1)) / self.counts[i]

            self.estimated_accuracy[i] = true_demand[i] * self.accuracy_temp
            self.estimated_accuracy[i] -= true_demand[i] * self.acc[i]
            self.estimated_accuracy[i] += true_demand[i] * self.ace[i]
            self.value_accuracy[i] = (self.estimated_accuracy[i] + self.value_accuracy[i] * (self.counts[i] - 1)) /self.counts[i]

            reward_temp=self.value_accuracy[i]-self.value_delay[i]-self.value_energy[i]
            self.value_reward[i]  = (reward_temp+ self.value_reward[i] * (self.counts[i] - 1)) /self.counts[i]

        return self.counts, self.value_delay, self.value_energy, self.value_accuracy, self.cur_choice,self.value_reward



