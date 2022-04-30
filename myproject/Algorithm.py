import math
import numpy as np
class Random():
    def __init__(self, arm_k, m):
        self.arm_k = arm_k
        self.m = m
    def pull(self, counts, values):
        return np.random.randint(0, self.arm_k, size = self.m)

class EpsilonGreedy():
    def __init__(self, arm_k, m, epsilon):
        self.arm_k = arm_k
        self.m = m
        self.epsilon = epsilon
    def pull(self, counts, values):
        res = []
        for arm in range(self.arm_k):
            if counts[arm] == 0 and len(res)<self.m:
                res.append(arm)
        if len(res)>0:
            return res
        greedyindex = np.random.random()
        if greedyindex < self.epsilon:
            return np.random.randint(0, self.arm_k, size = self.m)
        else:
            return np.argsort(values)[-self.m:]

class UCB1():
    def __init__(self, arm_k,m):
        self.arm_k = arm_k
        self.m=m
        self.UCB = np.zeros(self.arm_k)
    def pull(self, counts, values):
        for arm in range(self.arm_k):
            if counts[arm] == 0:
                return arm
        t = np.sum(counts)
        for arm in range(self.arm_k):
            self.UCB[arm] = values[arm] - np.sqrt(2 * np.log(t) / counts[arm])
        return np.argsort(values)[-self.m:]

class CUCB():
    def __init__(self, arm_k, m):
        self.arm_k = arm_k
        self.m = m  # 定义选取多少个臂
        self.MUCB = np.zeros(self.arm_k)
    def pull(self, counts, values):
        res = []
        for arm in range(self.arm_k):
            if counts[arm] == 0 and len(res)<self.m:
                res.append(arm)
        if len(res)>0:
            return res
        t = np.sum(counts)
        for arm in range(self.arm_k):
            self.MUCB[arm] = values[arm] + np.sqrt(2 * np.log(t) / counts[arm])
        return np.argsort(self.MUCB)[-self.m:]

class Softmax():
    def __init__(self, arm_k,m, T):
        self.arm_k = arm_k
        self.m = m  # 定义选取多少个臂
        self.T=T
        self.softmax = np.zeros(self.arm_k)
    def pull(self, counts, values):
        res = []
        temp=0
        for arm in range(self.arm_k):
            if counts[arm] == 0 and len(res)<self.m:
                res.append(arm)
            temp+=math.exp(values[arm]/self.T)
        if len(res)>0:
            return res
        for arm in range(self.arm_k):
            self.softmax[arm] = math.exp(values[arm]/self.T)/temp
        return np.argsort(self.softmax)[-self.m:]

class ACUCB():
    def __init__(self, arm_k, skc, m):
        self.arm_k = arm_k
        self.values = np.zeros(self.arm_k)
        self.counts = np.zeros(self.arm_k)
        self.skc = skc #定义自适应的变量
        self.m = m  # 定义选取多少个臂
        self.AMUCB = np.zeros(self.arm_k)
    def pull(self):
        for arm in range(self.arm_k):
            if self.counts[arm] == 0:
                return arm
        t = np.sum(self.counts)
        for arm in range(self.arm_k):
            sk1 = np.maximum(np.minimum((self.skc - np.amin(self.skc)) / (np.amax(self.skc) - np.amin(self.skc)), 1), 0)
            self.AMUCB[arm] = self.values[arm] - np.sqrt(2 * sk1 * np.log(t) / self.counts[arm])
        # print(self.UCB)
        return np.argsort(self.AMUCB)[0: self.m]
    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]

class BCUCB():
    def __init__(self, arm_k, m, B, c):
        self.arm_k = arm_k
        self.values = np.zeros(self.arm_k)
        self.counts = np.zeros(self.arm_k)
        self.m = m  # 定义选取多少个臂
        self.BMUCB = np.zeros(self.arm_k) # 关于Budget UCB算法的实现
        self.Budget = B # 所有的预算
        self.cost = c  # 探索的开销
    def pull(self):
        for arm in range(self.arm_k):
            if self.counts[arm] == 0:
                return arm
        t = np.sum(self.counts)
        for arm in range(self.arm_k):
            self.BMUCB[arm] = self.values[arm] - np.sqrt(2 * np.log(t) / self.counts[arm])
        return np.argsort(self.MUCB)[0: self.m]
    def update(self, arm, reward):
        self.counts[arm] += 1
        self.values[arm] += (reward - self.values[arm]) / self.counts[arm]
        self.Budget -= self.cost * self.m
