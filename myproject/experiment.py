import itertools
import numpy as np
import matplotlib.pyplot as plt
import torch
from armdata import edgecaching
from Algorithm import Random, EpsilonGreedy, UCB1, CUCB, ACUCB, BCUCB,Softmax
from DQN import DQN

#--------------------------------
# 定义超参数
#----------------------------------
service_number = 20  # 服务的个数
epsilon = 0.1   # epsilon 算法参数
choice_number = 5 # 每次选取的个数
temperature=0.2
obs_dim=20
act_dim=20
train_dim=10
cloud_size=5
pre_dim=5
previous_choice=[0,0,0,0,0]
current_choice=[0,0,0,0,0]
MEMORY_CAPACITY = 2000
pre_episodes=500
train_episodes=20000
episodes = 400  # 实验运行次数
number_experiments = 100  # 实验次数 为100
dqn = DQN(obs_dim, act_dim)

#--------------------------------
# 边缘云系统相关参数初始化
#----------------------------------
skc = np.random.normal(10, 15, service_number) / 10  # sk服从正态分布， 在云端进行处理
while sum(skc < 0):  # 确保生成的数据中不包含负值
    skc = np.random.normal(10, 15, service_number) / 10
wkc = np.random.normal(10, 15, service_number) / 10  # wk服从正态分布
while sum(wkc < 0):
    wkc = np.random.normal(10, 15, service_number) / 10
ske = np.random.normal(10, 15, service_number) / 3000  # 在边缘处理的延时
while sum(ske < 0):
    ske = np.random.normal(10, 15, service_number) / 3000
wke = np.random.normal(10, 5, service_number) / 3000
while sum(wke < 0):
    wke = np.random.normal(10, 5, service_number) / 3000
acc =np.random.randint(8000,10000,size=service_number)
acc=acc/10000
ace =np.random.randint(8000,10000,size=service_number)
ace=ace/10000
datac = np.random.normal(10, 15, service_number) / 10  # sk服从正态分布， 在云端进行处理
while sum(datac < 0):  # 确保生成的数据中不包含负值
    datac = np.random.normal(10, 15, service_number) / 10
datae = np.random.normal(10, 15, service_number) / 10  # wk服从正态分布
while sum(datae < 0):
    datae= np.random.normal(10, 15, service_number) / 10
demand_helper = np.random.rand(service_number)
expected_demand = demand_helper
caching_environment = edgecaching(service_number, skc, wkc, ske, wke,acc,ace,datac,datae,demand_helper)  #reset环境

a_temp = np.zeros(service_number).astype(int)
for i in range(service_number):
    a_temp[i] = i
list_temp = list(itertools.combinations(a_temp, 5))  # 这里的5默认为选取5个
array_regret = np.array(list_temp)  # 动作序列

#--------------------------------
# test reward
#----------------------------------
test_demand=np.zeros([30,service_number])
for i in range(30):
    temp_helper=np.random.rand(service_number)
    test_demand[i]=np.random.poisson(temp_helper* 10000000)/10000000
def test_reward():
    previous_choice = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    reward_temp=0
    for i in range(30):
        s=test_demand[i]
        a = dqn.reward_eval_action(s) #动作选取可再改
        r, d, e, acc = caching_environment.step(s, a, previous_choice)
        reward_temp+=r
        previous_choice = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in a:
            previous_choice[i] = 1
    return (reward_temp/30)

#--------------------------------
# 预存数据
#----------------------------------
for i_episode in range(pre_episodes):
    true_demand = caching_environment.setenvironment(pre_dim)
    s=true_demand[0]
    previous_choice=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    done=False
    cur=0
    ep_r = 0
    while True:
        a = dqn.choose_action(s)
        if cur ==(pre_dim-2):
            done = True
        s_=true_demand[cur+1]
        r,d,e,acc=caching_environment.step(s,a,previous_choice)
        dqn.store_transition(s, a, r, s_)
        if done:
            break
        s = s_
        cur=cur+1
        previous_choice=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        for i in a:
            previous_choice[i]=1

#--------------------------------
# 进行预训练
#----------------------------------
r_loss=np.zeros(train_episodes)
for i_episode in range(train_episodes):
    print("begin episode",i_episode)
    true_demand = caching_environment.setenvironment(train_dim)
    s=true_demand[0]
    previous_choice=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
    done=False
    cur=0
    ep_r = 0
    while True:
        a = dqn.choose_action(s)
        if cur ==(train_dim-2):
            done = True
        s_=true_demand[cur+1]
        r,d,e,acc=caching_environment.step(s,a,previous_choice)
        dqn.store_transition(s, a, r, s_)
        if dqn.memory_counter > MEMORY_CAPACITY:
            dqn.learn()
            ep_r += r
        if done:
            r_loss[i_episode] = test_reward()
            print('Ep: ', i_episode, ' |', 'Ep_r: ', r_loss[i_episode])
            break
        s = s_
        cur=cur+1
        previous_choice=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        for i in a:
            previous_choice[i]=1

dqn.storemodel()
plt.plot(range(0,i_episode+1), r_loss, label='Random', ls='--', color='b', marker='v',
             markevery=40, linewidth=1.5)
plt.xlabel('Episode', fontsize=15)
plt.ylabel('Average reward', fontsize=15)
plt.savefig('../picture/training.jpg')
plt.show()


#----------------------------------------------
# 对于对比实验的运行
#------------------------------------------------
def algorithm_run(caching_environment, algorithm, episodes):
    r, d,e, a,re,r_s,d_s,e_s,a_s,re_s = 0.0,0.0, 0.0, 0.0,0.0, [], [], [], [],[]
    cum_regret=0
    cre_regret=[]
    counts = np.zeros(service_number)
    value_delay = np.zeros(service_number)   # 对于延时的定义
    value_energy = np.zeros(service_number)  #对于能耗的定义
    value_accuracy = np.zeros(service_number) #对于准确度的定义
    prev_choice = np.zeros(service_number)   #记录前面一次的缓存选择
    value_reward= np.zeros(service_number)
    cal_regret = np.zeros(episodes)
    for episode in range(episodes):
        true_demand=np.random.rand(service_number)
        expected_reward,true_reward= caching_environment.generate_bandit_data(expected_demand, true_demand)
        # 选取臂
        arm = algorithm.pull(counts, value_reward)   #取从小到大排列的前m个小的
        reward,delay, energy, accuracy = caching_environment.step(true_demand, arm, prev_choice)
        counts_s, value_delay, value_energy, value_accuracy, prev_choice,value_reward = caching_environment.update(true_demand, arm, prev_choice)
        for i in arm:
            counts[i]=counts[i]+1
        r +=reward
        d += delay
        e += energy
        a += accuracy
        index_regret = np.argsort(true_reward)[::-1][0:5]
        for i in range(len(index_regret)):
            re+=index_regret[i]
        re=re/20-reward
        r_s.append(r / (episode + 1)) # average_reawrd
        d_s.append(d / (episode + 1))
        e_s.append(e / (episode + 1))
        a_s.append(a / (episode + 1))
        cum_regret +=re
        re_s.append(re)
        cre_regret.append(cum_regret)
        re=0.0
    return r_s,d_s,e_s,a_s,re_s,cre_regret

def algorithm_dqn(caching_environment, episodes):
    r, d,e, a,re,r_s,d_s,e_s,a_s,re_s = 0.0,0.0, 0.0, 0.0,0.0, [], [], [], [],[]
    cum_regret = 0
    cre_regret = []
    counts = np.zeros(service_number)
    value_delay = np.zeros(service_number)   # 对于延时的定义
    value_energy = np.zeros(service_number)  #对于能耗的定义
    value_accuracy = np.zeros(service_number) #对于准确度的定义
    prev_choice = np.zeros(service_number)   #记录前面一次的缓存选择
    value_reward= np.zeros(service_number)
    cal_regret = np.zeros(episodes)
    new_m = torch.load('dqn.pt').double()
    for episode in range(episodes):
        x=np.random.rand(service_number)
        # true_demand = np.random.poisson(x* 10000000)/10000000
        true_demand = np.random.rand(service_number)
        expected_reward,true_reward= caching_environment.generate_bandit_data(expected_demand, true_demand)
        predict = new_m(torch.from_numpy(true_demand))
        choice = predict.topk(k=5, dim=0, largest=True, sorted=True)
        arm = choice[1].numpy()
        reward,delay, energy, accuracy = caching_environment.step(true_demand, arm, prev_choice)
        counts, value_delay, value_energy, value_accuracy, prev_choice, value_reward = caching_environment.update(
            true_demand, arm, prev_choice)
        # 奖励
        r += reward
        d += delay
        e += energy
        a += accuracy
        index_regret = np.argsort(true_reward)[::-1][0:5]
        for i in range(len(index_regret)):
            re+=index_regret[i]
        re=re/20-reward
        r_s.append(r / (episode + 1))  # average_reawrd
        d_s.append(d / (episode + 1))
        e_s.append(e / (episode + 1))
        a_s.append(a / (episode + 1))
        cum_regret +=re
        re_s.append(re)
        cre_regret.append(cum_regret)
        re = 0.0
        state = true_demand
        s_=np.random.poisson(demand_helper* 10000000)/10000000
        dqn.store_transition(state,arm,reward,s_)
        dqn.learn()
    dqn.storemodel()
    return r_s, d_s, e_s, a_s,re_s,cre_regret

'''
main code
'''
#------------------------------------------
# 定义边缘缓存算法
#-----------------------------------------------
algorithm_random = Random(service_number, choice_number)
algorithm_greedy = EpsilonGreedy(service_number, choice_number, epsilon)
algorithm_cucb = CUCB(service_number, choice_number)  # 涉及选择的个数
algorithm_softmax = Softmax(service_number, choice_number,temperature)  # 涉及选择的个数

reward_random = np.zeros(episodes)
reward_greedy = np.zeros(episodes)
reward_cucb = np.zeros(episodes)
reward_dqn = np.zeros(episodes)
reward_softmax=np.zeros(episodes)

delay_random = np.zeros(episodes)
delay_greedy = np.zeros(episodes)
delay_cucb = np.zeros(episodes)
delay_dqn = np.zeros(episodes)
delay_softmax=np.zeros(episodes)

energy_random=np.zeros(episodes)
energy_softmax=np.zeros(episodes)
energy_greedy = np.zeros(episodes)
energy_cucb = np.zeros(episodes)
energy_dqn = np.zeros(episodes)

accuracy_random = np.zeros(episodes)
accuracy_greedy = np.zeros(episodes)
accuracy_cucb = np.zeros(episodes)
accuracy_dqn = np.zeros(episodes)
accuracy_softmax=np.zeros(episodes)

regret_random=np.zeros(episodes)
regret_greedy=np.zeros(episodes)
regret_cucb=np.zeros(episodes)
regret_dqn=np.zeros(episodes)
regret_softmax=np.zeros(episodes)

cregret_random=np.zeros(episodes)
cregret_greedy=np.zeros(episodes)
cregret_cucb=np.zeros(episodes)
cregret_dqn=np.zeros(episodes)
cregret_softmax=np.zeros(episodes)

for i in range(number_experiments):
    print("Running experiment:", i + 1)
    a,b,c,d,e,f=algorithm_run(caching_environment, algorithm_random, episodes)
    reward_random+=a
    delay_random+=b
    energy_random+=c
    accuracy_random+=d
    regret_random+=e
    cregret_random += f

    a1,b1,c1,d1,e1,f1=algorithm_run(caching_environment, algorithm_greedy, episodes)
    reward_greedy+=a1
    delay_greedy+=b1
    energy_greedy+=c1
    accuracy_greedy+=d1
    regret_greedy += e1
    cregret_greedy += f1

    a2,b2,c2,d2,e2,f2=algorithm_run(caching_environment, algorithm_cucb, episodes)
    reward_cucb+=a2
    delay_cucb+=b2
    energy_cucb+=c2
    accuracy_cucb+=d2
    regret_cucb += e2
    cregret_cucb += f2

    a3,b3,c3,d3,e3,f3=algorithm_dqn(caching_environment,episodes)
    reward_dqn+=a3
    delay_dqn+=b3
    energy_dqn+=c3
    accuracy_dqn+=d3
    regret_dqn += e3
    cregret_dqn += f3

    a4,b4,c4,d4,e4,f4=algorithm_run(caching_environment, algorithm_softmax, episodes)
    reward_softmax+=a4
    delay_softmax+=b4
    energy_softmax+=c4
    accuracy_softmax+=d4
    regret_softmax += e4
    cregret_softmax += f4

#---------------------
# 画图设置
#-------------------
ls_array = ['-', '-.', '--', ':', '-', '-']
marker_array = ['v', 's', '^', 'd', 'o', '.']

# reward
plt.figure(1)
plt.plot(range(episodes), reward_random / number_experiments, label = 'Random', ls='--', color ='b', marker='v', markevery=40, linewidth=1.5)
plt.plot(range(episodes), reward_greedy / number_experiments, label = '$\epsilon$-greedy (0.1)', ls='-.', color ='g', marker='s', markevery=40, linewidth=1.5)
plt.plot(range(episodes), reward_cucb / number_experiments, label = 'UCB', ls='-',  color ='k', marker='^', markevery=40, linewidth=1.5)
plt.plot(range(episodes), reward_dqn / number_experiments, label = 'Dqn', ls=':',  color ='r', marker='d', markevery=40, linewidth=1.5)
plt.plot(range(episodes), reward_softmax / number_experiments, label = 'Softmax', ls='-.',  color ='m', marker='s', markevery=40, linewidth=1.5)
plt.xlabel('Episode', fontsize=15)
plt.ylabel('Reward', fontsize=15)
plt.legend(loc='best', fontsize=12)
plt.savefig('../picture/reward1.jpg')
plt.show()

#dealy
plt.figure(2)
plt.plot(range(episodes), delay_random / number_experiments, label = 'Random', ls='--', color ='b', marker='v', markevery=40, linewidth=1.5)
plt.plot(range(episodes), delay_greedy / number_experiments, label = '$\epsilon$-greedy (0.1)', ls='-.', color ='g', marker='s', markevery=40, linewidth=1.5)
plt.plot(range(episodes), delay_cucb / number_experiments, label = 'UCB', ls='-',  color ='k', marker='^', markevery=40, linewidth=1.5)
plt.plot(range(episodes), delay_dqn / number_experiments, label = 'Dqn', ls=':',  color ='r', marker='d', markevery=40, linewidth=1.5)
plt.plot(range(episodes), delay_softmax / number_experiments, label = 'Softmax', ls='-.',  color ='m', marker='s', markevery=40, linewidth=1.5)
plt.xlabel('Episode', fontsize=15)
plt.ylabel('Delay', fontsize=15)
plt.legend(loc='best', fontsize=12)
plt.savefig('../picture/delay1.jpg')
plt.show()

#energy
plt.figure(3)
plt.plot(range(episodes), energy_random / number_experiments, label = 'Random', ls='--', color ='b', marker='v', markevery=40, linewidth=1.5)
plt.plot(range(episodes), energy_greedy / number_experiments, label = '$\epsilon$-greedy (0.1)', ls='-.', color ='g', marker='s', markevery=40, linewidth=1.5)
plt.plot(range(episodes), energy_cucb / number_experiments, label = 'UCB', ls='-',  color ='k', marker='^', markevery=40, linewidth=1.5)
plt.plot(range(episodes), energy_dqn / number_experiments, label = 'Dqn', ls=':',  color ='r', marker='d', markevery=40, linewidth=1.5)
plt.plot(range(episodes), energy_softmax / number_experiments, label = 'Softmax', ls='-.',  color ='m', marker='s', markevery=40, linewidth=1.5)
plt.xlabel('Episode', fontsize=15)
plt.ylabel('Energy', fontsize=15)
plt.legend(loc='best', fontsize=12)
plt.savefig('../picture/energy1.jpg')
plt.show()
plt.close()

#accurcy
plt.figure(4)
plt.plot(range(episodes), accuracy_random / number_experiments, label = 'Random', ls='--', color ='b', marker='v', markevery=40, linewidth=1.5)
plt.plot(range(episodes), accuracy_greedy / number_experiments, label = '$\epsilon$-greedy (0.1)', ls='-.', color ='g', marker='s', markevery=40, linewidth=1.5)
plt.plot(range(episodes), accuracy_cucb / number_experiments, label = 'UCB', ls='-',  color ='k', marker='^', markevery=40, linewidth=1.5)
plt.plot(range(episodes), accuracy_dqn / number_experiments, label = 'Dqn', ls=':',  color ='r', marker='d', markevery=40, linewidth=1.5)
plt.plot(range(episodes), accuracy_softmax / number_experiments, label = 'Softmax', ls='-.',  color ='m', marker='s', markevery=40, linewidth=1.5)
plt.xlabel('Episode', fontsize=15)
plt.ylabel('Accuracy', fontsize=15)
plt.legend(loc='best', fontsize=12)
plt.savefig('../picture/accuracy1.jpg')
plt.show()

plt.figure(5)
plt.plot(range(episodes), regret_random / number_experiments, label = 'Random', ls='--', color ='b', marker='v', markevery=40, linewidth=1.5)
plt.plot(range(episodes), regret_greedy / number_experiments, label = '$\epsilon$-greedy (0.1)', ls='-.', color ='g', marker='s', markevery=40, linewidth=1.5)
plt.plot(range(episodes), regret_cucb / number_experiments, label = 'UCB', ls='--',  color ='k', marker='^', markevery=40, linewidth=1.5)
plt.plot(range(episodes), regret_dqn / number_experiments, label = 'Dqn', ls=':',  color ='r', marker='d', markevery=40, linewidth=1.5)
plt.plot(range(episodes), regret_softmax / number_experiments, label = 'Softmax', ls='-.',  color ='m', marker='s', markevery=40, linewidth=1.5)
plt.xlabel('Episode', fontsize=15)
plt.ylabel('Regret', fontsize=15)
plt.legend(loc='best', fontsize=12)
plt.savefig('../picture/regret1.jpg')
plt.show()

plt.figure(6)
plt.plot(range(episodes), cregret_random / (number_experiments), label = 'Random', ls='--', color ='b', marker='v', markevery=40, linewidth=1.5)
plt.plot(range(episodes), cregret_greedy / (number_experiments), label = '$\epsilon$-greedy (0.1)', ls='-.', color ='g', marker='s', markevery=40, linewidth=1.5)
plt.plot(range(episodes), cregret_cucb / (number_experiments), label = 'UCB', ls='--',  color ='k', marker='^', markevery=40, linewidth=1.5)
plt.plot(range(episodes), cregret_dqn / (number_experiments), label = 'Dqn', ls=':',  color ='r', marker='d', markevery=40, linewidth=1.5)
plt.plot(range(episodes), cregret_softmax / (number_experiments), label = 'Softmax', ls='-.',  color ='m', marker='s', markevery=40, linewidth=1.5)
plt.xlabel('Episode', fontsize=15)
plt.ylabel('cumulative_regret', fontsize=15)
plt.legend(loc='best', fontsize=12)
plt.savefig('../picture/cumulative_regret1.jpg')
plt.show()