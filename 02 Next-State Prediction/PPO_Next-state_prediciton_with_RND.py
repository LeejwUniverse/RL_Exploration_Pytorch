import gym
import torch
import torch.nn as nn ## NN 사용을 위해서
import torch.nn.functional as F ## 활성함수 사용을 위해서
import torch.optim as optim ## adam과 같은 학습 최적화 알고리즘 사용을 위해서
from torch.distributions import Normal ## 분포 관련
from torch.distributions import Categorical

import numpy as np ## numpy 사용
import time
import pickle
from collections import deque ## deque 자료구조 사용. 파이썬에서는 list와 deque 속도 차이가 큽니다

class Actor_network(nn.Module):
    def __init__(self, obs_size, action_size):
        super(Actor_network, self).__init__()
        self.fc1 = nn.Linear(obs_size,64) ## input observation
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,64)
        self.fc4 = nn.Linear(64,action_size) ## output each action

    def forward(self,x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.fc4(x) 
        action_distribution = F.softmax(x, dim=-1) ## ouput is action distribution.
       
        return action_distribution
    
class Critic_network(nn.Module):
    def __init__(self, obs_size):
        super(Critic_network, self).__init__()
        self.fc1 = nn.Linear(obs_size,64)
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,64)
        self.fc_ex_v = nn.Linear(64,1) ## output extrinsic value
        self.fc_in_v = nn.Linear(64,1) ## output intrinsic value

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        ex_value = self.fc_ex_v(x)
        in_value = self.fc_in_v(x)

        return ex_value, in_value

class Predictor_network(nn.Module):
    def __init__(self, obs_size):
        super(Predictor_network, self).__init__()
        self.fc1 = nn.Linear(obs_size, 64) # input predictor: action + obs | target: next_obs
        self.fc2 = nn.Linear(64,64)
        self.fc3 = nn.Linear(64,64)
        self.fc4 = nn.Linear(64,1) ## output f(st+1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        f_st_1 = self.fc4(x)

        return f_st_1

def get_action(action_distribution): 
    distribution = Categorical(action_distribution) ## pytorch categorical 함수는 array에 담겨진 값들을 확률로 정해줍니다.
    action = distribution.sample().item()
    
    return action
    
def GAE(reward, mask, value, gamma): ## Reinforce 알고리즘을 보면 G라는 명칭이 있는데.
                                     ## 그 부분을 구해주는 함수입니다.
   
    Target_G = torch.zeros_like(reward) ## batch 사이즈만큼 0으로 채워진 G 배열을 만듭니다. 그냥 0으로 채워진 배열하나 만들어 둔거에요.
    Advantage = torch.zeros_like(reward)

    lmbda = 0.95
    
    next_value = 0
    Return = 0
    advantage = 0
    
    for t in reversed(range(0,len(reward))): # GAE와 RETURN 계산.
        Return = reward[t] + gamma * Return * mask[t]
        Target_G[t] = Return ## return이 critic loss에 쓰이는 V_target에 해당하는 값이다.
        
        delta = reward[t] + gamma * next_value * mask[t] - value.data[t]
        next_value = value.data[t]
        
        advantage = delta + gamma* lmbda * advantage * mask[t]
        Advantage[t] = advantage    
        
    return Target_G, Advantage
    
def surrogate_loss(actor, old_policy, Advantage, obs, action): # surrogate loss 계산.
    action_distribution = actor(torch.Tensor(obs))
    policy = action_distribution.gather(1,action)
    log_policy = torch.log(policy)
    log_old_policy = torch.log(old_policy)
    
    distribution = Categorical(action_distribution)
    entropy = distribution.entropy()
    
    ratio = torch.exp(log_policy - log_old_policy)
    
    ratio_A = ratio * Advantage

    return ratio, ratio_A, entropy

def observation_normalize(observation, obs_std): # In order to normalize observation
    obs_mean = np.mean(observation, 0)
    observation = (observation - obs_mean) / obs_std
    return observation

def get_MSE(predictor, target_predictor, observation, onehot_action, next_observation, obs_std):
    MSE = torch.nn.MSELoss() # define loss function.

    obs = torch.Tensor(observation_normalize(observation, obs_std))
    obs = torch.clamp(obs, -5, 5)
    next_obs = torch.Tensor(observation_normalize(next_observation, obs_std)) # normalize observation.
    next_obs = torch.clamp(next_obs, -5, 5)

    obs_action = torch.cat((obs, onehot_action), -1)

    f_prime = predictor(obs_action) # predictor | input is [obs, onehot_action]
    f = target_predictor(next_obs) # target predictor (fixed!) | input is [next_obs]

    return MSE(f_prime, f.detach())

def train(actor, critic, predictor, target_predictor, trajectories, actor_optimizer, critic_optimizer, predictor_optimizer, T_horizon, batch_size, epoch, obs_std, action_size):
    
    c_1 = 1
    c_2 = 0.0 # entropy bonus is zero, because we are using RND which is exploration technique.
    eps = 0.2
    
    trajectories = np.array(trajectories) ##deque인 type을 np.array로 바꿔 줍니다.
    obs = np.vstack(trajectories[:, 0]) 
    action = list(trajectories[:, 1])
    onehot_action = np.vstack(trajectories[:, 2])
    next_obs = np.vstack(trajectories[:, 3])
    extrinsic_reward = list(trajectories[:, 4])
    intrinsic_reward = list(trajectories[:, 5])
    mask = list(trajectories[:, 6])
    old_policy = np.vstack(trajectories[:, 7])
    
    obs = torch.Tensor(obs)
    action = torch.LongTensor(action).unsqueeze(1)
    onehot_action = torch.Tensor(onehot_action)
    next_obs = torch.Tensor(next_obs)
    extrinsic_reward = torch.Tensor(extrinsic_reward)
    intrinsic_reward = torch.Tensor(intrinsic_reward)
    mask = torch.Tensor(mask)
    old_policy = torch.Tensor(old_policy)
 
    """ update predictor """
    predictor_loss = get_MSE(predictor, target_predictor, obs.numpy(), onehot_action, next_obs.numpy(), obs_std)
    predictor_optimizer.zero_grad()
    predictor_loss.backward()
    predictor_optimizer.step()

    obs_std = torch.std(obs, 0).numpy()

    intrinsic_reward = intrinsic_reward / torch.std(intrinsic_reward) # normalize intrinsic reward.
      
    """ calculate Return and Advantage """
    ex_value, in_value = critic(obs)
    ex_Return, ex_Advantage = GAE(extrinsic_reward, mask, ex_value, 0.999) # best gamma, refer to fig 5.
    in_Return, in_Advantage = GAE(intrinsic_reward, mask, in_value, 0.99) # best gamma, refer to fig 5.
    
    total_Return = ex_Return + in_Return
    Advantage = ex_Advantage + in_Advantage
    
    mse_loss = torch.nn.MSELoss()

    r_batch = np.arange(len(obs))
    for i in range(epoch):
        np.random.shuffle(r_batch)
        
        for j in range(T_horizon//batch_size): ##2048/64  0 ~ 31
     
            mini_batch = r_batch[batch_size * j: batch_size * (j+1)]
            mini_batch = torch.LongTensor(mini_batch)
            
            obs_b = obs[mini_batch]
            action_b = action[mini_batch]
            
            total_Return_b = total_Return[mini_batch].detach()
          
            """ critic loss """
            ex_value, in_value = critic(obs_b)
            value = ex_value + in_value
            critic_loss = mse_loss(value.squeeze(1), total_Return_b).mean()
            
            """ actor loss """   
            old_policy_b = old_policy[mini_batch].detach()
            
            Advantage_b = Advantage[mini_batch].unsqueeze(1)
                        
            ratio, L_CPI, entropy = surrogate_loss(actor, old_policy_b, Advantage_b, obs_b, action_b)

            entropy = entropy.mean()
            clipped_surrogate = torch.clamp(ratio,1-eps,1+eps) * Advantage_b
            actor_loss = -torch.min(L_CPI.mean(),clipped_surrogate.mean())

            """ total loss """        
            L_CLIP= actor_loss + c_1 * critic_loss - c_2 * entropy

            actor_optimizer.zero_grad()
            L_CLIP.backward(retain_graph=True)
            actor_optimizer.step()

            critic_optimizer.zero_grad()
            L_CLIP.backward() 
            critic_optimizer.step()

    return obs_std


def main():
    
    episode = 10001
    maximum_steps = 300

    env = gym.make('CartPole-v1') ## 환경 정의
    
    obs_size = 4
    action_size = 2
    actor = Actor_network(obs_size, action_size) ## actor 생성
    critic = Critic_network(obs_size) ## critic 생성
    predictor = Predictor_network(obs_size + action_size) ## predictor 생성
    target_predictor = Predictor_network(obs_size) ## target predictor 생성

    learning_rate = 0.0003
    batch_size = 32
    epoch = 4
    T_horizon = 4096 

    print_interval = 100 ## 출력 
    actor_optimizer = optim.Adam(actor.parameters(), lr=learning_rate) ## actor에 대한 optimizer Adam으로 설정하기.
    critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate) ## critic에 대한 optimizer Adam으로 설정하기.
    predictor_optimizer = optim.Adam(predictor.parameters(), lr=learning_rate) ## predictor에 대한 optimizer Adam으로 설정하기.
    step = 0 ## 총 step을 계산하기 위한 step.
    ex_score = 0
    in_score = 0
    save_ex_score = []
    save_in_score = []
    print(env.observation_space) ## CartPole 환경에 대한 state 사이즈
    print(env.action_space) ## CartPole 환경에 대한 state 사이즈

    """ calculate initial observation std """
    store_observation = []
    M = 1000
    init_cnt = 0
    while True:
        _ = env.reset()
        while True: # observation normalize를 위한 단계.
            action = np.random.randint(action_size) # uniform random action.
            next_obs, reward, done, info = env.step(action)
            store_observation.append(next_obs)
            init_cnt +=1
            if init_cnt == M:
                break
            if done:
                break
        if init_cnt == M:
            break
    obs_std = np.std(store_observation, 0)

    """ online learning """
    trajectories = deque() ## (s,a,r,done 상태) 를 저장하는 history or trajectories라고 부름. 학습을 위해 저장 되어지는 경험 메모리라고 보면됨.
    
    for epi in range(episode): # episode.
        obs = env.reset() ## initialize environment

        for i in range(maximum_steps):
            #env.render() ## 게임을 실시간으로 실행하는 명령

            action_distribution = actor(torch.Tensor(obs)) ##actor로 부터 action에 대한 softmax distribution을 얻는다.
            action = get_action(action_distribution) ## sampling action.
            old_policy = action_distribution[action] # 실제 선택된 action의 확률 값을 loss 계산을 위해 저장한다.

            next_obs, extrinsic_reward, done, info = env.step(action) ## 선택된 가장 높은 action이 다음 step에 들어감.
            onehot_action = torch.zeros(action_size)
            onehot_action[action] = 1.0

            intrinsic_reward = get_MSE(predictor, target_predictor, obs, onehot_action, next_obs, obs_std).item() # || f'(st+1) - f(st+1) ||^2

            mask = 0 if done else 1 ## 게임이 종료됬으면, done이 1이면 mask = 0 생존유무 확인.
                
            trajectories.append((obs, action, onehot_action, next_obs, extrinsic_reward, intrinsic_reward, mask, old_policy.detach().numpy()))
            obs = next_obs # current observation을 next_observation으로 변경
            
            ex_score += extrinsic_reward # extrinsic_reward 갱신.
            in_score += intrinsic_reward # intrinsic_reward 갱신.
            step += 1

            if step % T_horizon == 0 and step != 0:
                obs_std = train(actor, critic, predictor, target_predictor, trajectories, actor_optimizer, critic_optimizer, predictor_optimizer, T_horizon, batch_size, epoch, obs_std, action_size) ## 본격적인 학습을 위한 train 함수.
                trajectories = deque() ## (s,a,r,done 상태) 를 저장하는 history or trajectories라고 부름. 학습을 위해 저장 되어지는 경험 메모리라고 보면됨.
                
            if done: ## 죽었다면 게임 초기화를 위한 반복문 탈출
                break
      
        if epi % print_interval == 0 and epi != 0:
            save_ex_score.append(ex_score/print_interval) ## reward score 저장.
            save_in_score.append(in_score/print_interval)
            print('episode: ', epi,' step: ', step, 'ex_score: ', ex_score/print_interval, 'in_score: ', in_score/print_interval) # log 출력.
            ex_score = 0
            in_score = 0
            with open('NSP_RND_ex.p', 'wb') as file:
                pickle.dump(save_ex_score, file)
            with open('NSP_RND_in.p', 'wb') as file:
                pickle.dump(save_in_score, file)
        
    env.close() ## 모든 학습이 끝나면 env 종료.

if __name__ == '__main__':
    main()
    
