

import tensorflow as tf
import numpy as np
import gym
import time
import matplotlib.pyplot as plt


#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
LR_D = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 10000
BATCH_SIZE = 32

RENDER = True
ENV_NAME = 'Pendulum-v0'

###############################  DDPG  ####################################

class DDPG(object):
    def __init__(self, a_dim, s_dim, a_bound,):
        self.memory = np.zeros((MEMORY_CAPACITY, s_dim * 2 + a_dim + 1), dtype=np.float32)
        self.pointer = 0
        self.sess = tf.Session()

        self.a_dim, self.s_dim, self.a_bound = a_dim, s_dim, a_bound,
        self.S = tf.placeholder(tf.float32, [None, s_dim], 's')
        self.S_ = tf.placeholder(tf.float32, [None, s_dim], 's_')
        self.R = tf.placeholder(tf.float32, [None, 1], 'r')

        with tf.variable_scope('Actor'):
            self.a = self._build_a(self.S, scope='eval', trainable=True)
            a_ = self._build_a(self.S_, scope='target', trainable=False)
        with tf.variable_scope('Critic'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            q = self._build_c(self.S, self.a, scope='eval', trainable=True)
            q_ = self._build_c(self.S_, a_, scope='target', trainable=False)
        
        with tf.variable_scope('Dynamics'):
            # assign self.a = a in memory when calculating q for td_error,
            # otherwise the self.a is from Actor when updating Actor
            self.s_hat = self._build_dyn(self.S, self.a, scope='eval', trainable=True)

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        self.dy_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Dynamics/eval')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU) * t + TAU * e)
                             for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        d_loss = tf.losses.mean_squared_error(labels=self.S_, predictions=self.s_hat)

        # r2 = tf.contrib.layers.l2_regularizer(scale=0.1)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_constant = 0.01  # Choose an appropriate one.
        d_loss1 = d_loss + reg_constant * sum(reg_losses)
        
        """
        r2 = tf.contrib.layers.l2_regularizer(scale=0.1)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        reg_constant = 0.01  # Choose an appropriate one.
        loss = my_normal_loss + reg_constant * sum(reg_losses)
        """

        self.dtrain = tf.train.AdamOptimizer(LR_D).minimize(d_loss1, var_list=self.dy_params)

        self.sess.run(tf.global_variables_initializer())

    def choose_action(self, s):
        return self.sess.run(self.a, {self.S: s[np.newaxis, :]})[0]

    def dynamics_train(self):
        # print(s.shape, a.shape, s_.shape)
        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        bs_ = bt[:, -self.s_dim:]
        self.sess.run(self.dtrain, {self.S_: bs_, self.S: bs, self.a: ba})

    def predict_shat(self, s, a):
        return self.sess.run(self.s_hat, {self.S: s[np.newaxis, :], self.a: a[np.newaxis, :]})

    def learn(self):
        # soft target replacement
        self.sess.run(self.soft_replace)

        indices = np.random.choice(MEMORY_CAPACITY, size=BATCH_SIZE)
        bt = self.memory[indices, :]
        bs = bt[:, :self.s_dim]
        ba = bt[:, self.s_dim: self.s_dim + self.a_dim]
        br = bt[:, -self.s_dim - 1: -self.s_dim]
        bs_ = bt[:, -self.s_dim:]

        self.sess.run(self.atrain, {self.S: bs})
        self.sess.run(self.ctrain, {self.S: bs, self.a: ba, self.R: br, self.S_: bs_})

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % MEMORY_CAPACITY  # replace the old memory with new memory
        self.memory[index, :] = transition
        self.pointer += 1

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 30, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_dyn(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            nl1 = 30
            r2 = tf.contrib.layers.l2_regularizer(scale=0.1)
            r22 = tf.contrib.layers.l2_regularizer(scale=0.01)
            # net_s = tf.layers.dense(s, 50, activation=tf.nn.relu, name='l_s', trainable=trainable)
            # net_a = tf.layers.dense(a, 50, activation=tf.nn.relu, name='l_a', trainable=trainable)
            wd_s = tf.get_variable('wd_s', [self.s_dim, nl1], trainable=trainable, regularizer=r2)
            wd_a = tf.get_variable('wd_a', [self.a_dim, nl1], trainable=trainable, regularizer=r2)
            bd   = tf.get_variable('bd', [1, nl1], trainable=trainable, regularizer=r2)
            net_sa = tf.nn.relu(tf.matmul(s, wd_s) + tf.matmul(a, wd_a) + bd)
            net_sa = tf.layers.dense(net_sa, s_dim, activation=tf.nn.relu, name='l_sa', trainable=trainable, kernel_regularizer=r22, bias_regularizer=r22)
            # a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return net_sa



    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 30
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

###############################  training  ####################################

env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

ddpg = DDPG(a_dim, s_dim, a_bound)

r_store = []
d_store_mean = []
d_store_var = []
d_losses = []
var = 3  # control exploration
count = 0;
total_sample = 0;
t1 = time.time()
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    # for k in range(MAX_EP_STEPS):
    #     a = ddpg.choose_action(s)
    #     a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
    #     s_, r, done, info = env.step(a)
    #     ddpg.dynamics_train(s, a, s_)
    #     s = s_

    for j in range(MAX_EP_STEPS):
        total_sample += 1
        # if RENDER:
        #     env.render()

        # Add exploration noise
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        s_, r, done, info = env.step(a)

        ## train dynamics model
        s_hat = ddpg.predict_shat(s, a)
        s_hat = s_hat.reshape(-1)
        d_loss = np.linalg.norm(s_hat - s_)
        d_losses.append(d_loss)
        # ddpg.dynamics_train(s, a, s_)

        ## train dynamics model
        # print(s_hat.reshape(-1).shape, s_.shape)
        # if i > MAX_EPISODES/2:
        #     ddpg.store_transition(s, a, r / 10, s_hat)        
        # else:
        #     ddpg.store_transition(s, a, r / 10, s_)

        ddpg.store_transition(s, a, r, s_)
        epsilon = 0.5
        drop_rate = 20
        power_fun = 0.9
        # eps_ = 0
        eps_ = epsilon* (power_fun**((1+i)/drop_rate))
        if(d_loss < eps_): # power loss function
        # if(d_loss < 0.5):
            ddpg.store_transition(s, a, r / 10, s_hat)
            count = count+1

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            ddpg.dynamics_train()
            ddpg.learn()

        s = s_
        ep_reward += r
        if j == MAX_EP_STEPS-1:
            print('Episode:', i, ' Reward: %04i' % int(ep_reward), 'Epsilon: %.3f' % eps_,'Explore: %.2f' % var, 'D_Mean: %.4f' %np.mean(d_losses), 'D_Var: %.4f' %np.var(d_losses))
            r_store.append(ep_reward)
            d_store_mean.append(np.mean(d_losses))
            d_store_var.append(np.var(d_losses))
            d_losses = []
            # if ep_reward > -300:RENDER = True
            break
print('Running time: ', time.time() - t1)
print("total used sample: ", total_sample)
print("total augmented sample: ", count)

fig, ax = plt.subplots(1,2,figsize=(14,5))
ax[0].plot(r_store)
ax[1].plot(range(len(r_store)), d_store_mean, label='mean')
# ax[1].plot(range(len(r_store)), d_store_var, label='var')
ax[1].fill_between(range(len(r_store)), np.array(d_store_mean) - np.array(d_store_var), np.array(d_store_mean) + np.array(d_store_var), color='gray', alpha=0.2)
np.save('new_exp/w_reward.npy', r_store)
fig.savefig('new_exp/results122.png')


######### prev
r_prev = np.load('new_exp/wo_reward.npy')
fig, ax = plt.subplots(1,2,figsize=(14,5))
ax[0].plot(r_store, 'b', label='with vi')
ax[0].plot(r_prev,'r', label='without vi')
ax[0].legend()

height = [40000, 40000 - count]
bars = ["without vi", "with vi"]
y_pos = np.arange(len(bars))
ax[1].bar(y_pos, height)
ax[1].xticks(y_pos, bars)

fig.savefig('new_exp/results133.png')



# Running time:  69.9507315158844
# total used sample:  40000
# total augmented sample:  14407
