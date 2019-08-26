

import tensorflow as tf
import numpy as np
import gym
import time
import matplotlib.pyplot as plt
from cont_cart_pole import *

#####################  hyper parameters  ####################

MAX_EPISODES = 200
MAX_EP_STEPS = 200
LR_A = 0.001    # learning rate for actor
LR_C = 0.002    # learning rate for critic
LR_D = 0.001    # learning rate for critic
GAMMA = 0.9     # reward discount
TAU = 0.01      # soft replacement
MEMORY_CAPACITY = 5000
BATCH_SIZE = 100

RENDER = False
# ENV_NAME = 'Pendulum-v0'
# ENV_NAME = 'CartPole-v0' #discrete
# ENV_NAME = 'MountainCar-v0' #discrete
# ENV_NAME = 'Acrobot-v1'
ENV_NAME = 'MountainCarContinuous-v0'

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

        ################################################################################################################
        with tf.variable_scope('Encoder'):
            [self.z_mu, self.z_logsigma] = self._build_enc(self.s, scope='eval', trainable=True)
        
        self.z_samples = sample_z(BATCH_SIZE, self.z_mu, self.z_logsigma)

        with tf.variable_scope('Decoder'):
            [self.x_mu, self.x_logsigma] = self._build_dec(self.z_samples, scope='eval', trainable=True)

        ################################################################################################################

        # networks parameters
        self.ae_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
        self.at_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target')
        self.ce_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
        self.ct_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')

        self.dy_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Dynamics/eval')
        self.enc_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Encoder/eval')
        self.dec_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Decoder/eval')

        # target net replacement
        self.soft_replace = [tf.assign(t, (1 - TAU)*t + TAU*e) for t, e in zip(self.at_params + self.ct_params, self.ae_params + self.ce_params)]

        q_target = self.R + GAMMA * q_
        # in the feed_dic for the td_error, the self.a should change to actions in memory
        self.td_error = tf.losses.mean_squared_error(labels=q_target, predictions=q)
        self.ctrain = tf.train.AdamOptimizer(LR_C).minimize(self.td_error, var_list=self.ce_params)

        a_loss = - tf.reduce_mean(q)    # maximize the q
        self.atrain = tf.train.AdamOptimizer(LR_A).minimize(a_loss, var_list=self.ae_params)

        d_loss = tf.losses.mean_squared_error(labels=self.S_, predictions=self.s_hat)

        # r2 = tf.contrib.layers.l2_regularizer(scale=0.1)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='Dynamics/eval')
        reg_constant = 0.01  # Choose an appropriate one.
        d_loss1 = d_loss + reg_constant * sum(reg_losses)
        
        ################################################################################################################
        # log-likelihood p(x/z) = log(sig^2) + (x - mu)^2 / 2sig^2
        reconstruct_loss = tf.reduce_sum(0.5* self.x_logsigma + (tf.square(self.S - self.x_mu) / (2.0*tf.exp(self.x_logsigma))), 1)
        # KL-div loss = Dkl(q(z/x)||p(z)) = Dkl(N(mu, sigma) || N(0,1))
        # Dkl = trace(sigma) + mu*mu - k -log(det(sigma))
        # Dkl = exp(log(sigma)) + mu*mu - 1 -log(sigma)
        dkl_loss = -0.5 * tf.reduce_sum(-tf.exp(self.z_logsigma) - tf.square(self.z_mu) + 1 + self.z_logsigma, 1)
        self.vae_loss = tf.reduce_mean(reconstruct_loss + dkl_loss)
        

        ################################################################################################################
        enc_reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='Encoder/eval')
        dec_reg_loss = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES, scope='Decoder/eval')
        reg_constant = 1  # Choose an appropriate one.
        self.vae_loss_with_reg = self.vae_loss + reg_constant * sum(enc_reg_loss) + reg_constant * sum(dec_reg_loss)
        ################################################################################################################
        

        # self.dec_train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.vae_loss, var_list=self.dec_params)
        # self.enc_train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.vae_loss, var_list=self.enc_params)
        # self.vae_train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.vae_loss, var_list=[self.dec_params, self.enc_params])
        self.vae_train = tf.train.AdamOptimizer(learning_rate=0.01).minimize(self.vae_loss_with_reg, var_list=[self.dec_params, self.enc_params])

        ################################################################################################################
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

    def get_td_error(self, s, a, r, s_):
        r = np.array([r])
        # print(s.shape, a.shape, r.shape, s_.shape)
        return self.sess.run(self.td_error, {self.S: s[np.newaxis, :], self.a: a[np.newaxis, :], self.R: r[np.newaxis, :], self.S_: s_[np.newaxis, :]})


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

    def _build_dyn(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            nl1 = 10
            r2 = tf.contrib.layers.l2_regularizer(scale=0.1)
            r22 = tf.contrib.layers.l2_regularizer(scale=0.01)
            wd_s = tf.get_variable('wd_s', [self.s_dim, nl1], trainable=trainable, regularizer=r2)
            wd_a = tf.get_variable('wd_a', [self.a_dim, nl1], trainable=trainable, regularizer=r2)
            bd   = tf.get_variable('bd', [1, nl1], trainable=trainable, regularizer=r2)
            net_sa = tf.nn.relu(tf.matmul(s, wd_s) + tf.matmul(a, wd_a) +bd)
            net_sa = tf.layers.dense(net_sa, s_dim, activation=tf.nn.sigmoid, name='l_sa', trainable=trainable, kernel_regularizer=r22, bias_regularizer=r22)
            # a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return net_sa

################################################################################################################################
    def _build_enc(self, s, scope, trainable):
        with tf.variable_scope(scope):
            nz = 1
            nx = 2
            enh1 = 10
            enh2 = 10

            r2 = tf.contrib.layers.l2_regularizer(scale=0.001)
            ew1 = tf.get_variable('ew1', tf.truncated_normal(shape=[nx,enh1], stddev=0.1), trainable=trainable, regularizer=r2)
            ew2 = tf.get_variable('ew2',tf.truncated_normal(shape=[enh1,enh2], stddev=0.1), trainable=trainable, regularizer=r2)
            ew3 = tf.get_variable('ew3',tf.truncated_normal(shape=[enh2,nz], stddev=0.1), trainable=trainable, regularizer=r2)
            ew4 = tf.get_variable('ew4',tf.truncated_normal(shape=[enh2,nz], stddev=0.1), trainable=trainable, regularizer=r2)

            eb1 = tf.get_variable('eb1',tf.constant(0.1, shape=[enh1]), trainable=trainable, regularizer=r2)
            eb2 = tf.get_variable('eb2',tf.constant(0.1, shape=[enh2]), trainable=trainable, regularizer=r2)
            eb3 = tf.get_variable('eb3',tf.constant(0.1, shape=[nz]), trainable=trainable, regularizer=r2)
            eb4 = tf.get_variable('eb4',tf.constant(0.1, shape=[nz]), trainable=trainable, regularizer=r2)
            # net_sa = tf.nn.relu(tf.matmul(s, wd_s) + tf.matmul(a, wd_a) +bd)
            # net_sa = tf.layers.dense(net_sa, s_dim, activation=tf.nn.sigmoid, name='l_sa', trainable=trainable, kernel_regularizer=r22, bias_regularizer=r22)

            eh1 = tf.nn.softplus(tf.matmul(s, ew1) + eb1)
            eh2 = tf.nn.softplus(tf.matmul(eh1, ew2) + eb2)

            # z ~ N(mu, sigma)
            z_mu = tf.add(tf.matmul(eh2, ew3), eb3)
            z_logsigma = tf.add(tf.matmul(eh2, ew4), eb4)

            return [z_mu, z_logsigma]

    def sample_z(self, batch_size=500, mu, logsigma):
        """z = sample_z(batch_size, z_mu, z_logsigma)"""
        eps = tf.random_normal(shape=[batch_size, nz], mean=0, stddev=1, dtype=tf.float32, seed=1337)
        logsigma = tf.sqrt(tf.exp(logsigma))
        z = tf.add(mu, tf.math.multiply(logsigma, eps))
        return z

    def _build_dec(self, z, scope, trainable):
        with tf.variable_scope(scope):
            nz = 1
            nx = 2
            dnh1 = 10
            dnh2 = 10

            r2 = tf.contrib.layers.l2_regularizer(scale=0.001)
            dw1 = tf.get_variable('dw1', tf.truncated_normal(shape=[nz,dnh1], stddev=0.1), trainable=trainable, regularizer=r2)
            dw2 = tf.get_variable('dw2',tf.truncated_normal(shape=[dnh1,dnh2], stddev=0.1), trainable=trainable, regularizer=r2)
            dw3 = tf.get_variable('dw3',tf.truncated_normal(shape=[dnh2,nx], stddev=0.1), trainable=trainable, regularizer=r2)
            dw4 = tf.get_variable('dw4',tf.truncated_normal(shape=[dnh2,nx], stddev=0.1), trainable=trainable, regularizer=r2)

            db1 = tf.get_variable('db1',tf.constant(0.1, shape=[dnh1]), trainable=trainable, regularizer=r2)
            db2 = tf.get_variable('db2',tf.constant(0.1, shape=[dnh2]), trainable=trainable, regularizer=r2)
            db3 = tf.get_variable('db3',tf.constant(0.1, shape=[nx]), trainable=trainable, regularizer=r2)
            db4 = tf.get_variable('db4',tf.constant(0.1, shape=[nx]), trainable=trainable, regularizer=r2)
            # net_sa = tf.nn.relu(tf.matmul(s, wd_s) + tf.matmul(a, wd_a) +bd)
            # net_sa = tf.layers.dense(net_sa, s_dim, activation=tf.nn.sigmoid, name='l_sa', trainable=trainable, kernel_regularizer=r22, bias_regularizer=r22)

            dh1 = tf.nn.softplus(tf.matmul(z, dw1) + db1)
            dh2 = tf.nn.softplus(tf.matmul(dh1, dw2) + db2)

            x_mu = tf.add(tf.matmul(dh2, dw3), db3)
            x_logsigma = tf.add(tf.matmul(dh2, dw4), db4)

            return [x_mu, x_logsigma]
################################################################################################################################

    def _build_a(self, s, scope, trainable):
        with tf.variable_scope(scope):
            net = tf.layers.dense(s, 10, activation=tf.nn.relu, name='l1', trainable=trainable)
            a = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, name='a', trainable=trainable)
            return tf.multiply(a, self.a_bound, name='scaled_a')

    def _build_c(self, s, a, scope, trainable):
        with tf.variable_scope(scope):
            n_l1 = 10
            w1_s = tf.get_variable('w1_s', [self.s_dim, n_l1], trainable=trainable)
            w1_a = tf.get_variable('w1_a', [self.a_dim, n_l1], trainable=trainable)
            b1 = tf.get_variable('b1', [1, n_l1], trainable=trainable)
            net = tf.nn.relu(tf.matmul(s, w1_s) + tf.matmul(a, w1_a) + b1)
            return tf.layers.dense(net, 1, trainable=trainable)  # Q(s,a)

###############################  training  ####################################
# env = ContinuousCartPoleEnv()
env = gym.make(ENV_NAME)
env = env.unwrapped
env.seed(1)

print(env.observation_space.shape, env.action_space)
s_dim = env.observation_space.shape[0]
a_dim = env.action_space.shape[0]
a_bound = env.action_space.high

ddpg = DDPG(a_dim, s_dim, a_bound)

r_store = []
d_store_mean = []
d_store_var = []
td_store_mean = []
td_store_var = []
d_losses = []
td_errores = []
var = 3  # control exploration
count = 0;
total_sample = 0;
t1 = time.time()
MAX_EPISODES = 100
for i in range(MAX_EPISODES):
    s = env.reset()
    ep_reward = 0
    # for k in range(MAX_EP_STEPS):
    #     a = ddpg.choose_action(s)
    #     a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
    #     s_, r, done, info = env.step(a)
    #     ddpg.dynamics_train(s, a, s_)
    #     s = s_
    done = False
    ep = 0;
    while not done or ep != 5000:
    # for j in range(MAX_EP_STEPS):
        total_sample += 1
        # if RENDER:
        #     env.render()

        # Add exploration noise
        a = ddpg.choose_action(s)
        a = np.clip(np.random.normal(a, var), -2, 2)    # add randomness to action selection for exploration
        s_, r, done, info = env.step(a)
        # print(r, end=" ")
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
        
        ## td error caluclation
        # td_err = ddpg.get_td_error(s, a, r, s_)
        # print(td_err)
        # td_errores.append(td_err)

        ddpg.store_transition(s, a, r, s_)
        if(d_loss < 0.5):
            ddpg.store_transition(s, a, r , s_hat)
            count = count+1

        if ddpg.pointer > MEMORY_CAPACITY:
            var *= .9995    # decay the action randomness
            ddpg.dynamics_train()
            ddpg.learn()

        # print(count)
        s = s_
        ep_reward += r
        # if j == MAX_EP_STEPS-1:
        # print(ep)
        ep = ep + 1
        if done or ep == 4999:
            print('Episode:', i, ' Reward: %i' % int(ep_reward), 'D_Mean: %.4f' %np.mean(d_losses), 'D_Var: %.4f' %np.var(d_losses))
            # print('Episode:', i, ' Reward: %i' % int(ep_reward), 'td-err: %.4f' % np.mean(td_errores), 'td-var: %.4f' % np.var(td_errores), 'D_Mean: %.4f' %np.mean(d_losses), 'D_Var: %.4f' %np.var(d_losses))
            r_store.append(ep_reward)
            d_store_mean.append(np.mean(d_losses))
            d_store_var.append(np.var(d_losses))
            # td_store_mean.append(np.mean(td_errores))
            # td_store_var.append(np.var(td_errores))
            d_losses = []
            td_errores = []
            # if ep_reward > -300:RENDER = True
            ep = 0
            break
print('Running time: ', time.time() - t1)
print("total used sample: ", total_sample)
print("total augmented sample: ", count)

fig, ax = plt.subplots(1,2,figsize=(18,5))
ax[0].plot(r_store)
ax[1].plot(range(len(r_store)), d_store_mean, label='dyn-mean')
# ax[1].plot(range(len(r_store)), d_store_var, label='var')
ax[1].fill_between(range(len(r_store)), np.array(d_store_mean) - np.array(d_store_var), np.array(d_store_mean) + np.array(d_store_var), color='gray', alpha=0.2)
# ax[2].plot(range(len(r_store)), td_store_mean, label='td-mean')
# # ax[1].plot(range(len(r_store)), d_store_var, label='var')
# ax[2].fill_between(range(len(r_store)), np.array(td_store_mean) - np.array(td_store_var), np.array(td_store_mean) + np.array(td_store_var), color='gray', alpha=0.2)
fig.savefig('results13.png')