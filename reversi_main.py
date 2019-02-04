import numpy as np

from collections import deque
from datetime import datetime
from keras import backend as K
from keras.activations import relu, tanh
from keras.layers import Activation, Add, AveragePooling2D, BatchNormalization, Convolution2D, Dense, Flatten, GlobalAveragePooling2D, Input, MaxPool2D, ReLU, ZeroPadding2D
from keras.losses import mse
from keras.models import Sequential, Model, load_model, clone_model
from keras.optimizers import Adam, SGD

from board import Board
from model_path import MODEL_PATH


REWARD_NONE = 0
REWARD_WIN = 1
REWARD_LOSE = -1

class RandomExplorer:

    def __init__(self, random_func):
        self.random_func = random_func

    def select_action(self, t, obs, greedy_func):
        return self.random_func()

class GreedyExplorer:

    def select_action(self, t, obs, greedy_func):
        return greedy_func(obs)

class LinearDecayEpsilonGreedy:

    def __init__(self, start_epsilon, end_epsilon, decay_steps, random_func):
        assert 0 <= start_epsilon <= 1
        assert 0 <= end_epsilon <= 1
        assert end_epsilon <= start_epsilon
        assert decay_steps != 0
        self.start_epsilon = start_epsilon
        self.end_epsilon = end_epsilon
        self.decay_steps = decay_steps
        self.random_func = random_func

    def compute_epsilon(self, t):
        if self.decay_steps <= t:
            return self.end_epsilon
        diff_epsilon = self.end_epsilon - self.start_epsilon
        return self.start_epsilon + diff_epsilon * (t / self.decay_steps)

    def select_action(self, t, obs, greedy_func):
        epsilon = self.compute_epsilon(t)
        if np.random.rand() <= epsilon:
            return self.random_func()
        return greedy_func(obs)


class ConstantEpsilonGreedy:

    def __init__(self, epsilon, random_func):
        self.epsilon = epsilon
        self.random_func = random_func

    def select_action(self, t, obs, greedy_func):
        if np.random.rand() <= self.epsilon:
            return self.random_func()
        return greedy_func(obs)


class ReplayBuffer:

    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 0
        self.clear()


    def __len__(self):
        return self.size

    def clear(self):
        self.observations = deque(maxlen=self.capacity)
        self.actions = deque(maxlen=self.capacity)
        self.rewards = deque(maxlen=self.capacity)
        self.next_observations = deque(maxlen=self.capacity)

    def add(self, observation, action, reward, next_observation):
        if self.capacity == len(self):
            self.observations.popleft()
            self.actions.popleft()
            self.rewards.popleft()
            self.next_observations.popleft()
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_observations.append(next_observation)
        self.size = min(self.size + 1, self.capacity)


    def sample(self, sample_size):
        assert 0 < sample_size
        sample_size = min(len(self), sample_size)
        indices = np.random.choice(len(self), sample_size, replace=False)
        sample_observations = [self.observations[i] for i in indices]
        sample_actions = [self.actions[i] for i in indices]
        sample_rewards = [self.rewards[i] for i in indices]
        sample_next_observations = [self.next_observations[i] for i in indices]
        return sample_observations, sample_actions, sample_rewards, sample_next_observations


class RandomAgent:

    def __init__(self, random_func):
        self.random_func = random_func

    def act(self, obs):
        return self.random_func()


class GreedyAgent:

    def __init__(self, model):
        self.model = model

    def act(self, obs):
        assert obs.ndim == 3
        obs = obs.reshape(1, *obs.shape)
        return np.argmax(self.model.predict(obs)[0])



class ConstantEpsilonGreedyAgent:

    def __init__(self, model, epsilon, random_func):
        self.agent = GreedyAgent(model)
        self.epsilon = epsilon
        self.random_func = random_func

    def set_model(self, model):
        self.agent.model = clone_model(model)
        self.set_weights(model)

    def set_weights(self, model):
        self.agent.model.set_weights(model.get_weights())

    def act(self, obs):
        if np.random.rand() <= self.epsilon:
            return self.random_func()
        return self.agent.act(obs)


class DQN:

    def __init__(
        self,
        q_function,
        replay_buffer,
        explorer,
        gamma = 0.99,
        train_interval=1,
        sync_target_interval=2,
        replay_size=1024,
        batch_size=32,
        epochs = 1
    ):
        assert replay_size <= replay_buffer.capacity
        self.q_function = q_function
        self.replay_buffer = replay_buffer
        self.explorer = explorer
        self.gamma = gamma
        self.train_interval = train_interval
        self.sync_target_interval = sync_target_interval
        self.replay_size = replay_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.t = 0
        self.target_q_function = clone_model(self.q_function)
        self.sync_target_q_function()

    def act(self, obs):
        obs = np.reshape(obs, (1, *obs.shape))
        action = np.argmax(self.q_function.predict(obs)[0])
        return action

    def explore(self, obs):
        return self.explorer.select_action(self.t, obs, self.act)

    def stop_episode_and_train(self):
        self.t += 1
        if self.t % self.train_interval == 0:
            self.train()
        if self.t % self.sync_target_interval == 0:
            self.sync_target_q_function()

    def train(self):
        if len(self.replay_buffer) == 0:
            return
        observations, actions, rewards, next_observations = self.replay_buffer.sample(self.replay_size)
        targets = np.empty((len(observations), Board.area()))
        for i, (observation, action, reward, next_observation) in enumerate(zip(observations, actions, rewards, next_observations)):
            target = reward
            if next_observation is None:    #最後の盤面
                available_positions = observation.transpose(2, 0, 1)[3]  # TODO: マジックナンバー修正
                targets[i] = np.where(available_positions.reshape(-1) == 1, REWARD_NONE, REWARD_LOSE)
                available_flags = targets[i] == REWARD_NONE
                observation = np.reshape(observation, (1, *observation.shape))
                targets[i][available_flags] = self.q_function.predict(observation)[0][available_flags]
            else:
                next_observation = np.reshape(next_observation, (1, *next_observation.shape))
                next_action = np.argmax(self.q_function.predict(next_observation)[0])
                target -= self.gamma * self.target_q_function.predict(next_observation)[0][next_action]
                available_positions = observation.transpose(2, 0, 1)[3]  # TODO: マジックナンバー修正
                targets[i] = np.where(available_positions.reshape(-1) == 1, REWARD_NONE, REWARD_LOSE)
                available_flags = targets[i] == REWARD_NONE
                observation = np.reshape(observation, (1, *observation.shape))
                prediction = self.q_function.predict(observation)[0]
                targets[i][available_flags] = prediction[available_flags]
            targets[i][action] = target
        self.q_function.fit(np.array(observations), targets, self.batch_size, self.epochs, verbose=False)

    def sync_target_q_function(self):
        self.target_q_function.set_weights(self.q_function.get_weights())




def random_vs_random(board):
    board.board_reset()
    board.show_board()
    while not board.game_end:
        print("---------------------------------------------------")
        color_name = "黒" if board.turn == Board.black() else "白"
        print(color_name + "のターン")
        if not board.available_pos:
            print("Passします。")
            board.do_pass()
            board.end_check()
        else:
            pos = board.random_action(ndim=0)
            print("({}{})に置きました".format(pos[0], Board.n2l(pos[1] + 1)))
            board.agent_action(pos)
            if board.pss == 1:
                board.pss = 0
        if board.game_end:
            board.show_board()
            print("---------------------------------------------------")
            print("*****")
            print("最後に手を打ったのは{}です".format("黒" if board.turn == Board.black() else "白"))
            print("*****")
            if board.winner == Board.black():
                print("黒の勝ち")
            elif board.winner == Board.white():
                print("白の勝ち")
            else:
                print("引き分け")
        else:
            board.change_turn()
            board.show_board()


def agent_vs_agent(board, agent, replay_buffer, train=True):
    last_observations = {}
    last_actions = {}
    last_player = None
    last_observations[Board.black()] = None
    last_observations[Board.white()] = None
    last_actions[Board.black()] = None
    last_actions[Board.white()] = None
    board.board_reset()
    while not board.game_end:
        if not board.available_pos:
            board.do_pass()
            board.end_check()
        else:
            opp = board.get_opp()
            input_datas = board.get_input_datas()
            if last_observations[opp] is not None:
                replay_buffer.add(last_observations[opp], last_actions[opp], REWARD_NONE, input_datas)
            action = agent.explore(input_datas) if train else agent.act(input_datas)
            if not (divmod(action, Board.size()) in board.available_pos):
                action = board.random_action()
            last_player = board.turn
            last_observations[board.turn] = input_datas
            last_actions[board.turn] = action
            board.agent_action(divmod(action, Board.size()))
            if board.pss == 1:
                board.pss = 0
        if board.game_end:
            reward = None
            if board.winner == last_player:
                reward = REWARD_WIN
            elif board.get_loser() == last_player:
                reward = REWARD_LOSE
            else:
                reward = REWARD_NONE
            replay_buffer.add(last_observations[last_player], last_actions[last_player], reward, None)
        else:
            board.change_turn()
    return board.winner



def test_model(board, agent_b, agent_w, color, loop, show_board=False):
    print("******************************")
    print("対戦開始")
    print("-AI: " + ("黒" if color == Board.black() else "白"))
    win = 0
    lose = 0
    draw = 0
    success = 0
    num_actions = 0
    agents = {}
    agents[Board.black()] = agent_b
    agents[Board.white()] = agent_w
    for _ in range(loop):
        board.board_reset()
        while not board.game_end:
            if show_board:
                board.show_board()
            if not board.available_pos:
                board.pss += 1
                board.end_check()
            else:
                miss = False
                input_datas = board.get_input_datas()
                pos = agents[board.turn].act(input_datas)
                pos = divmod(pos, Board.size())
                if board.is_available(pos):
                    board.agent_action(pos)
                else:
                    miss = True
                    board.agent_action(board.random_action(ndim=2))
                if board.turn == color:
                    num_actions += 1
                    if not miss:
                        success += 1
                if board.pss == 1:
                    board.pss = 0
            if board.game_end:
                if board.winner == color:
                    win += 1
                elif board.winner == Board.empty():
                    draw += 1
                else:
                    lose += 1
            else:
                board.change_turn()
    print("win: {}, lose: {}, draw: {}".format(win, lose, draw))
    print("平均プレイターン数: {}".format(success / num_actions))
    print("******************************")
    return win == loop


def make_samples(board, agent, replay_buffer, sample_size, verbose=100, train=True):
    episode = 0
    while len(replay_buffer) < sample_size:
        episode += 1
        agent_vs_agent(board, agent, replay_buffer, train)
        if 0 < verbose and episode % verbose == 0:
            print("\r{:%}".format(len(replay_buffer) / sample_size), end="")
    if 0 < verbose:
        print("\r", end="")


def residual_block(x, filters, pool=False, first_layer=False):
    filters_1 = int(filters / 4)
    filters_2 = filters
    if first_layer:
        y = Convolution2D(filters_1, (1, 1), kernel_initializer="he_normal")(x)
    else:
        strides = (2, 2) if pool else (1, 1)
        y = BatchNormalization()(x)
        y = ReLU()(y)
        y = Convolution2D(filters_1, (1, 1), strides=strides, kernel_initializer="he_normal")(y)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Convolution2D(filters_1, (3, 3), padding="same", kernel_initializer="he_normal")(y)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Convolution2D(filters_2, (1, 1), kernel_initializer="he_normal")(y)
    x_shape = K.int_shape(x)
    residual_shape = K.int_shape(y)
    if x_shape != residual_shape:
        if pool:
            x = Convolution2D(filters_2, (1, 1), strides=(2, 2))(x)
        else:
            x = Convolution2D(filters_2, (1, 1))(x)
    y = Add()([x, y])
    return y


def main():

    board = Board()

    ########################################################################################
    #========================================================================================
    filters_0 = 128
    filters_1 = 512
    filters_2 = 1024
    filters_3 = 2048
    num_actions = Board.area()
    input_shape = (Board.size(), Board.size(), Board.channel())
    inputs = Input(input_shape)
    x = ZeroPadding2D((2, 2))(inputs)
    x = Convolution2D(filters_0, (3, 3), kernel_initializer="he_normal")(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = MaxPool2D(strides=(1, 1))(x)
    x = residual_block(x, filters_1, first_layer=True)
    x = residual_block(x, filters_1)
    x = residual_block(x, filters_1)
    x = residual_block(x, filters_2, pool=True)
    x = residual_block(x, filters_2)
    x = residual_block(x, filters_2)
    x = residual_block(x, filters_2)
    x = residual_block(x, filters_2)
    x = residual_block(x, filters_2)
    x = residual_block(x, filters_2)
    x = residual_block(x, filters_2)
    x = residual_block(x, filters_2)
    x = residual_block(x, filters_2)
    x = residual_block(x, filters_2)
    x = residual_block(x, filters_2)
    x = residual_block(x, filters_3, pool=True)
    x = residual_block(x, filters_3)
    x = residual_block(x, filters_3)
    x = BatchNormalization()(x)
    x = ReLU()(x)
    x = GlobalAveragePooling2D()(x)
    outputs = Dense(num_actions)(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(), loss=mse)
    #========================================================================================
    ########################################################################################

    replay_buffer_capacity = 2**20
    replay_buffer = ReplayBuffer(replay_buffer_capacity)
    
    print("===========================================")
    print("サンプル作成開始: {}".format(datetime.now().strftime("%Y/%m/%d %H:%M:%S")))
    make_samples(board, RandomAgent(board.random_action), replay_buffer, replay_buffer_capacity, train=False)
    print("サンプル作成終了: {}".format(datetime.now().strftime("%Y/%m/%d %H:%M:%S")))
    print("===========================================")

    print("===========================================")
    print("訓練開始: {}".format(datetime.now().strftime("%Y/%m/%d %H:%M:%S")))
    print("===========================================")

    explorer = LinearDecayEpsilonGreedy(1.0, 0.5, 100000, board.random_action)
    agent = DQN(model, replay_buffer, explorer, sync_target_interval=15, replay_size=2048, batch_size=64)


    wins = {}
    wins[Board.empty()] = 0
    wins[Board.black()] = 0
    wins[Board.white()] = 0
    episode = 0
    random_tester = True
    tester = RandomAgent(board.random_action)
    while True:
        episode += 1
        print("\r{} episode elapsed".format(episode), end="")
        winner = agent_vs_agent(board, agent, replay_buffer)
        wins[winner] += 1
        agent.stop_episode_and_train()
        if episode % 10 == 0:
            model.save(MODEL_PATH)
            print("\r", end="")
            print("------------------------------------------------------")
            print("Episode-{}: black win {}, white win {}, draw {}".format(episode, wins[Board.black()], wins[Board.white()], wins[Board.empty()]))
            print(datetime.now().strftime("%Y/%m/%d %H:%M:%S"))
            wins[Board.empty()] = 0
            wins[Board.black()] = 0
            wins[Board.white()] = 0
            perfect = test_model(board, agent, tester, Board.black(), 10)
            perfect &= test_model(board, tester, agent, Board.white(), 10)
            if perfect:
                print("==================================")
                print("全勝したのでテストモデルを更新します。")
                print("==================================")
                if random_tester:
                    random_tester = False
                    tester = ConstantEpsilonGreedyAgent(load_model(MODEL_PATH), 0.2, board.random_action)
                else:
                    tester.set_weights(model)
            print("------------------------------------------------------")

    model.save(MODEL_PATH)


if __name__ == "__main__":
    main()


