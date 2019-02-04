import numpy as np

from keras.models import load_model

from board import Board
from model_path import BEST_MODEL_PATH

class HumanAgent:

    def __init__(self, board):
        self.board = board

    def act(self):
        while True:
            rows = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "6": 6, "7": 7}
            cols = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4, "f": 5, "g": 6, "h": 7}
            print("手を入力してください。: 例(f4)")
            action = input()
            if (len(action) == 2 and action[0] in cols and action[1] in rows):
                action = (rows[action[1]], cols[action[0]])
                if action in self.board.available_pos:                        
                    break
                else:
                    print("そこには置けません。")
            else:
                print("入力が正しくありません。")
        return action

class RandomAgent:

    def __init__(self, random_action):
        self.random_action = random_action

    def act(self):
        return self.random_action(ndim=2)

class DQNAgent:

    def __init__(self, board, model):
        self.board = board
        self.model = model


    def act(self):
        obs = self.board.get_input_datas()
        obs = obs.reshape((1, *obs.shape))
        action = divmod(np.argmax(self.model.predict(obs)[0]), Board.size())
        if action in self.board.available_pos:
            return action
        return self.board.random_action(ndim=2)


def get_player(msg, available_numbers):
    print(msg)
    while True:
        player = input()
        if player.isdecimal() and (int(player) in available_numbers):
            break
        else:
            print("入力が正しくありません。")
    return int(player)

def play():
    board = Board()
    agents = {0: RandomAgent(board.random_action), 1: DQNAgent(board, load_model(BEST_MODEL_PATH)), 2: HumanAgent(board)}
    agent_names = {0: "Random", 1: "DQN", 2: "あなた"}
    agent_indices = [k for k, v in agents.items()]
    agent_msg = "0: Random, 1: DQN, 2: Human"
    while True:
        print("==================================================================================")
        player_1 = get_player("先手を選んでください    " + agent_msg, agent_indices)
        player_2 = get_player("後手を選んでください    " + agent_msg, agent_indices)
        board.board_reset()
        while not board.game_end:
            print("---------------------------------------------------")
            color_name = agent_names[player_1] if board.turn == Board.black() else agent_names[player_2]
            print(color_name + "のターン")
            board.show_board()
            if not board.available_pos:
                print("置ける場所がありません。")
                print("Passします。")
                board.do_pass()
                board.end_check()
            else:
                player = player_1 if board.turn == Board.black() else player_2
                action = agents[player].act()
                print("({}{})に置きました".format(Board.n2l(action[1] + 1), action[0]))
                board.agent_action(action)
                if board.pss == 1:
                    board.pss = 0
            if board.game_end:
                print("---------------------------------------------------")
                board.show_board()
                print("黒: {}枚".format(board.nofb))
                print("白: {}枚".format(board.nofw))
                if board.nofw < board.nofb:
                    print(agent_names[player_1] + "の勝ち")
                elif board.nofb < board.nofw:
                    print(agent_names[player_2] + "の勝ち")
                else:
                    print("引き分け")
            else:
                board.change_turn()
        print("続けますか? (y/n)")
        cont = input()
        if cont == "n":
            break

play()