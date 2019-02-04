import numpy as np

class Board:

    @staticmethod
    def size():
        return 8

    @staticmethod
    def area():
        return Board.size() * Board.size()

    @staticmethod
    def channel():
        return 5

    @staticmethod
    def num_players():
        return 2

    @staticmethod
    def empty():
        return 0

    @staticmethod
    def black():
        return 1

    @staticmethod
    def white():
        return 2

    # @staticmethod
    # def history_size():
    #     return 7

    @staticmethod
    def dir():
        return ((-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1))

    @staticmethod
    def is_in_board(x, y):
        return 0 <= x < Board.size() and 0 <= y < Board.size()

    @staticmethod
    def n2l(n):
        l = ["", "a", "b", "c", "d", "e", "f", "g", "h"]
        assert 0 <= n < len(l)
        return l[n]

    @staticmethod
    def l2n(l):
        n = {"a":1, "b":2, "c":3, "d":4, "e":5, "f":6, "g":7, "h":8,}
        assert l in n
        return n[l]

    @staticmethod
    def stone(n):
        stone = [" ", "○", "●"]
        return stone[n]

    @staticmethod
    def filled(fill=0, ndim=2, dtype="i"):
        if ndim == 3:
            shape = (1, Board.size(), Board.size())
        elif ndim == 2:
            shape = (Board.size(), Board.size())
        else:
            shape = Board.area()
        return np.full(shape, fill, dtype)


    def __init__(self):
        self.board_reset()

    # ###########################
    # # 未テスト
    # ###########################
    # def make_board(self, black_image, white_image):
    #     assert black_image.shape == (Board.size(), Board.size())
    #     assert white_image.shape == (Board.size(), Board.size())
    #     board = np.zeros((Board.size(), Board.size()), dtype="i")
    #     board += np.where(black_image == 1, Board.black(), Board.empty())
    #     board += np.where(white_image == 1, Board.white(), Board.empty())

    def board_reset(self):
        self.board = np.zeros((Board.size(), Board.size()), dtype="i") # dtype="f" ??
        mid = Board.size()//2
        self.board[mid, mid] = Board.white()
        self.board[mid-1, mid-1] = Board.white()
        self.board[mid-1, mid] = Board.black()
        self.board[mid, mid-1] = Board.black()
        self.winner = Board.empty()
        self.turn = Board.black()
        self.game_end = False
        self.pss = 0
        self.nofb = 0
        self.nofw = 0
        self.available_pos = self.search_positions()
        #self.initialize_history()

    # def initialize_history(self):
    #     players = [self.black(), self.white()]
    #     self.histories = [[], [], []]
    #     for p in players:
    #         for _ in range(Board.history_size()):
    #             self.histories[p].append(Board.filled())

    def show_board(self):
        print(" ", end="")
        for i in range(1, Board.size() + 1):
            print(" {}".format(Board.n2l(i)), end="")
        print("")
        for i in range(Board.size()):
            print(i, end="")
            for j in range(Board.size()):
                if self.is_available((i, j)):
                    print(" ☆", end="")
                else:
                    print(" ", end="")
                    print(Board.stone(int(self.board[i][j])), end="")
            print("")

    def put_stone(self, pos):
        #assert self.is_available(pos)
        # opp = self.get_opp()
        # self.histories[opp].append(self.get_positions(opp))
        # self.histories[self.turn].append(self.get_positions(self.turn))
        self.board[pos[0], pos[1]] = self.turn
        self.do_reverse(pos)


    ###########################
    # 未テスト
    ###########################
    def do_pass(self):
        assert not self.available_pos
        self.pss += 1
        # opp = self.get_opp()
        # self.histories[opp].append(self.get_positions(opp))
        # self.histories[self.turn].append(self.get_positions(self.turn))

    # ###########################
    # # 未テスト
    # ###########################
    # def undo(self):
    #     assert Board.history_size() < len(self.histories[self.get_opp()])
    #     if Board.history_size() < len(self.histories[self.get_opp()]):
    #         board = np.zeros_like(self.board)
    #         board += np.where(self.histories[Board.black()][-1] == 1, Board.black(), Board.empty())
    #         board += np.where(self.histories[Board.white()][-1] == 1, Board.white(), Board.empty())
    #         self.histories[Board.black()].pop(-1)
    #         self.histories[Board.white()].pop(-1)
    #         self.board = board
    #         if self.game_end:
    #             self.game_end = False
    #         if 0 < self.pss:
    #             self.pss -= 1
    #         self.change_turn()

            

    def change_turn(self):
        self.turn = Board.white() if self.turn == Board.black() else Board.black()
        self.available_pos = self.search_positions()

    def random_action(self, ndim=1):
        if self.available_pos:
            arg_pos = np.random.choice(len(self.available_pos))
            pos = self.available_pos[arg_pos]
            if ndim==1:
                pos = pos[0] * Board.size() + pos[1]
            return pos
        else:
            return False

    def agent_action(self, pos):
        #assert self.is_available(pos)
        self.put_stone(pos)
        self.end_check()

    def do_reverse(self, pos):
        for di, dj, in Board.dir():
            opp = Board.black() if self.turn == Board.white() else Board.white()
            boardcopy = self.board.copy()
            i = pos[0]
            j = pos[1]
            flag = False
            while Board.is_in_board(i, j):
                i += di
                j += dj
                if Board.is_in_board(i, j) and boardcopy[i, j] == opp:
                    flag = True
                    boardcopy[i, j] = self.turn
                elif (not Board.is_in_board(i, j)) or (flag == False and boardcopy[i, j] == self.turn) or (boardcopy[i, j] == Board.empty()):
                    break
                elif boardcopy[i, j] == self.turn and flag == True:
                    self.board = boardcopy.copy()
                    break

    def search_positions(self):
        pos = []
        emp = np.where(self.board == 0)
        for i in range(emp[0].size):
            p = (emp[0][i], emp[1][i])
            if self.is_available(p):
                pos.append(p)
        return pos

    def is_available(self, pos):
        if self.board[pos[0], pos[1]] != Board.empty():
            return False
        opp = Board.black() if self.turn == Board.white() else Board.white()
        for di, dj, in Board.dir():
            i = pos[0]
            j = pos[1]
            flag = False
            while Board.is_in_board(i, j):
                i += di
                j += dj
                if Board.is_in_board(i, j) and self.board[i, j] == opp:
                    flag = True
                elif (not Board.is_in_board(i, j)) or (flag == False and self.board[i, j] != opp) or (self.board[i, j] == Board.empty()):
                    break
                elif self.board[i, j] == self.turn and flag == True:
                    return True
        return False

    def end_check(self):
        if np.count_nonzero(self.board) == Board.area() or self.pss == Board.num_players():
            self.game_end = True
            self.nofb = len(np.where(self.board==Board.black())[0])
            self.nofw = len(np.where(self.board==Board.white())[0])
            self.winner = Board.black() if self.nofw < self.nofb else Board.white() if self.nofb < self.nofw else Board.empty()

    def get_positions(self, color, emp=0, fill=1, ndim=2):
        p = np.array([], dtype="i")
        for i in range(Board.size()):
            for j in range(Board.size()):
                v = emp if color != self.board[i, j] else fill
                p = np.append(p, v)
        if ndim==3:
            shape = (1, Board.size(), Board.size())
        elif ndim==2:
            shape = (Board.size(), Board.size())
        else:
            shape = Board.area()
        return np.reshape(p, shape)

    def get_available_positions(self, available=1, unavailable=0, ndim=2):
        p = np.array([], dtype="i")
        for i in range(Board.size()):
            for j in range(Board.size()):
                v = available if self.is_available((i, j)) else unavailable
                p = np.append(p, v)
        if ndim==3:
            shape = (1, Board.size(), Board.size())
        elif ndim==2:
            shape = (Board.size(), Board.size())
        else:
            shape = Board.area()
        return np.reshape(p, shape)

    def get_input_datas(self):
        #################################
        # チャンネルの順番は入れ替え厳禁
        #################################
        #0ch        黒石の位置
        #1ch        白石の位置
        #2ch        空白の位置
        #3ch       合法手なら1, それ以外は0
        #4ch       黒番ならすべて1, 白番ならすべて0
        input_datas = np.empty((Board.channel(), Board.size(), Board.size()), dtype="i")
        input_datas[0] = self.get_positions(Board.black())
        input_datas[1] = self.get_positions(Board.white())
        input_datas[2] = self.get_positions(Board.empty())
        input_datas[3] = self.get_available_positions()
        input_datas[4] = np.ones((Board.size(), Board.size()), dtype="i") if self.turn == Board.black() else np.zeros((Board.size(), Board.size()), dtype="i")
        return input_datas.transpose(1, 2, 0)

    def get_opp(self):
        return Board.black() if self.turn == Board.white() else Board.white()

    def get_loser(self):
        if self.winner == Board.black():
            return Board.white()
        elif self.winner == Board.white():
            return Board.black()
        return Board.empty()


