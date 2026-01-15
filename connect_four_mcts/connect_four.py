import random
from copy import deepcopy

class ConnectFour:
    """
    Class representing the game

    n_rows, n_columns : 
        amount of rows and columns on the board (default 7)
    winning_legnth : 
        amount of symbols in a sequence required to win (default 4)
    board : 
        list of lists. Each list is one column
        Players are represented with 0 and 1, player 0 begins the game.

    current_player : 
        player to move. Only possible values are 0 and 1. Starting value is 0.

    move_history : 
        move history since beginning of the game, described by list of columns in which symbols where put in exact same sequence

    """
    n_rows: int
    n_columns: int
    winning_length: int
    board: list[list[int]]
    move_history: list[int]
    current_player: int
 
    def __init__(self, n_rows : int = 6, n_columns : int = 7, winning_length : int = 4):
        self.n_rows = n_rows
        self.n_columns = n_columns
        self.winning_length = winning_length
        self.current_player = 0
        self.board = [[] for _ in range(self.n_columns)]
        self.move_history = []

    def print_board(self):
        print("\n" + "="*50)
        print("Current Board:")
        print("="*50)
        print("  ", end="")
        for col in range(self.n_columns):
            print(f" {col} ", end="")
        print()
        
        for row in range(self.n_rows - 1, -1, -1):
            print(f"{row} ", end="")
            for col in range(self.n_columns):
                if len(self.board[col]) > row:
                    symbol = 'X' if self.board[col][row] == 0 else 'O'
                    print(f"|{symbol}|", end="")
                else:
                    print("| |", end="")
            print()
        
        print("  " + "---" * self.n_columns)
        print(f"Current player: {'X' if self.current_player == 0 else 'O'}")
        print("="*50)

    def get_legal_moves(self) -> list[int]:
        return [col for col in range(self.n_columns) if len(self.board[col]) < self.n_rows]

    def make_move(self, column: int) -> bool:
        '''
            Makes a move if it is possible and returns True
            If move is not possible, returns False and it is not made 
        '''
        if column < 0 or column >= self.n_columns:
            return False
        
        if len(self.board[column]) >= self.n_rows:
            return False
        
        # Add player token to a column
        self.board[column].append(self.current_player)
        self.move_history.append(column)
        
        # Change player
        self.current_player = 1 - self.current_player
        
        return True
    
    def is_full(self) -> bool:
        '''
            Checks if board is full
        '''
        return all(len(col) >= self.n_rows for col in self.board)
    
    def get_winner(self) -> int:
        for player in [0, 1]:
            # Check vertical lines
            for col in range(self.n_columns):
                if self.check_line_vertical(self.board[col], player, self.winning_length):
                    return player
            
            # Check horizontal lines
            for row in range(self.n_rows):
                if self.check_line_horizontal(row, player, self.winning_length):
                    return player
            
            # Check diagonal lines
            if self.check_diagonals(player, self.winning_length):
                return player
        
        if not self.get_legal_moves():
            return 2 # draw 
        else:
            return -1 # game continues

    def check_line_vertical(self, column: list[int], player: int, winning_length: int) -> bool:
        if len(column) < winning_length:
            return False
        
        count = 0
        for symbol in reversed(column):  
            if symbol == player:
                count += 1
                if count >= winning_length:
                    return True
            else:
                count = 0
        return False

    def check_line_horizontal(self, row: int, player: int, winning_length: int) -> bool:
        count = 0
        for col in range(self.n_columns):
            if len(self.board[col]) > row and self.board[col][row] == player:
                count += 1
                if count >= winning_length:
                    return True
            else:
                count = 0
        return False

    def check_diagonals(self, player: int, winning_length: int) -> bool:
        for start_col in range(self.n_columns - winning_length + 1):
            for start_row in range(self.n_rows - winning_length + 1):
                count = 0
                for i in range(winning_length):
                    col = start_col + i
                    row = start_row + i
                    if (len(self.board[col]) > row and 
                        self.board[col][row] == player):
                        count += 1
                    else:
                        break
                if count >= winning_length:
                    return True
        
        for start_col in range(self.n_columns - winning_length + 1):
            for start_row in range(winning_length - 1, self.n_rows):
                count = 0
                for i in range(winning_length):
                    col = start_col + i
                    row = start_row - i
                    if (len(self.board[col]) > row and 
                        self.board[col][row] == player):
                        count += 1
                    else:
                        break
                if count >= winning_length:
                    return True
        
        return False
    
    def get_key(self):
        '''
            Function for caching board state
        '''
        return (tuple(tuple(col) for col in self.board), self.current_player)
    
class Player:
    """ 
        Class representing alfa-beta pruning algorithm to play the game
    """
    def __init__(self, player_id: int = 0):
        self.max_depth = 8
        self.player_id = player_id  # 0 or 1
        
        self.opening_book = self._initialize_opening_book()
        self.max_opening_depth = 8

    def _initialize_opening_book(self) -> dict:
        opening_book = {}
        opening_book[tuple()] = 3
        
        opening_book[(0,)] = 3  
        opening_book[(1,)] = 3
        opening_book[(2,)] = 3
        opening_book[(3,)] = 2  
        opening_book[(4,)] = 3
        opening_book[(5,)] = 3
        opening_book[(6,)] = 3
        
        opening_book[(3, 2)] = 4   
        opening_book[(3, 4)] = 2   
        opening_book[(3, 1)] = 5   

        opening_book[(3, 2, 4)] = 1  
        opening_book[(3, 4, 2)] = 5  
        opening_book[(3, 2, 1)] = 4   
        opening_book[(3, 4, 5)] = 2   
        
        opening_book[(2, 3)] = 4    
        opening_book[(4, 3)] = 2    
        opening_book[(1, 3)] = 5 
        
        opening_book[(3, 3)] = 2    
        opening_book[(2, 4)] = 3    
        opening_book[(4, 2)] = 3    
        
        return opening_book

    def get_opening_move(self, game: ConnectFour) -> int:
        if len(game.move_history) > self.max_opening_depth:
            return None
            
        move_tuple = tuple(game.move_history)
        if move_tuple in self.opening_book:
            suggested_move = self.opening_book[move_tuple]
            if self.is_valid_move(game, suggested_move):
                return suggested_move
        
        symmetric_history = self.get_symmetric_history(game.move_history)
        symmetric_tuple = tuple(symmetric_history)
        
        if symmetric_tuple in self.opening_book:
            suggested_move = self.opening_book[symmetric_tuple]
            symmetric_move = self.get_symmetric_column(suggested_move, game.n_columns)
            if self.is_valid_move(game, symmetric_move):
                return symmetric_move
        
        return None

    def get_symmetric_history(self, history: list) -> list:
        return [5 - move for move in history]  

    def get_symmetric_column(self, column: int, n_columns: int) -> int:
        return n_columns - 1 - column

    def is_valid_move(self, game: ConnectFour, column: int) -> bool:
        return (0 <= column < game.n_columns and 
                len(game.board[column]) < game.n_rows)

    def make_move(self, game: ConnectFour) -> int:
        '''
            Returns column for best found move 
        '''
        valid_moves = self.get_valid_moves(game)
        if not valid_moves:
            return 0
        
        if len(valid_moves) == 1:
            return valid_moves[0]
        
        opening_move = self.get_opening_move(game)
        if opening_move is not None:
            return opening_move
        
        # check if it is possible to win in one move 
        for col in valid_moves:
            game_copy = self.make_move_copy(game, col)
            if game_copy.get_winner() == game.current_player:
                return col
        
        # check if oponnent has to be blocked
        for col in valid_moves:
            game_copy = deepcopy(game)
            game_copy.current_player = 1 - game_copy.current_player  # simulate oponnent move
            game_copy.board[col].append(game_copy.current_player)
            if game_copy.get_winner() == game_copy.current_player:
                return col
        
        # alpha-beta
        _, best_column = self.alpha_beta(game, self.max_depth, float('-inf'), float('inf'), True)
        
        if best_column is None or best_column not in valid_moves:
            # prefer center of the board
            center_col = game.n_columns // 2
            if center_col in valid_moves:
                return center_col
            return random.choice(valid_moves)
        
        return best_column

    def alpha_beta(self, game: ConnectFour, depth: int, alpha: float, beta: float, maximizing_player: bool):
        '''
            Alpha-beta algorithm implementation
        '''
        
        winner = game.get_winner()
        
        # Ending cases
        if winner is not None:
            return self.evaluate_terminal(game, winner), None
        
        if depth == 0:
            return self.evaluate_position(game), None
        
        if game.is_full():
            return 0, None  # Draw
        
        valid_moves = self.get_valid_moves(game)
        if not valid_moves:
            return 0, None
        
        # Sort moves - prefer center of the board
        valid_moves = self.order_moves(game, valid_moves)
        
        best_column = valid_moves[0]
        
        if maximizing_player:
            max_eval = float('-inf')
            for col in valid_moves:
                game_copy = self.make_move_copy(game, col)
                eval_score, _ = self.alpha_beta(game_copy, depth - 1, alpha, beta, False)
                
                if eval_score > max_eval:
                    max_eval = eval_score
                    best_column = col
                
                alpha = max(alpha, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            
            return max_eval, best_column
        else:
            min_eval = float('inf')
            for col in valid_moves:
                game_copy = self.make_move_copy(game, col)
                eval_score, _ = self.alpha_beta(game_copy, depth - 1, alpha, beta, True)
                
                if eval_score < min_eval:
                    min_eval = eval_score
                    best_column = col
                
                beta = min(beta, eval_score)
                if beta <= alpha:
                    break  # Alpha-beta pruning
            
            return min_eval, best_column

    def order_moves(self, game: ConnectFour, moves: list[int]) -> list[int]:
        center = game.n_columns // 2
        return sorted(moves, key=lambda x: abs(x - center))

    def evaluate_terminal(self, game: ConnectFour, winner: int) -> float:
        if winner == game.current_player:
            return -10000  # game lost (bad thing)
        else:
            return 10000   # game won (good thing)

    def get_valid_moves(self, game: ConnectFour) -> list[int]:
        return [col for col in range(game.n_columns) if len(game.board[col]) < game.n_rows]

    def make_move_copy(self, game: ConnectFour, column: int) -> ConnectFour:
        '''
            Helper func for creating a copy of the state of the game
        '''
        game_copy = deepcopy(game)
        game_copy.board[column].append(game_copy.current_player)
        game_copy.move_history.append(column)
        game_copy.current_player = 1 - game_copy.current_player
        return game_copy

    def evaluate_position(self, game: ConnectFour) -> float:
        score = 0
        
        # evaluate for both players
        my_player = game.current_player
        opp_player = 1 - my_player
        
        score += self.evaluate_all_windows(game, my_player) 
        score -= self.evaluate_all_windows(game, opp_player) * 1.1  # Slightly higher weight for defense
        
        # Board center bonus
        center_col = game.n_columns // 2
        center_count = sum(1 for piece in game.board[center_col] if piece == my_player)
        score += center_count * 10
        
        return score

    #
    # Helper functions below 
    #
    def evaluate_all_windows(self, game: ConnectFour, player: int) -> float:
        score = 0
        win_len = game.winning_length
        
        for row in range(game.n_rows):
            for col in range(game.n_columns - win_len + 1):
                window = self.get_horizontal_window(game, row, col, win_len)
                score += self.evaluate_window(window, player)
        
        for col in range(game.n_columns):
            for row in range(game.n_rows - win_len + 1):
                window = self.get_vertical_window(game, col, row, win_len)
                score += self.evaluate_window(window, player)
        
        for row in range(game.n_rows - win_len + 1):
            for col in range(game.n_columns - win_len + 1):
                window = self.get_diagonal_window(game, row, col, win_len, 1)
                score += self.evaluate_window(window, player)
        
        for row in range(win_len - 1, game.n_rows):
            for col in range(game.n_columns - win_len + 1):
                window = self.get_diagonal_window(game, row, col, win_len, -1)
                score += self.evaluate_window(window, player)
        
        return score

    def get_horizontal_window(self, game: ConnectFour, row: int, start_col: int, length: int) -> list:
        window = []
        for col in range(start_col, start_col + length):
            if len(game.board[col]) > row:
                window.append(game.board[col][row])
            else:
                window.append(None)
        return window

    def get_vertical_window(self, game: ConnectFour, col: int, start_row: int, length: int) -> list:
        window = []
        for row in range(start_row, start_row + length):
            if len(game.board[col]) > row:
                window.append(game.board[col][row])
            else:
                window.append(None)
        return window

    def get_diagonal_window(self, game: ConnectFour, start_row: int, start_col: int, length: int, direction: int) -> list:
        window = []
        for i in range(length):
            row = start_row + (i * direction)
            col = start_col + i
            if 0 <= row < game.n_rows and 0 <= col < game.n_columns and len(game.board[col]) > row:
                window.append(game.board[col][row])
            else:
                window.append(None)
        return window

    def evaluate_window(self, window: list, player: int) -> float:
        score = 0
        opp_player = 1 - player
        
        player_count = window.count(player)
        opp_count = window.count(opp_player)
        empty_count = window.count(None)
        
        # if oponnent has their tokens in this window current player cannot win
        if opp_count > 0:
            return 0
        
        # Evaluation based on current player tokens in this window 
        if player_count == 4:
            score += 1000
        elif player_count == 3 and empty_count == 1:
            score += 50
        elif player_count == 2 and empty_count == 2:
            score += 10
        elif player_count == 1 and empty_count == 3:
            score += 1
        
        return score