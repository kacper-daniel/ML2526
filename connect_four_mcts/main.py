from mcts import MCTS, MCTSNode
from connect_four import ConnectFour, Player

mcts = MCTS()

if __name__ == "__main__":
    game = ConnectFour()
    ai_player = Player() # alfa-beta pruning AI 

    winner = -1 
    while winner == -1:
        game.print_board()
        if game.current_player == 0:
            # move = int(input())
            # game.make_move(move)
            best_move = ai_player.make_move(game)
            game.make_move(best_move)
        else:
            #best_move = ai_player.make_move(game)
            best_move = mcts.search(MCTSNode(game), iterations=10000, choice_method="uct")
            game.make_move(best_move)
        winner = game.get_winner()

    game.print_board()
    print(f"Player {winner} wins")
