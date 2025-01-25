import numpy as np

player, opponent = 'x', 'o'

# Function to check if there are moves left on the board
def isMovesLeft(board):
    return any('_' in row for row in board)

# Evaluation function
def evaluate(b):
    # Check rows for a win
    for row in range(3):
        if b[row][0] == b[row][1] == b[row][2]:
            if b[row][0] == player:  # Player wins
                return 10
            elif b[row][0] == opponent:  # Opponent wins
                return -10

    # Check columns for a win
    for col in range(3):
        if b[0][col] == b[1][col] == b[2][col]:
            if b[0][col] == player:  # Player wins
                return 10
            elif b[0][col] == opponent:  # Opponent wins
                return -10

    # Check diagonals for a win
    if b[0][0] == b[1][1] == b[2][2]:
        if b[0][0] == player:  # Player wins
            return 10
        elif b[0][0] == opponent:  # Opponent wins
            return -10

    if b[0][2] == b[1][1] == b[2][0]:
        if b[0][2] == player:  # Player wins
            return 10
        elif b[0][2] == opponent:  # Opponent wins
            return -10

    # If no one has won, return 0
    return 0

# Minimax function
def minimax(board, depth, isMax):
    score = evaluate(board)

    if score == 10:
        return score
    if score == -10:
        return score
    if not isMovesLeft(board):
        return 0

    if isMax:
        best = -1000
        for i in range(3):
            for j in range(3):
                if board[i][j] == '_':
                    board[i][j] = player
                    best = max(best, minimax(board, depth + 1, not isMax))
                    board[i][j] = '_'
        return best
    else:
        best = 1000
        for i in range(3):
            for j in range(3):
                if board[i][j] == '_':
                    board[i][j] = opponent
                    best = min(best, minimax(board, depth + 1, not isMax))
                    board[i][j] = '_'
        return best

# Function to find the best move for the computer
def findBestMove(board):
    bestVal = -1000
    bestMove = (-1, -1)

    for i in range(3):
        for j in range(3):
            if board[i][j] == '_':
                board[i][j] = player
                moveVal = minimax(board, 0, False)
                board[i][j] = '_'
                if moveVal > bestVal:
                    bestMove = (i, j)
                    bestVal = moveVal

    return bestMove

# Function to display the board
def printBoard(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 5)

# Driver code
def main():
    board = [['_' for _ in range(3)] for _ in range(3)]
    print("Welcome to Tic-Tac-Toe!")
    printBoard(board)

    while isMovesLeft(board):
        # Player move
        while True:
            user_input = input("Enter your move (row and column: 0, 1, or 2) separated by space: ")
            row, col = map(int, user_input.split())
            if board[row][col] == '_':
                board[row][col] = opponent
                break
            else:
                print("Invalid move. Try again.")

        print("Your move:")
        printBoard(board)

        # if evaluate(board) == -10:
        #     print("You win!")
        #     return

        if not isMovesLeft(board):
            print("It's a tie!")
            return

        # Computer move
        print("Computer's move:")
        bestMove = findBestMove(board)
        board[bestMove[0]][bestMove[1]] = player
        printBoard(board)

        if evaluate(board) == -10:
            print("You win!")
            return

        if evaluate(board) == 10:
            print("Computer wins!")
            return



    print("It's a tie!")

if __name__ == "__main__":
    main()