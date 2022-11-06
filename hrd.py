# CSC384 Assignment 1
# BY Yiteng Sun
import sys
import copy


class PriorityQueueNode:

    # Create a single node for the following linked list implementation.
    def __init__(self, item, priority):
        self.item = item
        self.priority = priority
        self.next = None


class PriorityQueue:

    # This time I will use linked list method to create priority queue. For the priority queue,
    # there are three main methods we may need to use: check whether linked list is empty, push
    # something in our priority queue and pop the item with the highest priority.
    def __init__(self):
        self.front = None

    def is_empty(self):
        if self.front is None:
            return True
        else:
            return False

    def push(self, item, priority):
        if self.is_empty():
            self.front = PriorityQueueNode(item, priority)
        else:
            if self.front.priority > priority:
                node = PriorityQueueNode(item, priority)
                node.next = self.front
                self.front = node
            else:
                temp = self.front
                while temp.next is not None:
                    if temp.next.priority >= priority:
                        break
                    temp = temp.next
                node = PriorityQueueNode(item, priority)
                node.next = temp.next
                temp.next = node

    def pop(self):
        if not self.is_empty():
            temp = self.front
            self.front = self.front.next
            return [temp.priority, temp.item]
        else:
            return None


class Board:

    # For the board implementation, we need three variable: type of gameboard is List[list],
    # prev is Board and stepcost is int. Gameboard will store the data we derive from the file
    # and prev will store the previous gameboard(i.e. the state of board before the move)
    # stepcost is variable that we store the current total moves.
    def __init__(self, gameboard, prev=None):
        self.gameboard = copy.deepcopy(gameboard)
        # we use deepcopy to avoid the circumstance that we changed the
        # previous gameboard after we make a move.
        self.prev = prev
        if self.prev is not None:
            self.stepcost = prev.stepcost + 1
        else:
            self.stepcost = 0


def read_file():
    # Read the file and convert it into our data structure.
    file_name = sys.argv[1]
    boardfile = open(file_name, "r")
    board = Board([[0, 1, 2, 3],
                   [4, 5, 6, 7],
                   [8, 9, 8, 7],
                   [6, 5, 4, 3],
                   [2, 1, 0, 1]])
    curr_line = 0
    while (1):
        line = boardfile.readline()
        if not line:
            break

        for i in range(0, 4):
            number = line[i]
            board.gameboard[curr_line][i] = int(number)

        curr_line += 1

    boardfile.close()
    return board


def covert_to_valid_board(board) -> str:
    # convert the result to the correct format.
    result_strings = ""
    for i in range(0, 5):
        j = 0
        while j < 4:
            if board.gameboard[i][j] == 7:  # single 1x1
                result_strings += "4"
                j += 1
            elif board.gameboard[i][j] == 0:  # empty 1x1
                result_strings += "0"
                j += 1
            elif board.gameboard[i][j] == 1:  # 2x2
                result_strings += "1"
                j += 1
            elif j < 3 and board.gameboard[i][j] == board.gameboard[i][j + 1]:  # horizontal 1x2
                result_strings += "22"
                j += 2
            else:
                result_strings += "3"  # verticle 1x2
                j += 1
        result_strings += "\n"

    return result_strings


def create_priority_queue():
    return PriorityQueue()


def whether_goal(board):
    # Check the goal state: in the gameboard, we know that [3, 1], [3, 2]
    # [4, 1], [4, 2] are goal states.
    if (board.gameboard[3][1] != 1 or board.gameboard[3][2] != 1 or
            board.gameboard[4][1] != 1 or board.gameboard[4][2] != 1):
        return "NO"
    else:
        return "YES"


def is_empty(board) -> list:
    # loop over the gameboard to check every empty position.
    empty_position = []
    for i in range(0, 5):
        for j in range(0, 4):
            if board.gameboard[i][j] == 0:
                empty_position.append(i)
                empty_position.append(j)
    return empty_position


def make_single_move(board, position_x, position_y) -> list:
    # first create a list to store all the successor.
    successor_list = []

    # Moving Up
    if position_x < 4 and board.gameboard[position_x + 1][position_y] == 7:
        # create a new State to continue the following steps
        successor = Board(board.gameboard, board)
        successor.gameboard[position_x + 1][position_y] = 0
        successor.gameboard[position_x][position_y] = 7
        successor_list.append(successor)

    # Moving Left
    if position_y < 3 and board.gameboard[position_x][position_y + 1] == 7:
        # create a new State to continue the following steps
        successor = Board(board.gameboard, board)
        successor.gameboard[position_x][position_y + 1] = 0
        successor.gameboard[position_x][position_y] = 7
        successor_list.append(successor)

    # Moving Down
    if position_x > 0 and board.gameboard[position_x - 1][position_y] == 7:
        # create a new State to continue the following steps
        successor = Board(board.gameboard, board)
        successor.gameboard[position_x - 1][position_y] = 0
        successor.gameboard[position_x][position_y] = 7
        successor_list.append(successor)

    # Moving Right
    if position_y > 0 and board.gameboard[position_x][position_y - 1] == 7:
        # create a new State to continue the following steps
        successor = Board(board.gameboard, board)
        successor.gameboard[position_x][position_y - 1] = 0
        successor.gameboard[position_x][position_y] = 7
        successor_list.append(successor)

    return successor_list


def make_big_move(board, empty_position) -> list:
    # this function is used to move 2x2 squares. Quiet similar to 1x1 squares move
    successor_list = []
    pixel1_x = empty_position[0]
    pixel1_y = empty_position[1]
    pixel2_x = empty_position[2]
    pixel2_y = empty_position[3]

    # Move up
    if pixel1_x < 3 and pixel1_x == pixel2_x and abs(pixel1_y - pixel2_y) == 1:
        if board.gameboard[pixel1_x + 1][pixel1_y] == 1 and board.gameboard[pixel2_x + 1][pixel2_y] == 1:
            successor = Board(board.gameboard, board)
            successor.gameboard[pixel1_x][pixel1_y] = 1
            successor.gameboard[pixel2_x][pixel2_y] = 1
            successor.gameboard[pixel1_x + 2][pixel1_y] = 0
            successor.gameboard[pixel2_x + 2][pixel2_y] = 0
            successor_list.append(successor)

    # Move Down
    if pixel1_x > 1 and pixel1_x == pixel2_x and abs(pixel1_y - pixel2_y) == 1:
        if board.gameboard[pixel1_x - 1][pixel1_y] == 1 and board.gameboard[pixel2_x - 1][pixel2_y] == 1:
            successor = Board(board.gameboard, board)
            successor.gameboard[pixel1_x][pixel1_y] = 1
            successor.gameboard[pixel2_x][pixel2_y] = 1
            successor.gameboard[pixel1_x - 2][pixel1_y] = 0
            successor.gameboard[pixel2_x - 2][pixel2_y] = 0
            successor_list.append(successor)

    # Move Right
    if pixel1_y > 1 and pixel1_y == pixel2_y and abs(pixel1_x - pixel2_x) == 1:
        if board.gameboard[pixel1_x][pixel1_y - 1] == 1 and board.gameboard[pixel2_x][pixel2_y - 1] == 1:
            successor = Board(board.gameboard, board)
            successor.gameboard[pixel1_x][pixel1_y] = 1
            successor.gameboard[pixel2_x][pixel2_y] = 1
            successor.gameboard[pixel1_x][pixel1_y - 2] = 0
            successor.gameboard[pixel2_x][pixel2_y - 2] = 0
            successor_list.append(successor)

    # Move Left
    if pixel1_y < 2 and pixel1_y == pixel2_y and abs(pixel1_x - pixel2_x) == 1:
        if board.gameboard[pixel1_x][pixel1_y + 1] == 1 and board.gameboard[pixel2_x][pixel2_y + 1] == 1:
            successor = Board(board.gameboard, board)
            successor.gameboard[pixel1_x][pixel1_y] = 1
            successor.gameboard[pixel2_x][pixel2_y] = 1
            successor.gameboard[pixel1_x][pixel1_y + 2] = 0
            successor.gameboard[pixel2_x][pixel2_y + 2] = 0
            successor_list.append(successor)

    return successor_list


def make_vertical_1x2_move(board, empty_position) -> list:
    # this function is for finding the potential 1x2 grids to move
    successor_list = []
    pixel1_x = empty_position[0]
    pixel1_y = empty_position[1]
    pixel2_x = empty_position[2]
    pixel2_y = empty_position[3]

    # Move Up
    if pixel1_x < 3 and board.gameboard[pixel1_x + 1][pixel1_y] == board.gameboard[pixel1_x + 2][pixel1_y]:
        if (board.gameboard[pixel1_x + 1][pixel1_y] != 0 and board.gameboard[pixel1_x + 1][pixel1_y] != 1
                and board.gameboard[pixel1_x + 1][pixel1_y] != 7):
            record = board.gameboard[pixel1_x + 1][pixel1_y]
            successor = Board(board.gameboard, board)
            successor.gameboard[pixel1_x][pixel1_y] = record
            successor.gameboard[pixel1_x + 2][pixel1_y] = 0
            successor_list.append(successor)

    if pixel2_x < 3 and board.gameboard[pixel2_x + 1][pixel2_y] == board.gameboard[pixel2_x + 2][pixel2_y]:
        if (board.gameboard[pixel2_x + 1][pixel2_y] != 0 and board.gameboard[pixel2_x + 1][pixel2_y] != 1
                and board.gameboard[pixel2_x + 1][pixel2_y] != 7):
            record = board.gameboard[pixel2_x + 1][pixel2_y]
            successor = Board(board.gameboard, board)
            successor.gameboard[pixel2_x][pixel2_y] = record
            successor.gameboard[pixel2_x + 2][pixel2_y] = 0
            successor_list.append(successor)

    # Same algorithm to write move down
    if pixel1_x > 1 and board.gameboard[pixel1_x - 1][pixel1_y] == board.gameboard[pixel1_x - 2][pixel1_y]:
        if (board.gameboard[pixel1_x - 1][pixel1_y] != 0 and board.gameboard[pixel1_x - 1][pixel1_y] != 1
                and board.gameboard[pixel1_x - 1][pixel1_y] != 7):
            record = board.gameboard[pixel1_x - 1][pixel1_y]
            successor = Board(board.gameboard, board)
            successor.gameboard[pixel1_x][pixel1_y] = record
            successor.gameboard[pixel1_x - 2][pixel1_y] = 0
            successor_list.append(successor)

    if pixel2_x > 1 and board.gameboard[pixel2_x - 1][pixel2_y] == board.gameboard[pixel2_x - 2][pixel2_y]:
        if (board.gameboard[pixel2_x - 1][pixel2_y] != 0 and board.gameboard[pixel2_x - 1][pixel2_y] != 1
                and board.gameboard[pixel2_x - 1][pixel2_y] != 7):
            record = board.gameboard[pixel2_x - 1][pixel2_y]
            successor = Board(board.gameboard, board)
            successor.gameboard[pixel2_x][pixel2_y] = record
            successor.gameboard[pixel2_x - 2][pixel2_y] = 0
            successor_list.append(successor)

    # A little different algorithm to perform move left and right, quite similar to 2x2 move.
    # First perform move left
    if (pixel1_y == pixel2_y and abs(pixel1_x - pixel2_x) == 1 and pixel1_y > 0 and
            board.gameboard[pixel1_x][pixel1_y - 1] == board.gameboard[pixel2_x][pixel2_y - 1]):
        record = board.gameboard[pixel1_x][pixel1_y - 1]
        if record != 0 and record != 1 and record != 7:
            successor = Board(board.gameboard, board)
            successor.gameboard[pixel1_x][pixel1_y] = record
            successor.gameboard[pixel2_x][pixel2_y] = record
            successor.gameboard[pixel1_x][pixel1_y - 1] = 0
            successor.gameboard[pixel2_x][pixel2_y - 1] = 0
            successor_list.append(successor)

    # Second perform move left
    if (pixel1_y == pixel2_y and abs(pixel1_x - pixel2_x) == 1 and pixel1_y < 3 and
            board.gameboard[pixel1_x][pixel1_y + 1] == board.gameboard[pixel2_x][pixel2_y + 1]):
        record = board.gameboard[pixel1_x][pixel1_y + 1]
        if record != 0 and record != 1 and record != 7:
            successor = Board(board.gameboard, board)
            successor.gameboard[pixel1_x][pixel1_y] = record
            successor.gameboard[pixel2_x][pixel2_y] = record
            successor.gameboard[pixel1_x][pixel1_y + 1] = 0
            successor.gameboard[pixel2_x][pixel2_y + 1] = 0
            successor_list.append(successor)

    return successor_list


def make_horizontal_1x2_move(board, empty_position):
    successor_list = []
    pixel1_x = empty_position[0]
    pixel1_y = empty_position[1]
    pixel2_x = empty_position[2]
    pixel2_y = empty_position[3]

    # Move Up
    if (pixel1_x < 4 and pixel1_x == pixel2_x and abs(pixel1_y - pixel2_y) == 1 and
            board.gameboard[pixel1_x + 1][pixel1_y] == board.gameboard[pixel2_x + 1][pixel2_y]):
        record = board.gameboard[pixel1_x + 1][pixel1_y]
        if record != 0 and record != 1 and record != 7:
            successor = Board(board.gameboard, board)
            successor.gameboard[pixel1_x][pixel1_y] = record
            successor.gameboard[pixel2_x][pixel2_y] = record
            successor.gameboard[pixel1_x + 1][pixel1_y] = 0
            successor.gameboard[pixel2_x + 1][pixel2_y] = 0
            successor_list.append(successor)

    # Move Down
    if (pixel1_x > 0 and pixel1_x == pixel2_x and abs(pixel1_y - pixel2_y) == 1 and
            board.gameboard[pixel1_x - 1][pixel1_y] == board.gameboard[pixel2_x - 1][pixel2_y]):
        record = board.gameboard[pixel1_x - 1][pixel1_y]
        if record != 0 and record != 1 and record != 7:
            successor = Board(board.gameboard, board)
            successor.gameboard[pixel1_x][pixel1_y] = record
            successor.gameboard[pixel2_x][pixel2_y] = record
            successor.gameboard[pixel1_x - 1][pixel1_y] = 0
            successor.gameboard[pixel2_x - 1][pixel2_y] = 0
            successor_list.append(successor)

    # Moving Left
    if pixel1_y > 1 and board.gameboard[pixel1_x][pixel1_y - 1] == board.gameboard[pixel1_x][pixel1_y - 2]:
        if (board.gameboard[pixel1_x][pixel1_y - 1] != 0 and board.gameboard[pixel1_x][pixel1_y - 1] != 1
                and board.gameboard[pixel1_x][pixel1_y - 1] != 7):
            record = board.gameboard[pixel1_x][pixel1_y - 1]
            successor = Board(board.gameboard, board)
            successor.gameboard[pixel1_x][pixel1_y] = record
            successor.gameboard[pixel1_x][pixel1_y - 2] = 0
            successor_list.append(successor)

    if pixel2_y > 1 and board.gameboard[pixel2_x][pixel2_y - 1] == board.gameboard[pixel2_x][pixel2_y - 2]:
        if (board.gameboard[pixel2_x][pixel2_y - 1] != 0 and board.gameboard[pixel2_x][pixel2_y - 1] != 1
                and board.gameboard[pixel2_x][pixel2_y - 1] != 7):
            record = board.gameboard[pixel2_x][pixel2_y - 1]
            successor = Board(board.gameboard, board)
            successor.gameboard[pixel2_x][pixel2_y] = record
            successor.gameboard[pixel2_x][pixel2_y - 2] = 0
            successor_list.append(successor)

    # Moving Right
    if pixel1_y < 2 and board.gameboard[pixel1_x][pixel1_y + 1] == board.gameboard[pixel1_x][pixel1_y + 2]:
        if (board.gameboard[pixel1_x][pixel1_y + 1] != 0 and board.gameboard[pixel1_x][pixel1_y + 1] != 1
                and board.gameboard[pixel1_x][pixel1_y + 1] != 7):
            record = board.gameboard[pixel1_x][pixel1_y + 1]
            successor = Board(board.gameboard, board)
            successor.gameboard[pixel1_x][pixel1_y] = record
            successor.gameboard[pixel1_x][pixel1_y + 2] = 0
            successor_list.append(successor)

    if pixel2_y < 2 and board.gameboard[pixel2_x][pixel2_y + 1] == board.gameboard[pixel2_x][pixel2_y + 2]:
        if (board.gameboard[pixel2_x][pixel2_y + 1] != 0 and board.gameboard[pixel2_x][pixel2_y + 1] != 1
                and board.gameboard[pixel2_x][pixel2_y + 1] != 7):
            record = board.gameboard[pixel2_x][pixel2_y + 1]
            successor = Board(board.gameboard, board)
            successor.gameboard[pixel2_x][pixel2_y] = record
            successor.gameboard[pixel2_x][pixel2_y + 2] = 0
            successor_list.append(successor)

    return successor_list


def all_possible_result(board) -> list:
    # For checking all the possible moves, we first need to check empty position:
    empty_position = is_empty(board)
    result = []
    x_1 = empty_position[0]
    y_1 = empty_position[1]
    # For the first empty grid, check whether it contains the potential move.
    result += make_single_move(board, x_1, y_1)
    x_2 = empty_position[2]
    y_2 = empty_position[3]
    # For the second empty grid, check whether it contains the potential move.
    result += make_single_move(board, x_2, y_2)

    # for 2x2 and 1x2 blocks, find their potential moves.
    result += make_big_move(board, empty_position)
    result += make_vertical_1x2_move(board, empty_position)
    result += make_horizontal_1x2_move(board, empty_position)

    return result


def calculate_actual_cost(board):
    return board.stepcost


def calculate_manhattan(board):
    # Find Cao Cao
    for i in range(0, 5):
        for j in range(0, 4):
            if board.gameboard[i][j] == 1:
                position_x = i
                position_y = j
                # Calculate the distance.
                calculation = abs(position_x - 3) + abs(position_y - 1)
                return calculation

    return None


def calculate_total_cost(board):
    # f(n) = g(n) + h(n)
    return calculate_actual_cost(board) + calculate_manhattan(board)


def calculate_mine_heuristic(board):
    # Find Cao Cao
    for i in range(0, 5):
        for j in range(0, 4):
            if board.gameboard[i][j] == 1:
                position_x = i
                position_y = j
                # This is advanced heuristic function designed by myself. I will discuss it
                # in advanced.pdf
                calculation = 5*(abs(position_x - 3) + abs(position_y - 1))
                return calculation

    return None


def calculate_mine_total_cost(board):
    # f(n) = g(n) + h(n)
    return calculate_mine_heuristic(board) + calculate_actual_cost(board)


def astar(starter_board: Board):
    # create a frontier = {initial}
    frontier = create_priority_queue()
    frontier.push(starter_board, calculate_total_cost(starter_board))
    # create an explored set to record the state we have discovered.
    explored = set()

    # The searching loop
    while not frontier.is_empty():
        select = frontier.pop()[1]
        # pruning: if the selected state we already have discovered, just ignore it.
        if covert_to_valid_board(select) in explored:
            continue
        else:
            explored.add(covert_to_valid_board(select))
            possible_results = all_possible_result(select)
            # check all the possible result and do another search.
            for pos_res in possible_results:
                # if the pos_res is our solution board, print the result.
                if whether_goal(pos_res) == "YES":
                    return [pos_res, calculate_total_cost(pos_res)]
                else:
                    frontier.push(pos_res, calculate_total_cost(pos_res))
    # if we didnot find the result return none.
    return None


def dfs(starter_board: Board):
    # create a LIFO stack as frontier: [starter_board]
    frontier = [starter_board]
    explored = set()

    # start with the initial board
    explored.add(covert_to_valid_board(starter_board))

    # loop over the frontier to start the search
    while len(frontier) != 0:
        curr = frontier.pop()  # pop the last-in element
        possible_results = all_possible_result(curr)

        # check all the possible result and do another search.
        for pos_res in possible_results:
            if covert_to_valid_board(pos_res) not in explored:
                explored.add(covert_to_valid_board(pos_res))
                frontier.append(pos_res)
            if whether_goal(pos_res) == "YES":
                return pos_res, calculate_actual_cost(pos_res)

    return None


def mine_astar(starter_board: Board):
    # the following code is the same to astar function, I use my designed heuristic function instead.
    frontier = create_priority_queue()
    frontier.push(starter_board, calculate_mine_total_cost(starter_board))

    explored = set()

    while not frontier.is_empty():
        select = frontier.pop()[1]

        if covert_to_valid_board(select) in explored:
            continue
        else:
            explored.add(covert_to_valid_board(select))
            possible_results = all_possible_result(select)

            for pos_res in possible_results:
                if whether_goal(pos_res) == "YES":
                    return [pos_res, calculate_mine_total_cost(pos_res)]
                else:
                    frontier.push(pos_res, calculate_mine_total_cost(pos_res))

    return None


def make_output(algorithm, solution):
    # read the filename and split ".txt"
    filename = sys.argv[1]
    fnwotxt = filename[:-4]
    # generate the output filename to match the system call.
    output_filename = f"{fnwotxt}sol_{algorithm}.txt"
    # write the file
    output_file = open(output_filename, "w")
    output_file.write(f"Cost of the solution: {solution[1]}\n")
    final_result = solution[0]
    sol_to_print = []
    while final_result != None:  # append all the board to the list.
        sol_to_print.append(final_result)
        final_result = final_result.prev
    for i in range(1, len(sol_to_print) + 1):  # write all the boards to the file.
        output_file.write(covert_to_valid_board(board=sol_to_print[-i]))
        output_file.write("\n")

    output_file.close()


if __name__ == "__main__":
    board = read_file()
    optimal_result = astar(board)
    make_output(algorithm="astar", solution=optimal_result)
    want_result = dfs(board)
    make_output(algorithm="dfs", solution=want_result)

