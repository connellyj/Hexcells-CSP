from enum import Enum
from functools import reduce
import numpy as np
import collections


class Hex:
    def __init__(self, typeString):
        self.index = None
        self.unknown = False
        self.type, self.number = self.parseTypeString(typeString)
        self.error = False

    def __str__(self):
        if self.error:
            return '?'
        if self.unknown:
            return Hex.Type.GOLD.value
        if self.is_num():
            return self.type.value + str(self.number)
        return self.type.value

    def parseTypeString(self, s):
        if s[0] == Hex.Type.SECRET.value:
            self.unknown = True
            if len(s) == 2:
                if s[1] == Hex.Type.QUESTION.value:
                    return Hex.Type.QUESTION, None
                else:
                    return Hex.Type.NUM, int(s[1])
            if len(s) == 3:
                if s[1] == Hex.Type.ADJ.value:
                    return Hex.Type.ADJ, int(s[2])
                if s[1] == Hex.Type.APART.value:
                    return Hex.Type.APART, int(s[2])
        if len(s) == 2:
            if s[0] == Hex.Type.ADJ.value:
                return Hex.Type.ADJ, int(s[1])
            if s[0] == Hex.Type.APART.value:
                return Hex.Type.APART, int(s[1])
        if s == Hex.Type.GOLD.value:
            self.unknown = True
            return Hex.Type.BLUE, None
        if s == Hex.Type.BLUE.value:
            return Hex.Type.BLUE, None
        if s == Hex.Type.QUESTION.value:
            return Hex.Type.QUESTION, None
        if s == Hex.Type.SPACE.value:
            return Hex.Type.SPACE, None
        if s == Hex.Type.EMPTY.value:
            return Hex.Type.EMPTY, None
        return Hex.Type.NUM, int(s)

    def is_black(self):
        return (self.type == Hex.Type.NUM or self.type == Hex.Type.ADJ or self.type == Hex.Type.APART
                or self.type == Hex.Type.QUESTION) and not self.unknown

    def is_num(self):
        return (self.type == Hex.Type.NUM or self.type == Hex.Type.ADJ or self.type == Hex.Type.APART) \
               and not self.unknown

    class Type(Enum):
        BLUE = 's'
        GOLD = 'y'
        NUM = ''
        EMPTY = '_'
        SPACE = ' '
        SECRET = '.'
        ADJ = '{'
        APART = '-'
        QUESTION = 'b'


class Grouping:
    def __init__(self, items, value, wrap):
        self.items = items
        self.value = value
        self.wrap = wrap

    def num_unsolved(self):
        return len([h for h in self.items if h.unknown])

    def num_solved(self):
        return len([h for h in self.items if h.type == Hex.Type.BLUE and not h.unknown])

    def get_solved_locations(self):
        return [i for i, h in enumerate(self.items) if h.type == Hex.Type.BLUE and not h.unknown]

    def is_adjacent_to_solved(self, index):
        locs = self.get_solved_locations()
        if locs:
            if self.wrap:
                lowCompare = locs[0] - 1 if locs[0] != 0 else len(self.items) - 1
                highCompare = locs[-1] + 1 % len(self.items)
                return index == lowCompare or index == highCompare
            else:
                return index == locs[0] - 1 or index == locs[-1] + 1
        return True

    def __str__(self):
        return 'val: ' + str(self.value.number) + ' items: ' + reduce(lambda a, b: str(a) + str(b), self.items)


class Board:
    DIAG = 0
    COL = 1

    def __init__(self, fileName):
        self.numUnknown = 0
        self.columns, self.diags, self.lineConstraints, self.areas = self.parse_input(fileName)

    def __str__(self):
        board = ''
        for line in self.columns:
            board += reduce(lambda a, b: str(a) + str(b), line) + '\n'
        return board

    @staticmethod
    def get_adj_bounds(leftmost, rightmost, constraint):
        n = constraint.num_solved()
        if constraint.wrap:
            right = (rightmost + constraint.value.number - n) % len(constraint.items)
            left = leftmost - constraint.value.number + n
            if left < 0:
                left = len(constraint.items) - abs(left)
        else:
            right = rightmost + constraint.value.number - n
            right = len(constraint.items) - 1 if right > len(constraint.items) - 1 else right
            left = leftmost - constraint.value.number + n
            left = 0 if left < 0 else left
        return left, right

    @staticmethod
    def get_hexes_between(left, right, index, constraint):
        betweenRight = []
        betweenLeft = []
        if constraint.wrap:
            if left > index:
                betweenLeft = constraint.items[index + 1:left]
            elif left < index:
                betweenLeft = constraint.items[index + 1:] + constraint.items[:left]
            if right > index:
                betweenRight = constraint.items[right + 1:] + constraint.items[:index]
            elif right < index:
                betweenRight = constraint.items[right + 1:index]
        else:
            if left > index:
                betweenLeft = constraint.items[index + 1:left]
            elif right < index:
                betweenRight = constraint.items[right + 1:index]
        return betweenLeft, betweenRight

    @staticmethod
    def out_of_range(left, right, index, wrap):
        if wrap:
            if left < right:
                return index > right or index < left
            else:
                return right < index < left
        else:
            return left > index < right or index > right

    @staticmethod
    def is_continuous(locs, constraint):
        if not locs:
            return False
        together = locs[-1] == locs[0] + constraint.value.number - 2
        if constraint.wrap and not together and locs[0] == 0:
            prev = locs[0]
            index = 0
            for i in locs[1:]:
                if i != prev + 1:
                    break
                prev = i
                index += 1
            together = locs[index] == len(constraint.items) - (constraint.value.number - 1 - (index + 1))
        return together

    def solve_basic_constraints(self):
        constraints = self.get_constraints()
        constraintLoopCount = 0
        while not self.solved():
            constraintLoopCount += 1
            changeMade = False
            for k, val in constraints.items():
                for v in val:
                    if k.unknown:
                        # If all solved, this hex is black
                        if v.value.number - v.num_solved() == 0:
                            k.unknown = False
                            changeMade = True
                            self.numUnknown -= 1
                        # If there are as many gold hexes as hexes that need to be blue, this hex is blue
                        elif v.value.number - v.num_solved() == v.num_unsolved():
                            k.unknown = False
                            changeMade = True
                            self.numUnknown -= 1
                        # If this hex is out of range to be adjacent to other solved hexes, this hex is black
                        elif v.value.type == Hex.Type.ADJ:
                            if v.num_solved() > 0:
                                locs = v.get_solved_locations()
                                index = v.items.index(k)
                                # is this hex too far away to be a part of the solved group?
                                left, right = Board.get_adj_bounds(locs[0], locs[-1], v)
                                dist = Board.out_of_range(left, right, index, v.wrap)
                                # is there a black hex between this hex and the solved ones?
                                betweenLeft, betweenRight = Board.get_hexes_between(locs[0], locs[-1], index, v)
                                blackLeft = len([h for h in betweenLeft if h.is_black()]) > 0
                                blackRight = len([h for h in betweenRight if h.is_black()]) > 0
                                distRight = Board.out_of_range(
                                    right, len(v.items) - 1 if left - 1 < 0 else left - 1, index, v.wrap
                                )
                                distLeft = Board.out_of_range(
                                    right + 1 % len(v.items), left, index, v.wrap
                                )
                                if dist or (not distLeft and blackLeft) or (not distRight and blackRight):
                                    k.unknown = False
                                    changeMade = True
                                    if not k.is_black():
                                        print("ERROR! Adjacent solved incorrectly.")
                                        k.error = True
                                    self.numUnknown -= 1
                        # If making this hex blue would make all solved hexes adjacent, this hex is black
                        elif v.value.type == Hex.Type.APART:
                            locs = v.get_solved_locations()
                            if len(locs) == v.value.number - 1:
                                together = Board.is_continuous(locs, v)
                                if together:
                                    if (locs[0] == 0 and v.items[-1] == k) or v.items[locs[0] - 1] == k or \
                                            (locs[-1] == len(v.items) - 1 and v.items[0] == k) or \
                                            v.items[(locs[-1] + 1) % len(v.items)] == k:
                                        k.unknown = False
                                        changeMade = True
                                        if not k.is_black():
                                            print("ERROR! Apart solved incorrectly.")
                                            k.error = True
                                        self.numUnknown -= 1
            # Can't solve anymore
            if not changeMade:
                return
            # Update constraints with new revealed constraints
            constraints = self.recalculate_constraints()
        print('Number of times looped through all constraints: ' + str(constraintLoopCount))

    def get_unsolved_hex(self, alreadyGuessed):
        for i in range(len(self.columns)):
            for j in range(len(self.columns[i])):
                cur = self.columns[i][j]
                if cur.unknown and cur not in alreadyGuessed:
                    cur.index = (i, j)
                    return cur
        return None

    def solve_search(self):
        print('search started')
        constraints = self.get_constraints()
        BLACK = 0
        BLUE = 1
        nextToGuess = self.get_unsolved_hex([])
        frontier = [
            {nextToGuess: BLUE},
            {nextToGuess: BLACK}
        ]
        visited = []
        while frontier:
            cur = frontier.pop(0)
            visited.append(cur)
            nextToGuess = self.get_unsolved_hex(list(cur.keys()))
            # There's nothing left to guess so we're done
            if not nextToGuess:
                for k, val in cur.items():
                    h = self.columns[k.index[0]][k.index[1]]
                    h.type = Hex.Type.BLUE if val == BLUE else Hex.Type.QUESTION
                    h.unknown = False
                return
            # Check if the chosen hex can be blue or black and still be a valid board
            blueValid = True
            blackValid = True
            for g in constraints[nextToGuess]:
                numSolved = g.num_solved()
                numUnsolved = g.num_unsolved()
                for h in g.items:
                    if h in cur.keys():
                        numUnsolved -= 1
                        if cur[h] == BLUE:
                            numSolved += 1
                if g.value.number - numSolved == numUnsolved:
                    blackValid = False
                if g.value.number == numSolved:
                    blueValid = False
                if blueValid and g.value.type == Hex.Type.ADJ:
                    index = g.items.index(nextToGuess)
                    if not g.is_adjacent_to_solved(index):
                        blueValid = False
                if blueValid and g.value.type == Hex.Type.APART:
                    locs = g.get_solved_locations()
                    continuous = Board.is_continuous(locs, g)
                    if locs and continuous:
                        index = g.items.index(nextToGuess)
                        if g.is_adjacent_to_solved(index):
                            blueValid = False
            # Add the new nodes to the frontier
            if blueValid:
                blue = cur.copy()
                blue[nextToGuess] = BLUE
                if blue not in visited:
                    frontier.insert(0, blue)
            if blackValid:
                black = cur.copy()
                black[nextToGuess] = BLACK
                if black not in visited:
                    frontier.insert(0, black)

    def solve(self):
        self.solve_basic_constraints()
        if not self.solved():
            self.solve_search()

    def solved(self):
        return self.numUnknown == 0

    def make_hexes(self, ls):
        hexes = []
        for i in ls:
            h = Hex(i)
            hexes.append(h)
            if h.unknown:
                self.numUnknown += 1
        return hexes

    def get_constraints(self):
        constraints = collections.defaultdict(set)
        for g in self.areas:
            for h in g.items:
                if h.unknown:
                    constraints[h].add(g)
        return constraints

    def recalculate_constraints(self):
        self.areas = Board.get_areas(self.columns, self.diags, self.lineConstraints)
        return self.get_constraints()

    def parse_input(self, fileName):
        f = open(fileName, 'r')
        file = f.read()
        data = file.split('+')
        lines = data[0].split('\n')
        splitLines = list(map(lambda l: l.split(' '), lines))
        for line in splitLines:
            for i in range(len(line) - 1, -1, -1):
                if line[i] == '':
                    line[i] = ' '
                else:
                    line.insert(i + 1, ' ')
        parseLines = list(map(lambda l: self.make_hexes(l), splitLines))
        maxLength = len(max(parseLines, key=lambda l: len(l)))
        for line in parseLines:
            while len(line) < maxLength:
                line.append(Hex(' '))
        lineConstraints = []
        for lC in data[1:]:
            lines = lC.split('\n')
            constraints = {}
            for c in lines:
                if c != '':
                    splitLine = c.split(' ')
                    constraints[int(splitLine[0])] = Hex(splitLine[1])
            lineConstraints.append(constraints)
        diagonals = Board.get_diagonals(parseLines)
        return parseLines, diagonals, lineConstraints, Board.get_areas(parseLines, diagonals, lineConstraints)

    @staticmethod
    def get_neighbors(col, row, columns):
        neighbors = [Hex(Hex.Type.SPACE.value)] * 6
        if row + 1 < len(columns[col]):
            if col - 1 >= 0:
                neighbors[0] = columns[col - 1][row + 1]
            if col + 1 < len(columns):
                neighbors[2] = columns[col + 1][row + 1]
        if row - 1 >= 0:
            if col - 1 >= 0:
                neighbors[5] = columns[col - 1][row - 1]
            if col + 1 < len(columns):
                neighbors[3] = columns[col + 1][row - 1]
        if row + 2 < len(columns[col]):
            neighbors[1] = columns[col][row + 2]
        if row - 2 >= 0:
            neighbors[4] = columns[col][row - 2]
        return neighbors

    @staticmethod
    def get_areas(columns, diagonals, lineConstraints):
        areas = []
        for i, c in enumerate(columns):
            for j, h in enumerate(c):
                if h.is_num():
                    areas.append(Grouping(Board.get_neighbors(i, j, columns), h, True))
        for i, constraints in enumerate(lineConstraints):
            lines = columns if i == Board.COL else diagonals
            for k in list(constraints.keys()):
                areas.append(Grouping(lines[k], constraints[k], False))
        return areas

    @staticmethod
    def get_diagonals(columns):
        # see https://stackoverflow.com/questions/6313308/get-all-the-diagonals-in-a-matrix-list-of-lists-in-python
        npArr = np.asarray(columns)
        diags = [npArr[::-1, :].diagonal(i) for i in range(-npArr.shape[0] + 1, npArr.shape[1])]
        diags.extend(npArr.diagonal(i) for i in range(npArr.shape[1] - 1, -npArr.shape[0], -1))
        # remove empty diagonals
        return filter(lambda ls: len([h for h in ls if h.type == Hex.Type.SPACE]) != len(ls), diags)


def main():
    for i in range(1, 7):
        print('--------------- ' + str(i) + ' ---------------')
        b = Board('input' + str(i) + '.txt')
        print('unsolved')
        print(str(b))
        print()
        b.solve()
        print('solved:')
        print(str(b))
        print()


if __name__ == "__main__":
    main()
