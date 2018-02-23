from enum import Enum
from functools import reduce
import numpy as np
import collections


class Hex:
    def __init__(self, typeString):
        self.unknown = False
        self.type, self.number = self.parseTypeString(typeString)

    def __str__(self):
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
    def __init__(self, items, value):
        self.items = items
        self.value = value

    def num_unsolved(self):
        return len([h for h in self.items if h.unknown])

    def num_solved(self):
        return len([h for h in self.items if h.type == Hex.Type.BLUE and not h.unknown])

    def get_solved_locations(self):
        return [i for i, h in enumerate(self.items) if h.type == Hex.Type.BLUE and not h.unknown]

    def __str__(self):
        return 'val: ' + str(self.value.number) + ' items: ' + reduce(lambda a, b: str(a) + str(b), self.items)


class Board:
    def __init__(self):
        self.numUnknown = 0
        self.columns, self.diags, self.areas = self.parse_input()

    def __str__(self):
        board = ''
        for line in self.columns:
            board += reduce(lambda a, b: str(a) + str(b), line) + '\n'
        return board

    def get_adj_bounds(self, leftmost, rightmost, constraint, wrap):
        n = constraint.num_solved()
        if wrap:
            right = (rightmost + constraint.value.number - n) % len(constraint.items)
            left = leftmost - constraint.value.number + n
            if left < 0:
                left = len(constraint.items) - abs(left)
            return left, right
        else:
            pass

    def get_hexes_between(self, left, right, index, constraint, wrap):
        betweenRight = []
        betweenLeft = []
        if wrap:
            if left > index:
                betweenLeft = constraint.items[index + 1:left + 1]
            elif left < index:
                betweenLeft = constraint.items[index + 1:] + constraint.items[:left + 1]
            if right > index:
                betweenRight = constraint.items[right:] + constraint.items[:index]
            elif right < index:
                betweenRight = constraint.items[right:index]
            return betweenLeft, betweenRight
        else:
            pass

    def out_of_range(self, left, right, index, wrap):
        if wrap:
            if left < right:
                return index > right or index < left
            else:
                return right < index < left
        else:
            pass

    def solve(self):
        constraints = self.get_constraints()
        while not self.solved():
            for k, val in constraints.items():
                for v in val:
                    print(str(self))
                    if k.unknown:
                        if v.value.number - v.num_solved() == 0:
                            k.unknown = False
                            self.numUnknown -= 1
                        elif v.value.number - v.num_solved() == v.num_unsolved():
                            k.unknown = False
                            self.numUnknown -= 1
                        elif v.value.type == Hex.Type.ADJ:
                            if v.num_solved() > 0:
                                locs = v.get_solved_locations()
                                index = v.items.index(k)
                                left, right = self.get_adj_bounds(locs[0], locs[-1], v, True)
                                # is there a black hex between this hex and the solved ones?
                                betweenLeft, betweenRight = self.get_hexes_between(left, right, index, v, True)
                                blackLeft = len([h for h in betweenLeft if h.is_black()]) > 0
                                blackRight = len([h for h in betweenRight if h.is_black()]) > 0
                                # is this hex too far away to be a part of the solved group?
                                dist = self.out_of_range(left, right, index, True)
                                # add case where one side has black and other is too far
                                between = self.out_of_range(right, left, index, True)
                                # if left > right:
                                #     between = index <= right or index >= left
                                # else:
                                #     between = left <= index <= right
                                if dist or (not between and blackLeft) or (not between and blackRight):
                                    print(str(index))
                                    print(str(left))
                                    print(str(right))
                                    print(reduce(lambda a, b: str(a) + ' ' + str(b), betweenRight))
                                    print(str(dist) + ' ' + str(not between and blackLeft) + ' ' + str(not between and blackRight))
                                    k.unknown = False
                                    self.numUnknown -= 1
                        elif v.value.type == Hex.Type.APART:
                            locs = v.get_solved_locations()
                            if len(locs) == v.value.number - 1:
                                together = locs[-1] == locs[0] + v.value.number - 2
                                if not together and locs[0] == 0:
                                    prev = locs[0]
                                    index = 0
                                    for i in locs[1:]:
                                        if i != prev + 1:
                                            break
                                        prev = i
                                        index += 1
                                    together = locs[index + 1] == len(v.items) - (v.value.number - 1 - (index + 1))
                                if together:
                                    if (locs[0] == 0 and v.items[-1] == k) or v.items[locs[0] - 1] == k or \
                                            (locs[-1] == len(v.items) - 1 and v.items[0] == k) or \
                                            v.items[(locs[-1] + 1) % len(v.items)]:
                                        k.unknown = False
                                        self.numUnknown -= 1
            constraints = self.recalculate_constraints()

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
        self.areas = Board.get_areas(self.columns)
        return self.get_constraints()

    def parse_input(self):
        f = open('input3.txt', 'r')
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
        return parseLines, Board.get_diagonals(parseLines), Board.get_areas(parseLines)

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
    def get_areas(columns):
        areas = []
        for i, c in enumerate(columns):
            for j, h in enumerate(c):
                if h.is_num():
                    areas.append(Grouping(Board.get_neighbors(i, j, columns), h))
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
    b = Board()
    b.solve()
    print(str(b))


if __name__ == "__main__":
    main()
