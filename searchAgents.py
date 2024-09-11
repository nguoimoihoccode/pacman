import math
import os
import numpy as np
from game import Directions
from game import Agent
from game import Actions
import util
import time
import random
import search

class RandomAgent(Agent):
    def getAction(self, state):
        legalActions = state.getLegalPacmanActions()
        if len(legalActions) > 0:
            return random.choice(legalActions)
        else:
            return Directions.STOP
        
'''
# TODO 03: SearchAgent
Implement a subclass of Agent class.
For each game step, getAction() method is invoked and 
the returned action is performed.
'''

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

class SearchAgent(Agent):
    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction is None: raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP
        
class SingleFoodSearchProblem:
    def __init__(self, costFn = lambda x: 1):
        #Node là 1 điểm có 1 giá trị với toạ độ x và y
        self.MT_origin = []
        #MT là ma trận lưu lại lần lượt các kí tự trong file input.
        #Ma trận này, theo lí thuyết về mảng 2 chiều trong lập trình, thì toạ độ sẽ bị đảo ngược.
        #Do đó, cần thêm một ma trận nữa để transpose lại để dùng được toạ độ:
    
        self.MT = np.array([])
        #Cách dùng toạ độ của ma trận đảo ngược ví dụ self.MT[11][3]='P' ( theo trục x,y )
        #Nếu theo ma trận ban đầu thì là self.MT_origin[3][11]='P (theo trục y,x)
        self.lenght = 0
        self.width = 0
        self.current=[]
        self.goal = []  #vị trí đích  (.)
        self.costFn = costFn
        #self.goals = []
    def __str__(self):
        line='' #line là chuỗi gồm các giá trị của mỗi dòng ví dụ như: "% %%        % %      %" là dòng index=1
        res = '' #Là chuỗi lưu toàn bộ ma trận (1 state)
        self.MT_origin = np.array(self.MT).T.tolist() # dòng này dùng để đảo ngược lại ma trận đã tính toán và lưu vào MT_origin để in ra
        # MT_origin dùng để in ra kết quả theo YC1-4
        for eachLine in self.MT_origin: 
            for value in eachLine: 
                line+=value 
            res+=line+"\n"
            line=''
        return res

    def print(self):
        print(str(self))

    def load_from_file(self, filename):
        if os.path.exists(filename):
            with open(filename) as g:
                y = 0
                for line in g: 
                    temp = []
                    x = 0
                    for position in line.strip(): 
                        temp.append(position)
                        if position == "P":
                            self.current = [x,y]
                        if position == ".":
                            self.goal = [x,y]
                        x+=1
                    self.MT_origin.append(temp) 
                    y+=1
                    #position là từng kí tự tại một line
                    #thêm từng line với kiểu ['kí tự 1', 'kí tự 2', ...]
               
                self.MT = np.array(self.MT_origin).T
                self.width = len(self.MT_origin)-1
                self.lenght = len(self.MT_origin)-1
                self.MT = self.MT.tolist()
    #def current()
    def successor(self, state):   #Trả về các giá trị ' ' liền kề của P hiện tại
        x, y = state
        #current = state
        dx = [-1,0,1,0]
        dy = [0,1,0,-1]
        # dx = [0,1,0,-1]
        # dy = [-1,0,1,0]
        result = []
        for i in range(4):
            x1 = x + dx[i]
            y1 = y + dy[i]
            if self.MT[x1][y1] != '%':
                if(i == 0):
                    result.append(([x1,y1], 'W'))
                if(i == 1):
                    result.append(([x1,y1], 'S') )
                if(i == 2):
                    result.append(([x1,y1], 'E') )
                if(i == 3):
                    result.append(([x1,y1], 'N') )
                # if(i == 0):
                #     result.append(([x1,y1], 'N'))
                # if(i == 1):
                #     result.append(([x1,y1], 'E') )
                # if(i == 2):
                #     result.append(([x1,y1], 'S') )
                # if(i == 3):
                #     result.append(([x1,y1], 'W') )
                
        return result   #result = [([x,y],"W") , ([x,y],"W")]     result[0][0][0]
    
   
    def successor_2(self, state):   #Trả về các giá trị ' ' liền kề của P hiện tại
        x, y = state
      
        dx = [-1,0,1,0]
        dy = [0,1,0,-1]
        result = []
        for i in range(4):
            x1 = x + dx[i]
            y1 = y + dy[i]
            nextState =(x1, y1)
            if self.MT[x1][y1] != '%':
                if(i == 0):
                    result.append((nextState, 'W', self.costFn(nextState)))
                if(i == 1):
                    result.append((nextState, 'S', self.costFn(nextState)) )
                if(i == 2):
                    result.append((nextState, 'E', self.costFn(nextState)) )
                if(i == 3):
                    result.append((nextState, 'N', self.costFn(nextState)) )
                
        return result   #result = [([x,y],"W") , ([x,y],"W")]     result[0][0][0]
    
    
 

    def animate(self, actions) -> None:
        tmp_list = list()
        # self.print()
        for action in actions:
            tmp_list.clear()
            os.system("cls")  #xoá màn hình ở đây
            self.print()   
            a = input()
            if action == 'N':
                tmp_list.append([self.current[0],self.current[1]])
                self.current[1] -= 1
                x = self.current[0]
                y = self.current[1]
                self.MT[x][y] = "P"
                tmp = tmp_list.pop()
                self.MT[tmp[0]][tmp[1]] = ' '
                self.print()
            if action == 'S':
                tmp_list.append([self.current[0],self.current[1]])
                self.current[1] += 1
                x = self.current[0]
                y = self.current[1]
                self.MT[x][y] = "P"
                tmp = tmp_list.pop()
                self.MT[tmp[0]][tmp[1]] = ' '
                self.print()
            if action == 'W':
                tmp_list.append([self.current[0],self.current[1]])
                self.current[0] -= 1
                x = self.current[0]
                y = self.current[1]
                self.MT[x][y] = "P"
                tmp = tmp_list.pop()
                self.MT[tmp[0]][tmp[1]] = ' '
                self.print()
            if action == 'E':
                tmp_list.append([self.current[0],self.current[1]])
                self.current[0] += 1
                x = self.current[0]
                y = self.current[1]
                self.MT[x][y] = "P"
                tmp = tmp_list.pop()
                self.MT[tmp[0]][tmp[1]] = ' '
                self.print()
            if action == 'STOP':
                break

    def goal_test(self,state):    #Hàm xác định kết quả của bài toán    #state [x,y]
        return True if self.goal == state else False
    
    def path_cost(self):
        #Chỗ này tụi mình chưa biết làm như nào nên tạm thời để vậy
        return []
# g = SingleFoodSearchProblem()
# g.load_from_file('input.txt')
# g.print()
# print(g.current())
# # print(g.goal_test())
# print(g.successor((12,6)))

class MultiFoodSearchProblem(SingleFoodSearchProblem):
    def __init__(self, costFn = lambda x: 1):
        self.MT_origin = []
        self.MT = np.array([])
        self.current = []
        self.goal= [] #list goal
        self.height = 0
        self.width = 0
        self.costFn = costFn

    def get_goal(self):       #Hàm dùng để xác định vị trí hiện tại của các điểm cuối '.'
        res=[]
        x = 0
        for line in self.MT: 
            y = 0
            for position in line:
                if position == '.':
                    res.append([x,y])
                y+=1
            x+=1
        return res
    
    def goal_test(self):                      #Hàm xác định kết quả của bài toán
        return (len(self.goal)==0)
    # def get_current(self):
    #     res=[]
    #     x = 0
    #     for line in self.MT: 
    #         y = 0
    #         for position in line:
    #             if position == 'P':
    #                 res = [x,y]
    #             y+=1
    #         x+=1
    #     return res
    def get_current(self): #state 
        for i in range(len(self.MT)): #current = state
            for j in range(len(self.MT[0])):
                if (self.MT[i][j]=="P"):
                    return [i,j]
        
    def load_from_file(self, filename):
        if os.path.exists(filename):
            with open(filename) as g:
                y = 0
                for line in g: 
                    temp = []
                    x = 0
                    for position in line.strip(): 
                        temp.append(position)
                        if position == "P":
                            self.current = [x,y]
                        if position == ".":
                            self.goal.append([x,y]) 
                        x+=1
                    self.MT_origin.append(temp) 
                    y+=1
                    #p9osition là từng kí tự tại một line
                    #thêm từng line với kiểu ['kí tự 1', 'kí tự 2', ...]
                
                self.MT = np.array(self.MT_origin).T
                self.width = len(self.MT_origin)-1
                self.lenght = len(self.MT_origin)-1
                self.MT = self.MT.tolist()

# g = MultiFoodSearchProblem()
# g.load_from_file('inputMulti.txt')
# g.animate(['N','N','S','S','Stop'])
# print(g.get_goal())

class EightQueenProblem:
    def __init__(self):
        self.MT = np.array([])
        self.MT_origin = []
        self.queens = []
    def __str__(self):
        line='' #line là chuỗi gồm các giá trị của mỗi dòng ví dụ như: "% %%        % %      %" là dòng index=1
        res = '' #Là chuỗi lưu toàn bộ ma trận (1 state)
        self.MT_origin = np.array(self.MT).T.tolist() # dòng này dùng để đảo ngược lại ma trận đã tính toán và lưu vào MT_origin để in ra
        # MT_origin dùng để in ra kết quả theo YC1-4
        for eachLine in self.MT_origin: 
            for value in eachLine: 
                line+=value +' ' 
            res+=line+"\n"
            res.strip()
            line=''
        return res

    def print(self):
        print(str(self))

    def load_from_file(self, filename):
        if os.path.exists(filename):
            with open(filename) as g:
                y = 0
                for line in g: 
                    temp = []
                    x = 0
                    for position in line.strip().split(" "): 
                        # print(position)
                        temp.append(position)
                        if position == "Q":
                            self.queens.append([x,y])
                        x+=1
                    self.MT_origin.append(temp) 
                    y+=1
                self.MT = np.array(self.MT_origin).T
                self.MT = self.MT.tolist()
    def attack_range(self,state):  #trả về danh sách các điểm con hậu state có thể đi
        result = []
        cut = []
        x,y = state
        for i in range(8):
            result.append([i,state[1]])  #đường ngang
            result.append([state[0],i])  #đường dọc
            for j in range(8):
                if i-j == x-y:  #đường chéo trên trái xuống dưới phải
                    result.append([i,j])
                if i+j == x+y:   #đường chéo dưới trái lên trên phải
                    result.append([i,j])
        #loại bỏ toạ độ trùng:
        for i in result:
            if i not in cut:
                cut.append(i)
        return cut
    # Q=[x,y]  k thay đổi theo hướng x và y
    # [x+k,y] k là số nguyên âm hoặc dương (đường ngang) #Chỉ dành cho trường hợp x và y trùng nhau 
    # [x,y+k] k là số nguyên âm hoặc dương (đường dọc)
    # [x+k,y+k] (xéo trên xuống từ trái sang phải)
    # [x+k,y-k] (xéo dưới lên từ trái sang phải)
    def h(self):
        visited=[]
        count=0 #dùng để đếm các số cặp hậu tấn công
        for q in self.queens:   #q là vị trí mỗi con hậu    
            visited.append(q) # thêm các con hậu đã được đếm rồi vào visited
            for position in self.attack_range(q): #position là vị trí các điểm con hậu đi được
                if position not in visited: # dùng để check xem cặp hậu này đã đếm chưa nếu chưa thì +1
                    if self.MT[position[0]][position[1]]=="Q":
                        count+=1    
        return count
    
q = EightQueenProblem()
q.load_from_file("./input/eight_queens01.txt")
print("attack range" , q.attack_range([0,1]))
print(q.MT)
q.print()

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

def ucs(problem):
    if type(problem)==pb.SingleFoodSearchProblem:
        goals = [problem.goal] 
        problem.goal = [problem.goal]
    if type(problem)==pb.MultiFoodSearchProblem:
        goals=problem.goal   #vị trí cuối      [[x,y],[x,y],...]

    frontier = fr.PriorityQueue()
    #node <- a node with State = problem.INITIAL-STATE, PATH-COST = 0
    node = { 'state': tuple(problem.current), 'cost': 0} #[x;y]
    #frontier <- a FIFO queue with node as the only element
    frontier.insert(node) 
    # explored <- an empty set
    explored = set()
    actions = []
    # loop do
    while True:
    #   if EMPTY? (frontier) then return failure
        if frontier.isEmpty():
            raise Exception('Search failed!')
    #   node <- POP(frontier) /* chooses the shallowest node in frontier */
        node = frontier.delete()

    # if problem.GOAL-TEST(node.STATE) then return SOLUTION(node)
        if list(node['state']) in goals:
            action = []
            temp ={'state': node['state'], 'cost': 0}
            goals.remove(list(node['state']))
            while 'parent' in node:
                # print(node['action'])
                action.append(node['action'])
                node = node['parent']
            action.reverse()
            actions.extend(action)
            if (len(goals)== 0):
                actions.append('STOP')
                return actions
            else:
                frontier = fr.PriorityQueue()
                frontier.insert(temp) 
                explored = set()
                node = frontier.delete()
                
    #   add node.STATE to explored
        explored.add(node['state'])

    # for each action in problem.ACTION( node.STA TE ) do
        successors = problem.successor_2(node['state'])

        for successor in successors:
    #   chil <- child-NODE (problem, node, action)
            child = {'state': tuple(successor[0]), 'action':successor[1], 'cost': successor[2] + node['cost'] , 'parent': node }
    #   if child.State is not in explored or frontier then
            temp ={'state': child['state'], 'cost': child['cost'],'action': child['action'], 'parent': node}
            if(child['state'] not in explored):
                frontier.insert_2(temp)
            else:
                frontier.change(temp)


def Euclide_heuristic(state: pb.SingleFoodSearchProblem):
    current = state.current
    if(type(state)==pb.SingleFoodSearchProblem):
        goal=state.goal[0]     #[[x,y]]
        # print(goal)
        return math.sqrt((current[0]-goal[0])**2+(goal[1]-current[1])**2)
    goal = state.get_goal()   #[[x,y], [x,y], ...]
    min = math.inf
    for i in goal: 
        res= math.sqrt((current[0]-i[0])**2+(i[1]-current[1])**2)
        if min>res:
            min=res
    return min

def Manhattan_heuristic(state):
    current = state.current
    goal = state.goal   #[[x,y], [x,y], ...]
    # print(goal)
    min = math.inf
    for i in goal: 
        res= abs(current[0]-i[0])+abs(i[1]-current[1])
        if min>res:
            min=res
    return min

def backtrace(parent,src,dst):  #dst node[0] backtrace([x1,y1], ([2, 4], 'E'), ) ([2, 4], 'E')
    path=[dst]
    while path[-1][0]!=src:
    #   print(path[-1],parent[tuple(path[-1][0])])
      path.append(parent[tuple(path[-1][0])])
    path.reverse()
    path.pop(0)
    
    kq= [i[1] for i in path]
    kq.append("STOP")
    return kq

    
def ASTAR(problem: pb.MultiFoodSearchProblem, fn_heuristic):
    if type(problem)==pb.SingleFoodSearchProblem:
        dst = [problem.goal] 
        problem.goal = [problem.goal]
    if type(problem)==pb.MultiFoodSearchProblem:
        dst=problem.goal   #vị trí cuối      [[x,y],[x,y],...]
    path=[]
    src=problem.current #vị trí nguồn   [x,y]
    dst=problem.goal  #vị trí cuối      [[x,y],[x,y],...]
    problem_new = copy.deepcopy(problem) 
    queue = fr.PriorityQueue()

    visited = [src]
    parent = {} 
    queue.insert([(src,""),fn_heuristic(problem)])
    check=False
    while queue.isEmpty()==False:
        u=queue.delete()  #u là 1 node lấy từ queue ra với cú pháp [(x,y),Vị trí] , fn_heuristic]
        road=u[1]-fn_heuristic(problem_new)
        count=0 #dùng để đếm số successor của u[0][0]
        if(u[0][0] in dst):
            path.extend(backtrace(parent,src,u[0]))
            dst.remove(u[0][0])
            problem_new.goal.remove(u[0][0])
            if(len(dst)>0):
                path.pop(len(path)-1)
                queue = fr.PriorityQueue()
                src=u[0][0]
                visited = [src]
                parent = {} 
                queue.insert([(src,""),fn_heuristic(problem_new)])
            else:
                return path
        for i in problem.successor(u[0][0]):
            if i[0] not in visited:
                count+=1
                #Ta cần trọng số fn_heuristic(state) tham số state truyền vào sẽ là vị trí next của current
                # problem_new = copy.deepcopy(problem)   #tạo state cho next của current bằng cách sao chép state cũ
                visited.append(i[0])
                parent[tuple(i[0])]=u[0] #lưu lại parent

                problem_new.current=i[0]
                temp=problem_new.MT[i[0][0]][i[0][1]] # swap giữa current và next của nó để được state mới
                problem_new.MT[i[0][0]][i[0][1]]=problem_new.MT[u[0][0][0]][u[0][0][1]]
                problem_new.MT[u[0][0][0]][u[0][0][1]]=temp

                queue.insert([i,road+fn_heuristic(problem_new)+1])

        if(count==0 and u[0][0] in dst):
            # problem_new = copy.deepcopy(problem)
            problem_new.current=u[0][0]
            temp=problem_new.MT[u[0][0][0]][u[0][0][1]] # swap giữa current và next của nó để được state mới
            problem_new.MT[u[0][0][0]][u[0][0][1]]=problem_new.MT[src[0]][src[1]]
            problem_new.MT[src[0]][src[1]]=' '
            if(u[0][0] in dst):
                path.extend(backtrace(parent,src,u[0]))
                dst.remove(u[0][0])
                problem_new.goal.remove(u[0][0])
                if(len(dst)>0):
                    path.pop(len(path)-1)
                    queue = fr.PriorityQueue()
                    src=u[0][0]
                    visited = [src]
                    parent = {} 
                    queue.insert([(src,""),fn_heuristic(problem_new)])
                else:
                    return path
    return path
