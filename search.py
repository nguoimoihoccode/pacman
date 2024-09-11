'''
# TODO 01: Problem formulation
Implement a class representing the problem
'''

'''
# TODO 02: Search strategies
Implement a class with methods as search strategies
'''

import util

class SearchProblem:
    def getStartState(self):
        util.raiseNotDefined()

    def isGoalState(self, state):
        util.raiseNotDefined()

    def getSuccessors(self, state):
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        util.raiseNotDefined()



def uniformCostSearch(problem):
    startingNode = problem.getStartState()
    if problem.isGoalState(startingNode):
        return []
    visitedNodes = []
    priorityQueue = util.PriorityQueue()
    priorityQueue.push((startingNode, [], 0), 0)

    while not  priorityQueue.isEmpty():
        currentNode, actions, oldCost =  priorityQueue.pop()
        if currentNode not in visitedNodes:
            visitedNodes.append(currentNode)
            if problem.isGoalState(currentNode):
                return actions
            for nextNode, action, cost in problem.getSuccessors(currentNode):
                nextAction = actions + [action]
                priority = oldCost + cost
                priorityQueue.push((nextNode, nextAction, priority), priority)

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    startingNode = problem.getStartState()
    if problem.isGoalState(startingNode):
        return []

    visitedNodes = []

    priorityQueue = util.PriorityQueue()
    priorityQueue.push((startingNode, [], 0), 0)

    while not priorityQueue.isEmpty():
        currentNode, actions, oldCost = priorityQueue.pop()
        if currentNode not in visitedNodes:
            visitedNodes.append(currentNode)
            if problem.isGoalState(currentNode):
                return actions
            for nextNode, action, cost in problem.getSuccessors(currentNode):
                nextAction = actions + [action]
                newCostToNode = oldCost + cost
                heuristicCost = newCostToNode + heuristic(nextNode, problem)
                priorityQueue.push((nextNode, nextAction, newCostToNode), heuristicCost)

    util.raiseNotDefined()


astar = aStarSearch
ucs = uniformCostSearch