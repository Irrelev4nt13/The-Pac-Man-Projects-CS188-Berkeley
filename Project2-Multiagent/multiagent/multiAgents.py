# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
from pacman import GameState


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def getAction(self, gameState: GameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState: GameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"

        """ On https://pacman.fandom.com/wiki/Point_Configurations we can see the following
        point distribution which I'm gonna use to make my evaluationFunction.
        Pac-Dot = 10 Pts
        1st Ghost = 200 Pts
        2nd Ghost = 400 Pts
        3rd Ghost = 800 Pts
        4th Ghost = 1600 Pts"""

        dead_ghosts = 0
        total_score = 0
        # If the next state is a win state then return infinity in order to "show" that is a good move.
        if successorGameState.isWin():
            return float("inf")
        # If the next state is a win state then return -infinity in order to "show" that is a bad move.
        if successorGameState.isLose():
            return float("-inf")
        # Calculates the manhattanDistance from next pacman's position to all foods and take the minimum one.
        # Divide 10 by that distance and then add it to total_score.
        food_distances = {food: util.manhattanDistance(newPos, food) for food in newFood.asList()}
        food, min_food_dist = min(food_distances.items(), key=lambda data: data[1])
        total_score += 10 / min_food_dist
        # For every ghost in the next game state calculates the distance from pacman.
        # Ιf there is a ghost whose distance is less than 2 so it is close to the pacman then it checks if it is scared 
        # Αnd acts accordingly by returning -infinity if it is not to "show" that he will die or 200*2^number of dead ghosts.
        for i in range(len(successorGameState.getGhostPositions())):
            if util.manhattanDistance(newPos, successorGameState.getGhostPositions()[i]) < 2:
                if newScaredTimes[i] != 0:
                    total_score += 200 * (2 ** dead_ghosts)
                    dead_ghosts += 1
                else:
                    return float("-inf")
        # Return the result.
        return successorGameState.getScore() + total_score


def scoreEvaluationFunction(currentGameState: GameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn='scoreEvaluationFunction', depth='2'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)


class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def minimax_decision(self, depth, agentIndex, gameState: GameState):
        # If all agents has play increase depth and return to pacman.
        if agentIndex == gameState.getNumAgents():
            depth += 1
            agentIndex = 0
        # Check if the resursion has come to an end.
        if self.depth == depth or gameState.isWin() or gameState.isLose():
            return [self.evaluationFunction(gameState), None]
        # If the agent is the pacman.
        if agentIndex == 0:
            # Initialize the value and the action
            v, v_action = float("-inf"), None
            for action in gameState.getLegalActions(agentIndex):
                # Take the new value.
                v_new = self.minimax_decision(depth, agentIndex + 1, gameState.generateSuccessor(agentIndex, action))[0]
                # Change the new value and the action only if the new value is greater.
                (v, v_action) = (v_new, action) if v_new > v else (v, v_action)
            # Return the result.
            return [v, v_action]
        # If the agent is a ghost.
        else:
            # Initialize the value and the action.
            v, v_action = float("inf"), None
            for action in gameState.getLegalActions(agentIndex):
                # Take the new value.
                v_new = self.minimax_decision(depth, agentIndex + 1, gameState.generateSuccessor(agentIndex, action))[0]
                # Change the new value and the action only if the new value is smaller.
                (v, v_action) = (v_new, action) if v_new < v else (v, v_action)
            # Return the result.
            return [v, v_action]

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

        gameState.getNumAgents():
        Returns the total number of agents in the game

        gameState.isWin():
        Returns whether or not the game state is a winning state

        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        return self.minimax_decision(0, 0, gameState)[1]


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def alpha_beta_search(self, alpha, beta, depth, agentIndex, gameState: GameState):
        # If all agents has play increase depth and return to pacman.
        if agentIndex == gameState.getNumAgents():
            depth += 1
            agentIndex = 0
        # Check if the resursion has come to an end.
        if self.depth == depth or gameState.isWin() or gameState.isLose():
            return [self.evaluationFunction(gameState), None]
        # If the agent is the pacman.
        if agentIndex == 0:
            # Initialize the value and the action
            v, v_action = float("-inf"), None
            for action in gameState.getLegalActions(agentIndex):
                # Take the new value.
                v_new = self.alpha_beta_search(alpha, beta, depth, agentIndex + 1,
                                               gameState.generateSuccessor(agentIndex, action))[0]
                # Change the new value and the action only if the new value is greater.
                (v, v_action) = (v_new, action) if v_new > v else (v, v_action)
                # Checks if it needs to stop and return.
                if v > beta:
                    return [v, v_action]
                # Find new alpha
                alpha = max(alpha, v)
            # Return the result.
            return [v, v_action]
        else:
            # Initialize the value and the action
            v, v_action = float("inf"), None
            for action in gameState.getLegalActions(agentIndex):
                # Take the new value.
                v_new = self.alpha_beta_search(alpha, beta, depth, agentIndex + 1,
                                               gameState.generateSuccessor(agentIndex, action))[0]
                # Change the new value and the action only if the new value is smaller.
                (v, v_action) = (v_new, action) if v_new < v else (v, v_action)
                # Checks if it needs to stop and return.
                if v < alpha:
                    return [v, v_action]
                # Find new beta
                beta = min(beta, v)
            # Return the result.
            return [v, v_action]

    def getAction(self, gameState: GameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        return self.alpha_beta_search(float("-inf"), float("inf"), 0, 0, gameState)[1]


class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def expectimax_decision(self, depth, agentIndex, gameState: GameState):
        # If all agents has play increase depth and return to pacman.
        if agentIndex == gameState.getNumAgents():
            depth += 1
            agentIndex = 0
        # Check if the resursion has come to an end.
        if self.depth == depth or gameState.isWin() or gameState.isLose():
            return [self.evaluationFunction(gameState), None]
        # If the agent is the pacman.
        if agentIndex == 0:
            # Initialize the value and the action
            v, v_action = float("-inf"), None
            for action in gameState.getLegalActions(agentIndex):
                # Take the new value.
                v_new = \
                    self.expectimax_decision(depth, agentIndex + 1, gameState.generateSuccessor(agentIndex, action))[0]
                # Change the new value and the action only if the new value is greater.
                (v, v_action) = (v_new, action) if v_new > v else (v, v_action)
            # Return the result.
            return [v, v_action]
        # If the agent is a ghost.
        else:
            # Initialize sum.
            v = 0
            for action in gameState.getLegalActions(agentIndex):
                # Increase the sum.
                v += self.expectimax_decision(depth, agentIndex + 1, gameState.generateSuccessor(agentIndex, action))[0]
            # Return the result as the average of the sum.
            return [v / len(gameState.getLegalActions(agentIndex))]

    def getAction(self, gameState: GameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        return self.expectimax_decision(0, 0, gameState)[1]


def betterEvaluationFunction(currentGameState: GameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    """ On https://pacman.fandom.com/wiki/Point_Configurations we can see the following
    point distribution which im gonna use to make my evaluationFunction and the betterEvaluationFunction.
    Pac-Dot = 10 Pts
    Power Pellet = 50 Pts
    1st Ghost = 200 Pts
    2nd Ghost = 400 Pts
    3rd Ghost = 800 Pts
    4th Ghost = 1600 Pts
    I'm also gonna add a -1 pts for every second the pacman does not move. The idea was taken by the current pacman implementation
    where i spotted this when i was playing the game with the command python3 pacman.py
    'Stop' = -1 Pts """

    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    dead_ghosts = 0
    total_score = 0
    # If the next state is a win state then return infinity in order to "show" that is a good move.
    if currentGameState.isWin():
        return float("inf")
    # If the next state is a win state then return -infinity in order to "show" that is a bad move.
    if currentGameState.isLose():
        return float("-inf")
    # Calculates the manhattanDistance from next pacman's position to all foods and take the minimum one.
    # Divide 10 by that distance and then add it to total_score.
    food_distances = {food: util.manhattanDistance(newPos, food) for food in newFood.asList()}
    food, min_food_dist = min(food_distances.items(), key=lambda data: data[1])
    total_score += 10 / min_food_dist
    # Decreasing the total score by -1 for every Stop action it finds.
    total_score -= (-1) * currentGameState.getLegalActions().count('Stop')
    # Get the list with pellet.
    pellet_list = currentGameState.getCapsules()
    # Checks if the pellet list has at least one pellet.
    if pellet_list:
        # Calculates the manhattanDistance from next pacman's position to all pellets and take the minimum one.
        # Divide 50 by that distance and then add it to total_score.
        pellet_distances = {pellet: util.manhattanDistance(newPos, pellet) for pellet in pellet_list}
        pellet, min_pellet_dist = min(pellet_distances.items(), key=lambda data: data[1])
        total_score += 50 / min_pellet_dist
    # For every ghost in the next game state calculates the distance from pacman.
    # Ιf there is a ghost whose distance is less than 2 so it is close to the pacman then it checks if it is scared. 
    # Αnd acts accordingly by dividing the total score in half if it is not to "show" that he will die.
    # Or 200*2^number of dead ghosts.
    for i in range(len(currentGameState.getGhostPositions())):
        if util.manhattanDistance(newPos, currentGameState.getGhostPositions()[i]) < 2:
            if newScaredTimes[i] != 0:
                total_score += 200 * (2 ** dead_ghosts)
                dead_ghosts += 1
            else:
                total_score /= 2
    # Return the result.
    return currentGameState.getScore() + total_score


# Abbreviation
better = betterEvaluationFunction
