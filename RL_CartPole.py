"""
Cart Pole
    - discretize states space
    - removed cart position component
"""

import gym, collections
import numpy as np

verbose = False

# CartPole environment 
env = gym.make('CartPole-v0')

# Discount Rate
discountRate = 0.98

# Number of runs experiment
NR = 100

# Number of episodes to observe in a single experiment
NE = 2000

# Max time steps in episode
T = 200
env.max_episode_steps = T

# All states reached during experiment
allStates = []

# Observed environment states boundary
#minBoundaries = [-4.8, -4.8, -0.22, -3.6]
#maxBoundaries = [4.8, 4.8,  0.22, 3.6]
minBoundaries = [ -4.8, -0.22, -3.6]
maxBoundaries = [ 4.8,  0.22, 3.6]

# Map states space to ds discrete values
ds = 10
discreteSpace = [np.linspace(minBoundaries[i], maxBoundaries[i], ds) for i in range(len(maxBoundaries))]

# Inti explore result for calculate mean value
initExploreResult = 10

# Calculate mean value from lastNTry  
lastNTry = 30

# Minimal explore vs exploit rate
minimalEve = 0.1


def exploreRate(i):
    
    # Minimal Explore Vs Exploit rate 
    global minimalEve
    
    # Epsilon greedy 
    epsGreedy = False
    if epsGreedy:
        return minimalEve    
    
    rate = 1 - i * (1/NE)
    return rate if rate > minimalEve else minimalEve


def updateBoundaries(observation):
    
    global minBoundaries
    global maxBoundaries
    global discreteSpace

    if (minBoundaries>observation).any():
        print (observation)
    elif (maxBoundaries<observation).any():
        print (observation)
    else:    
        return        


def policyUpdate(idx,action):
    return


def qValuesUpdate(idx, action, reward):
 
    global qValues
    global policy
        
    if qValues[idx][action] < reward: # if new reward for state idx and action is higher then old
        
        if verbose:
            print ("Update {0}\taction {1}\treward {2}".format(idx,action,reward))
        
        qValues[idx][action] = reward # save new reward
        
        # update policy, set best action (0 or 1) for state idx   
        if (qValues[idx][0] > qValues[idx][1]):
            policy[idx] = 0
        else:
            policy[idx] = 1

    else:
        if verbose:
            print ("Best {0}\taction {1}\treward {2}".format(idx,policy[idx],qValues[idx][action]))
    
    return



""" 
 Get best action from policy for observed state. Policy value can be:
  -1 - unknow
  0  - move left
  1  - move right
""" 
def getBestAction(observation):

    global discreteSpace
    global maxBoundaries

    x = np.delete(observation, 0)

    idx = tuple([int(np.digitize(x[i], discreteSpace[i])) for i in range(len(maxBoundaries))])

    if (policy[idx] < 0): # unknow best policy for this state
        if verbose:
            print ("Unknow best {0} action, move random".format(idx))
        return np.random.randint(2) #return random action
    else:
        if verbose:
            print (idx, "Best action: {0}, qValue {1}".format(policy[idx],qValues[idx]))
        return policy [idx]

"""
 Run single episode 
"""
def runEpisode(env, T, eve, render = False):

    global allStates
    global discreteSpace
    global maxBoundaries
    global discountRate
 
    # All states in single episode
    episodeStates = []
    observation = env.reset()

    totalReward = 0

    for t in range(T):
    
        if verbose:
            print ("\nstep {0} eve {1}".format(t,eve))
 
        if eve > np.random.random(): #random explore
            action = env.action_space.sample()
            if verbose:
                print ("random move {0}".format(action))
        else: # get best action for curent state 
            if verbose:
                print ("get best policy action")
            action = getBestAction(observation)
        
        if (render):
            env.render()
            
        prevObservation = observation
        observation, reward, done, info = env.step(action)

        episodeStates.append([t,prevObservation, action, reward])


        if done:
            # backward calculate rewards for episode 
            for move in range(len (episodeStates)-1,0,-1):
                update = episodeStates[move][3] * discountRate ** (len (episodeStates)-1-move)

                if (update > 0.01): #minimal update that matter 
                    totalReward += update

                # add totalreward to states table
                episodeStates[move].append(totalReward)

                #add state to global states table  
                allStates.append(episodeStates[move])

                # get state without cart position 
                x = np.delete(episodeStates[move][1], 0)

                # get index of x in discretized states space
                idx = tuple([int(np.digitize(x[i], discreteSpace[i])) for i in range(len(maxBoundaries)) ])

                # update Q-values 
                # remember that from state idx doing action 'episodeStates[move][2]' you'll get totalReward       
                qValuesUpdate(idx,episodeStates[move][2],totalReward)

            return t



if __name__ == "__main__":

    for e in range(1,NR):        
        # Clear policy 
        qValues = np.zeros((ds,ds,ds,2))
        policy = np.full((ds,ds,ds),-1)
    
        # Table of 100 last exploit run results
        exploreResults = [initExploreResult] * 100

        mean = 0
        rate = 10
        maxi = 0

        # In one run try NE episodes
        for i in range(1,NE):
        
            rate = (sum(exploreResults[-lastNTry:])/lastNTry)*(1.0/T)

            if (rate < np.random.random()):
                # random decision explore or exploit base on explore rate 
                t = runEpisode(env, T, exploreRate(i))
            else: 
                # exploite
                t = runEpisode(env, T, 0)
                
                #remove oldest exploit result
                exploreResults.pop(0)                
                
                # and append newest exploit result
                exploreResults.append(t)
                
                # calculate new mean from 100 
                mean = np.mean(exploreResults)
 
                if mean > maxi:
                    maxi = mean 

        print ("Run {} max last 100 explore try mean {}".format(e,maxi))

        res = runEpisode(env, T, 0, True)
        print ("Test: score {}".format(t))
    
    
    print (minBoundaries)
    print (maxBoundaries)
    count = np.unique(policy, return_counts=True)
    print (count)
