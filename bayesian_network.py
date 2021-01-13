""" Bayesian networks """

from probability import BayesNet, enumeration_ask, elimination_ask, rejection_sampling, likelihood_weighting, gibbs_ask
from timeit import timeit, repeat
import pickle
import numpy as np

T, F = True, False

class DataPoint:
    """
    Represents a single datapoint gathered from one lap.
    Attributes are exactly the same as described in the project spec.
    """
    def __init__(self, muchfaster, early, overtake, crash, win):
        self.muchfaster = muchfaster
        self.early = early
        self.overtake = overtake
        self.crash = crash
        self.win = win

def generate_bayesnet():
    """
    Generates a BayesNet object representing the Bayesian network in Part 2
    returns the BayesNet object
    """
    bayes_net = BayesNet()
    # load the dataset, a list of DataPoint objects
    data = pickle.load(open("data/bn_data.p","rb"))
    # BEGIN_YOUR_CODE ######################################################
    count_occ = {'MuchFaster': 0,'Early': 0,'Overtake': 0,'Crash': 0,'Win': 0}
    perms = [[[0 for k in range(3)] for j in range(2)] for i in range(2)] #2x2x3 array:
    for p in data:
        control = [0,0,0,0]
        if(p.muchfaster):
            count_occ["MuchFaster"] += 1
            control[0] = 1
        if(p.early):
            count_occ["Early"] += 1 
            control[1] = 1
        if(p.overtake):
            count_occ["Overtake"] += 1
            perms[control[0]][control[1]][0] += 1
            control[2] = 1
        if(p.crash):
            count_occ["Crash"] += 1 
            perms[control[0]][control[1]][1] += 1
            control[3] = 1
        if(p.win):
            count_occ["Win"] += 1
            perms[control[2]][control[3]][2] += 1
    tot_attempts = len(data)

    T, F = True, False
    bayes_net = BayesNet([
    ('MuchFaster', '', count_occ["MuchFaster"]/tot_attempts),
    ('Early', '', count_occ["Early"]/tot_attempts),
    ('Overtake', 'MuchFaster Early',
    {(T, T): perms[1][1][0]/tot_attempts, (T, F): perms[1][0][0]/tot_attempts, (F, T): perms[0][1][0]/tot_attempts, (F, F): perms[0][0][0]/tot_attempts}),
    ('Crash', 'MuchFaster Early',
    {(T, T): perms[1][1][1]/tot_attempts, (T, F): perms[1][0][1]/tot_attempts, (F, T): perms[0][1][1]/tot_attempts, (F, F): perms[0][0][1]/tot_attempts}),
    ('Win', 'Overtake Crash', 
    {(T, T): perms[1][1][2]/tot_attempts, (T, F): perms[1][0][2]/tot_attempts, (F, T): perms[0][1][2]/tot_attempts, (F, F): perms[0][0][2]/tot_attempts})
    ])
    
    # END_YOUR_CODE ########################################################
    return bayes_net
    
def find_best_overtake_condition(bayes_net):
    """
    Finds the optimal condition for overtaking the car, as described in Part 3
    Returns the optimal values for (MuchFaster,Early)
    """
    # BEGIN_YOUR_CODE ######################################################
    # print(enumeration_ask('Overtake', dict(MuchFaster=T, Early=T), bayes_net).show_approx())
    # print(enumeration_ask('Overtake', dict(MuchFaster=T, Early=F), bayes_net).show_approx())
    # print(enumeration_ask('Overtake', dict(MuchFaster=F, Early=T), bayes_net).show_approx())
    # print(enumeration_ask('Overtake', dict(MuchFaster=F, Early=F), bayes_net).show_approx())
    # print(elimination_ask('Overtake', dict(MuchFaster=T, Early=T), bayes_net).show_approx())
    # print(elimination_ask('Overtake', dict(MuchFaster=T, Early=F), bayes_net).show_approx())
    # print(elimination_ask('Overtake', dict(MuchFaster=F, Early=T), bayes_net).show_approx())
    # print(elimination_ask('Overtake', dict(MuchFaster=F, Early=F), bayes_net).show_approx())
    T, F = True, False
    crash = [0.0]*4
    win = [0.0]*4
    crash[0] = elimination_ask('Crash', dict(MuchFaster=T, Early=T), bayes_net)
    crash[1] = elimination_ask('Crash', dict(MuchFaster=T, Early=F), bayes_net)
    crash[2] = elimination_ask('Crash', dict(MuchFaster=F, Early=T), bayes_net)
    crash[3] = elimination_ask('Crash', dict(MuchFaster=F, Early=F), bayes_net)

    win[0] = elimination_ask('Win', dict(MuchFaster=T, Early=T, Crash=F), bayes_net)
    win[1] = elimination_ask('Win', dict(MuchFaster=T, Early=F, Crash=F), bayes_net)
    win[2] = elimination_ask('Win', dict(MuchFaster=F, Early=T, Crash=F), bayes_net)
    win[3] = elimination_ask('Win', dict(MuchFaster=F, Early=F, Crash=F), bayes_net)
    
    max_p = 0
    hold = 0
    for i in range(4):
        prob = crash[i][F] * win[i][T]
        if prob > max_p:
            max_p = prob
            hold = i   
    #print(max_p)
    if hold==0:
        return ('T', 'T')
    elif hold==1:
        return ('T', 'F')
    elif hold==2:
        return ('F', 'T')
    else:
        return ('F', 'F')
    # END_YOUR_CODE ########################################################

def main():
    bayes_net = generate_bayesnet()
    cond = find_best_overtake_condition(bayes_net)
    print("Best overtaking condition: MuchFaster={}, Early={}".format(cond[0],cond[1]))

if __name__ == "__main__":
    main()
