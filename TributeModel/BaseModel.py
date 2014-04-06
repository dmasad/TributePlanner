'''
Simplified Tribute Model
========================================

Agents are atomic political actors, interacting across a fixed network.
Agents are endowed with initial Resources. When activated, agents select
a neighbor to threaten. The threatened neighbor then has to decide whether
to pay tribute, or go to war.

'''
from __future__ import division

import copy
import random
import networkx as nx


class Model(object):
    '''
    Baseline model object.
    '''

    def __init__(self, graph):
        '''
        Create a new Tribute Model with the given interaction graph.

        Args:
            graph: Interction graph, where each node will become an agent.
        '''

        self.N = len(graph)
        self.graph = graph
        # Instantiate the agents
        self.agents = {}
        for node in self.graph.nodes():
            a = Agent(node, self)
            self.graph.node[node]["Actor"] = a
            self.agents[node] = a

        # Model parameters:
        self.war_cost = 0.25 # Fraction of wealth inflicted as damage in war
        self.tribute = 250   # Tribute paid in wealth
        self.harvest = 20
        self.actions_per_turn = 3
        self.verbose = False


        # Model run statistics
        self.wars = 0
        
        # Introspective variables for recursive copying.
        self.depth = 0 # Current copy depth; 0 is the 'real' model.

        self.war_series = []
        self.agent_wealth_series = {i: [] for i in self.agents}

        # Visualization
        self.pos = nx.spring_layout(self.graph)


    def step(self):
        '''
        Run a single step of the model.
        '''
        self.wars = 0
        agents_to_act = set()
        while len(agents_to_act) < self.actions_per_turn:
            a = random.choice(self.agents.keys())
            agents_to_act.add(a)

        for agent_id in agents_to_act:
            agent = self.agents[agent_id]
            agent.choose_target()

        self.war_series.append(self.wars)


        # Harvest stage for each agent:
        for agent in self.agents.values():
            self.agent_wealth_series[agent.id_num].append(agent.wealth)
            agent.change_wealth(self.harvest)


    def war(self, attacker_id, defender_id):
        '''
        Execute a war between two agents.
        '''

        if self.verbose:
            print "War between", attacker_id, "and", defender_id

        self.wars += 1

        attacker = self.agents[attacker_id]
        defender = self.agents[defender_id]
        attacker_wealth = attacker.wealth
        defender_wealth = defender.wealth

        attacker.change_wealth(-self.war_cost * defender_wealth)
        defender.change_wealth(-self.war_cost * attacker_wealth)


class Agent(object):
    def __init__(self, id_num, model):
        '''
        Create a new agent.
        '''
        self.id_num = id_num
        self.model = model
        self.wealth = random.randint(300, 500) # Random initial wealth

    def change_wealth(self, delta_wealth):
        '''
        Adjust wealth, with a floor of 0.
        '''
        self.wealth += delta_wealth
        if self.wealth < 0:
            self.wealth = 0

    def choose_target(self):
        '''
        Pick a neighbor and make a threat.
        '''
        
        if self.wealth == 0:
            return

        neighbor_ids = self.model.graph.neighbors(self.id_num)
        vulnerability = {}
        for nid in neighbor_ids:
            neighbor = self.model.agents[nid]
            #vulnerability[nid] = self.evaluate_vulnerability(neighbor)
            vulnerability[nid] = self.evaluate_gain(neighbor)

        target_id = max(vulnerability, 
            key=lambda x: vulnerability[x])

        if vulnerability[target_id] > 0:
            target = self.model.agents[target_id]
            target.receive_threat(self.id_num)


    def evaluate_vulnerability(self, target):
        '''
        Evaluate a target's vulnerability using Axelrod's original heuristic.

            Vulnerability = (Own Wealth - Target Wealth) / Own Wealth

        Args:
            target: An Agent object to evaluate.
        '''

        target_wealth = target.wealth
        return (self.wealth - target_wealth) / self.wealth

    def evaluate_gain(self, target):
        '''
        Statically evaluate the results of war or tribute by creating a recursive
        copy of the model and testing the outcome.
        '''
        model_copy = copy.deepcopy(self.model)
        model_copy.depth += 1
        # Simulate the threat
        if self.model.verbose:
            print "- Begin internal simulation -"
        model_copy.agents[target.id_num].receive_threat(self.id_num)
        if self.model.verbose:
            print "- End internal simulation -"
        #new_wealth = model_copy.agents[self.id_num].wealth
        #gain = new_wealth - self.wealth
        gain = model_copy.agents[self.id_num].evaluate_position()
        return gain

    def evaluate_position(self):
        '''
        Evaluate hegemony position, defined as difference between own wealth
        and greatest neighbor wealth.
        '''

        neighbor_ids = self.model.graph.neighbors(self.id_num)
        neighbor_wealth = [self.model.agents[id_num].wealth 
                    for id_num in neighbor_ids]
        max_wealth = max(neighbor_wealth)
        return self.wealth - max_wealth




    def receive_threat(self, attacker_id):
        '''
        Decide whether to pay tribute, or go to war.
        Currently only short-term assessment
        This is where the bulk of the AI will go.
        '''

        attacker = self.model.agents[attacker_id]
        war_cost = attacker.wealth * self.model.war_cost
        if war_cost < self.model.tribute:
            self.model.war(attacker_id, self.id_num)
        else:
            tribute = min(self.model.tribute, self.wealth)
            attacker.change_wealth(tribute)
            self.change_wealth(-tribute)

            if self.model.verbose:
                print self.id_num, "paying tribute to", attacker_id

















