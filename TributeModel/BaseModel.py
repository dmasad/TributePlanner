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
from itertools import cycle

import networkx as nx
import matplotlib.pyplot as plt

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

        # Model parameters:
        self.war_cost = 0.25 # Fraction of wealth inflicted as damage in war
        self.tribute = 250   # Tribute paid in wealth
        self.harvest = 20 # Resources gained every turn
        self.actions_per_turn = 3 # Agents activated per turn
        self.verbose = False # Whether to print verbose outputs
        self.min_depth = 2 # Minimum agent depth
        self.max_depth = 4 # Maximum agent depth

        self.heuristic = False
        self.random_execution = False

        # Set up the graph
        self.N = len(graph)
        self.graph = graph
        # Instantiate the agents
        self.agents = {}
        for node in self.graph.nodes():
            a = Agent(node, self)
            self.graph.node[node]["Actor"] = a
            self.agents[node] = a


        self.step_counter = -1 

        # Shuffle execution order once, for the deterministic method:
        shuffled_agents = self.agents.keys()
        random.shuffle(shuffled_agents)
        self.run_order = shuffled_agents



        # Model run statistics
        self.wars = 0
        
        # Introspective variables for recursive copying.
        self.depth = 0 # Current copy depth; 0 is the 'real' model.

        self.war_series = [] # Count the number of wars fought
        self.war_scale = [] # Track the relative damage done in each war.
        self.agent_wealth_series = {i: [] for i in self.agents}

        # Visualization
        self.pos = nx.spring_layout(self.graph)


    def step(self):
        '''
        Run a single step of the model.
        '''

        if self.random_execution:
            self.random_step()
        else:
            self.deterministic_step()

    def random_step(self):
        '''
        Run a step involving several randomly-selected agents
        '''
        self.wars = 0
        self.step_counter += 1
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



    def deterministic_step(self):
        '''
        Run a single step of the model using the deterministic step, with fixed
        execution order.
        '''
        self.step_counter += 1
        #next_agent = self.run_order.next()
        next_agent = self.run_order[self.step_counter % self.N]
        agent = self.agents[next_agent]
        agent.choose_target()

        # End of turn, i.e. set number of steps:
        #self.step_counter += 1
        if self.step_counter % self.actions_per_turn == 0:
            self.war_series.append(self.wars)
            self.wars = 0
            for agent in self.agents.values():
                self.agent_wealth_series[agent.id_num].append(agent.wealth)
                agent.change_wealth(self.harvest)



    def war(self, attacker_id, defender_id):
        '''
        Execute a war between two agents.
        '''

        if self.verbose:
            print "\t"*self.depth, "War between", attacker_id, "and", defender_id

        self.wars += 1 # Increment the counter of wars fought this turn

        attacker = self.agents[attacker_id]
        defender = self.agents[defender_id]
        attacker_wealth = attacker.wealth
        defender_wealth = defender.wealth

        attacker.change_wealth(-self.war_cost * defender_wealth)
        defender.change_wealth(-self.war_cost * attacker_wealth)

        # Record the war damage
        if attacker_wealth > 0:
            attacker_damage = (self.war_cost * defender_wealth) / attacker_wealth
        else:
            attacker_damage = 1
        if defender_wealth > 0:
            defender_damage = (self.war_cost * attacker_wealth) / defender_wealth
        else:
            defender_damage = 1


        self.war_scale += [attacker_damage, defender_damage]


    def copy(self):
        '''
        Make a deep copy of this object and return it, incrementing the depth.
        '''
        clone = copy.deepcopy(self)
        clone.depth += 1
        if self.verbose:
            print "\t"*self.depth, "Spawning new model at depth", clone.depth
        return clone

    def plot(self, fig=None):
        '''
        '''
        if fig is None:
            fig = plt.figure(figsize=(12,4))
        ax1 = fig.add_subplot(121)
        ax1.bar(range(len(self.war_series)), self.war_series)

        ax2 = fig.add_subplot(122)
        for s in self.agent_wealth_series.values():
            ax2.plot(s)
        return fig

    def to_dict(self):
        '''
        Output a representation of the model to a dictionary
        '''

        params = {
            "N": self.N,
            "war_cost": self.war_cost,
            "tribute": self.tribute,
            "harvest": self.harvest,
            "actions_per_turn": self.actions_per_turn,
            "random_execution": self.random_execution,
            "min_depth": self.min_depth,
            "max_depth": self.max_depth,
            "heuristic": self.heuristic
        }

        graph = nx.to_dict_of_dicts(self.graph)

        agent_data = {
            i: 
                {
                    "starting_wealth": agent.starting_wealth,
                    "wealth": agent.wealth,
                    "max_depth": agent.max_depth
                }
                for i, agent in self.agents.items() }

        output = {
            "step": self.step_counter,
            "parameters": params,
            "graph": graph,
            "agents": agent_data,
            "war_series": self.war_series,
            "war_scale": self.war_scale,
            "agent_wealth_series": self.agent_wealth_series

        }

        return output






class Agent(object):
    '''
    An agent represents an atomic political actor in the model. 
    This version of the agent is endowed with lookahead, and can make decisions 
    to maximize wealth / power relative to its immediate neighbors on the 
    network. 
    '''
    def __init__(self, id_num, model):
        '''
        Create a new agent.
        '''
        self.id_num = id_num
        self.model = model
        self.wealth = random.randint(300, 500) # Random initial wealth
        self.max_depth = random.randint(model.min_depth, model.max_depth)
        self.starting_wealth = self.wealth

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

        if self.model.verbose:
            print "\t"*self.model.depth, self.id_num, "active"
        neighbor_ids = self.model.graph.neighbors(self.id_num)
        vulnerability = {}
        for nid in neighbor_ids:
            neighbor = self.model.agents[nid]
            if self.model.heuristic:
                vulnerability[nid] = self.evaluate_vulnerability(neighbor)
            else:
                vulnerability[nid] = self.evaluate_gain(neighbor)

        target_id = max(vulnerability, 
            key=lambda x: vulnerability[x])

        if vulnerability[target_id] > self.evaluate_position():
            if self.model.verbose:
                print "\t"*self.model.depth, self.id_num, "threatening", target_id
            target = self.model.agents[target_id]
            target.receive_threat(self.id_num)
        else:
            if self.model.verbose:
                print "\t"*self.model.depth, self.id_num, "taking no action."



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
        model_copy = self.copy_model()
        # Simulate the threat
        #if self.model.verbose:
        #    print "- Begin internal simulation -"
        
        model_copy.agents[target.id_num].receive_threat(self.id_num)
        gain = self.look_ahead(model_copy)

        #if self.model.verbose:
        #    print "- End internal simulation -"
        #new_wealth = model_copy.agents[self.id_num].wealth
        #gain = new_wealth - self.wealth
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
        Use one-action lookahead static evaluation.
        '''
        # Simulate tribute scenario
        tribute_scenario = self.copy_model()
        tribute = min(self.model.tribute, self.wealth)
        tribute_scenario.agents[self.id_num].change_wealth(-tribute)
        tribute_scenario.agents[attacker_id].change_wealth(tribute)
        #tribute_score = tribute_scenario.agents[self.id_num].evaluate_position()
        tribute_score = self.look_ahead(tribute_scenario)

        # Simulate war scenario
        war_scenario = self.copy_model()
        war_scenario.war(attacker_id, self.id_num)
        war_score = self.look_ahead(war_scenario)
        #war_score = war_scenario.agents[self.id_num].evaluate_position()        

        if self.model.verbose:
            print "\t"*self.model.depth, self.id_num, tribute_score, war_score

        if war_score > tribute_score:
            self.model.war(attacker_id, self.id_num)
            return "war"
        else:
            #tribute = min(self.model.tribute, self.wealth)
            if self.model.verbose:
                print "\t"*self.model.depth, self.id_num, "paying tribute to", attacker_id
            self.model.agents[attacker_id].change_wealth(tribute)
            self.change_wealth(-tribute)
            return "tribute"

    def copy_model(self):
        '''
        Create an internal copy of the model, and then modify it so that no
        agent has greater max_depth than the originating agent.
        '''
        copy = self.model.copy()
        for agent in copy.agents.values():
            agent.max_depth = min(agent.max_depth, self.max_depth)
        return copy


    def look_ahead(self, model):
        '''
        Evaluate a model copy forward
        '''
        steps = self.max_depth - model.depth
        for i in range(steps):
            model.depth += 1 # Temporary?
            model.step()

        return model.agents[self.id_num].evaluate_position()






















