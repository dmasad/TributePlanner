'''
Batch Runner
========================================

Tool for running several model instances in bulk, then outputting the
results to a file.

Currently some rules are hard-coded, primarily the small-world topology.
'''

import random
import json

import networkx as nx

from BaseModel import Model


class BatchRunner(object):

    def __init__(self, iter_count, step_count, G=None, **kwargs):
        '''
        Set up a new batch runner.

        Args:
            iter_count: Number of model instances to create and run
            step_count: How many steps to run each model for.
            G: If not None, initialize all models on the same graph. 
            **kwargs: Any model parameters to set.
        '''

        self.model_outputs = []
        self.models = []
        self.outputs = []
        self.step_count = step_count

        # Prepare models
        while len(self.models) < iter_count:
            if G is None:
                # Here comes the hard-coded bit
                G = nx.watts_strogatz_graph(10, 3, 0.2)
                if not nx.is_connected(G): continue
            
            m = Model(G)
            # Set the parameters:
            for attr, val in kwargs.items():
                if hasattr(m, attr):
                    setattr(m, attr, val)

            # Coerce minimum and maximum depth
            # TODO: Better solution for this
            for agent in m.agents.values():
                agent.max_depth = random.randint(m.min_depth, m.max_depth)

            self.models.append(m)

    def run_all(self):
        '''
        Run all the models one at a time, and store the results.
        '''

        for i, model in enumerate(self.models):
            print "Running", i
            for s in range(self.step_count):
                model.step()
            self.outputs.append(model.to_dict())

    



