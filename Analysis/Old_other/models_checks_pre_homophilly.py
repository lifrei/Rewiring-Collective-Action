#To Check:
#link is broken only if link established
#check why average degree of bridge_if_same is declining 
# in bridge and biased rewiring agents establish links before breaking, random is opposite
#switch code so 

import numpy as np
import random
import sys 
sys.path.append("..")
import pandas as pd
from copy import deepcopy
from statistics import stdev, mean
import imageio
import networkx as nx
from networkx.algorithms.community import louvain_communities as community_louvain
from scipy.stats import truncnorm
import os
from operator import itemgetter
import heapq
from IPython.display import Image
import time
import matplotlib.patches as mpatches
import dill
import igraph as ig
import leidenalg as la
import math
import matplotlib.pyplot as plt
import seaborn as sns
from Auxillary.DPAH import DPAH
from collections import Counter
from fast_pagerank import pagerank_power
from joblib import delayed
from joblib import Parallel
from node2vec import Node2Vec

#random.seed(1574705741)    ## if you need to set a specific random seed put it here
#np.random.seed(1574705741)
#Helper functions

def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm(
            (low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

def setArgs(newArgs):
    global args
    for arg, value in newArgs.items():
        args[arg] = value

def getRandomExpo():
    x = np.random.exponential(scale=0.6667)-1
    if(x>1): return 1
    elif (x< -1): return -1
    return x

#Constants and Variables

## for explanation of these, see David's paper / thesis (Dynamics of collective action to conserve a large common-pool resource // Simple models for complex nonequilibrium Problems in nanoscale friction and network dynamics)
STATES = [1, -1] #1 being cooperating, -1 being defecting
defectorUtility = 0.0 # not used anymore
politicalClimate = 0.05 
newPoliticalClimate = 1*politicalClimate # we can change the political climate mid run
stubbornness = 0.6
degree = 8 
timesteps= 1000 #70000 
continuous = True
skew = -0.25
initSD = 0.15
mypalette = ["blue","red","green", "orange", "magenta","cyan","violet", "grey", "yellow"] # set the colot of plots
randomness = 0.10
gridtype = 'cl' # this is actually set in run.py for some reason... overrides this
gridsize = 33   # used for grid networks
nwsize = 40 #1089  # nwsize = 1089 used for CSF (Clustered scale free network) networks
friendship = 0.5
friendshipSD = 0.15
clustering = 0.5 # CSF clustering in louvain algorithm
#new variables:
breaklinkprob = 1 
rewiringMode = "None"
polarisingNode_f = 0
establishlinkprob = 1 # breaklinkprob and establishlinkprob are used in random rewiring. Are always chosen to be the same to keep average degree constant!
rewiringAlgorithm = 'None' #None, random, biased, bridge
#the rewiringAlgorithm variable was meant to enable to do multiple runs at once. However the loop where the specification 
#given in the ArgList in run.py file overrules what is given in line 65 does not work. Unclear why. 
#long story short. All changes to breaklinkprob, establishlinkprob and rewiringAlgorithm have to be specified here in the models file


#print(os.getcwd())
# the arguments provided in run.py overrides these values
args = {"defectorUtility" : defectorUtility, 
        "politicalClimate" : politicalClimate, 
        "stubbornness": stubbornness, "degree":degree, "timesteps" : timesteps, "continuous" : continuous, "type" : gridtype, "skew": skew, "initSD": initSD, "newPoliticalClimate": newPoliticalClimate, "randomness" : randomness, "friendship" : friendship, "friendshipSD" : friendshipSD, "clustering" : clustering,
        "rewiringAlgorithm" : rewiringAlgorithm,
        "rewiringMode": rewiringMode,
        "breaklinkprob" : breaklinkprob,
        "establishlinkprob" : establishlinkprob,
        "polarisingNode_f": polarisingNode_f}

#%% simulate 
def getargs():
    return args



def simulate(i, newArgs): #RG for random graph (used for testing)
    setArgs(newArgs)
    #global args

    # random number generators
    friendshipWeightGenerator = get_truncated_normal(args["friendship"], args["friendshipSD"], 0, 1) 
    initialStateGenerator = get_truncated_normal(args["skew"], args["initSD"], -1, 1)
    ind = None

    # network type to use, I always ran on a cl 
    if(args["type"] == "cl"):
        model =ClusteredPowerlawModel(nwsize, args["degree"], skew=args["skew"], friendshipWeightGenerator=friendshipWeightGenerator, initialStateGenerator=initialStateGenerator)
  
    elif(args["type"] == "sf"):
        model = ScaleFreeModel(nwsize, args["degree"], skew=args["skew"], friendshipWeightGenerator=friendshipWeightGenerator, initialStateGenerator=initialStateGenerator)
    elif(args["type"] == "grid"):
        ind = [10,64, 82]
        if(args["degree"]>2): doubleDegree = True
        else:doubleDegree = False
        model = GridModel(gridsize, skew=args["skew"], doubleDegree = doubleDegree, friendshipWeightGenerator=friendshipWeightGenerator, initialStateGenerator=initialStateGenerator)
    elif(args["type"] == "rand"):
        model = RandomModel(nwsize, args["degree"], skew=args["skew"], friendshipWeightGenerator=friendshipWeightGenerator, initialStateGenerator=initialStateGenerator)
    elif(args["type"] == "FB"):
        model = EmpiricalModel(f"../Pre_processing/networks_processed/{args['top_file']}", nwsize, args["degree"],  skew=args["skew"], friendshipWeightGenerator=friendshipWeightGenerator, initialStateGenerator=initialStateGenerator)
    elif(args["type"] == "Twitter"):
        model = EmpiricalModel(f"../Pre_processing/networks_processed/{args['top_file']}", nwsize, args["degree"],  skew=args["skew"], friendshipWeightGenerator=friendshipWeightGenerator, initialStateGenerator=initialStateGenerator)
    elif(args["type"] == "DPAH"):
        #args degre a.k.a 'm' is just passed here as a dummy variable but does not affect the DPAH model
        model = DPAHModel(nwsize, args["degree"], skew=args["skew"], friendshipWeightGenerator=friendshipWeightGenerator, initialStateGenerator=initialStateGenerator)
    else:
        model = RandomModel(nwsize, args["degree"],  friendshipWeightGenerator=friendshipWeightGenerator, initialStateGenerator=initialStateGenerator)

    #model.addInfluencers(newArgs["influencers"], index=ind, hub=False, allSame=False)
    #saving clustering and small world coefficient
    #C_0, S_0 = model.property_checks(R_G)
    #print(f"Start: Clustering, Small World Coefficient = {C_0, S_0}")
    res = model.runSim(args["timesteps"], clusters=True, drawModel=False, gifname= 'trialtrial') ## gifname provide a gif name if you want a gif animation, need to specify time stamps later on
    #C_end, S_end = model.property_checks(R_G)
    
    #storing difference in respective model variables
    # model.clustering_diff = C_end - C_0
    # model.small_world_diff = S_end - S_0
    # print(f"End: Clustering, Small World Coefficient = {C_end, S_end}")
    #draw_model(model)
    return model


#%% Agent class 
class Agent:
    def __init__(self, state, state2, stubbornness):
        self.state = state 
        self.state2 = state2 #in the beginning there was an idea of implementing a second state, however the results turned complicated so we stuck to just having one state opinion -> this is not used in the model
        self.interactionsReceived = 0
        self.interactionsGiven = 0
        self.stubbornness = stubbornness
        self.type = "converging"

    # this part is important, it defines the agent-agent interaction
    def consider(self, neighbour, neighboursWeight, politicalClimate):
        self.interactionsReceived +=1
        neighbour.addInteractionGiven()
        if(self.stubbornness >= 1): return
        
        mod = 1 
        if (self.type == "polarising") & (self.state*neighbour.state < 0):
            mod = -1 
     
        global args
        weight = self.state*self.stubbornness + politicalClimate + args["defectorUtility"] + neighboursWeight*neighbour.state 


        p1 = 0
        sample = random.uniform(-randomness,randomness)
        check = (weight + sample)

        if(check < -randomness): 
            p1 = 0

        elif(check > randomness): 
            p1 = 1

        else: 
            p1 = 1/(2*randomness)*(randomness + sample)

        p2 = 1 - p1
        delta = mod*(abs(self.state - neighbour.state)*(p1*(1-self.state) - p2*(1+self.state)))

        self.state += delta

        if(self.state > 1):
            self.state = STATES[0]
        elif(self.state <-1):
            self.state = STATES[1]       

    def addInteractionGiven(self):
        self.interactionsGiven +=1

    def setState(self, newState):
        if(newState >= STATES[1] and newState <= STATES[0]):
            self.state = newState
        else:
            print("Error state outside state range: ", newState)

#%% Model 

# this class contains functions and values partaining to the model, some of it is in use, some of it is not
class Model:
    # initial values
    def __init__(self, friendshipWeightGenerator = None, initialStateGenerator=None):
        global args
        self.graph = nx.Graph()
        self.politicalClimate = args["politicalClimate"]
        self.ratio = []
        self.states = []
        self.statesds = []
        self.degrees = []
        self.degreesSD = []
        self.mindegrees_l = []
        self.maxdegrees_l = []
        self.defectorDefectingNeighsList = []
        self.cooperatorDefectingNeighsList = []
        self.defectorDefectingNeighsSTDList = []
        self.cooperatorDefectingNeighsSTDList =[]
        self.pos = []
        self.friendshipWeightGenerator = friendshipWeightGenerator
        self.initialStateGenerator = initialStateGenerator
        self.clusteravg = []
        self.clusterSD = []
        self.NbAgreeingFriends = []
        self.avgNbAgreeingList = []
        self.partition = None
        self.test = []
        self.clustering_diff = []
        self.small_world_diff = []

    # picks a randon agent to perform an interaction with a random neighbour and then to rewire
    def interact(self, network_type = "undirected"):
        #print('starting interaction')
        nodeIndex = random.randint(0, len(self.graph) - 1)
        #print("in interact: ", nodeIndex)
        node = self.graph.nodes[nodeIndex]['agent']
     
            
        neighbours =  list(self.graph.adj[nodeIndex].keys())
        if(len(neighbours) == 0):
            return nodeIndex
        
        chosenNeighbourIndex = neighbours[random.randint(0, len(neighbours)-1)]
        chosenNeighbour = self.graph.nodes[chosenNeighbourIndex]['agent']
        weight = self.graph[nodeIndex][chosenNeighbourIndex]['weight']

        node.consider(chosenNeighbour, weight, self.politicalClimate)
        
        
        rewiringAlgorithm = args["rewiringAlgorithm"]
        self.algo = rewiringAlgorithm
        #the same random agent rewires after interacting:
        if rewiringAlgorithm != None: 
            if rewiringAlgorithm == 'random':
                self.randomrewiring(nodeIndex)
            elif rewiringAlgorithm == 'biased':
                self.biasedrewiring(nodeIndex)
            elif rewiringAlgorithm == 'bridge':
                self.bridgerewiring(nodeIndex)
            elif rewiringAlgorithm == "wtf":
                self.wtf(nodeIndex)
            elif rewiringAlgorithm == "node2vec":
                self.nodes2vec_learn(nodeIndex)
                self.nodes2vec_rewire(nodeIndex)
    
        
        #print('ending interaction')
        return nodeIndex

#%%% Rewiring algorithms
#rewiring--------------------------------------------------------------------------------
    
    def quick_sigma(self, test_graph, R_G, *args):
            
        def calc(graph):
            C, L = nx.average_clustering(graph), nx.average_shortest_path_length(graph)
            return C, L 
            
        out  = list(map(calc, [R_G, test_graph]))
        
        Cr, Lr, C, L = list(sum(out,()))
        
        sw_sigma = (C/Cr)/(L/Lr)
                
        return sw_sigma  
    
    
    def select_kmax_new(self, node_list):
        
        degree_sum = 0
        degree_sum = sum(self.graph.degree(i) for i in node_list)
        
        node = None 
        while node == None:
            for i in node_list:
                prob = self.graph.degree(i)/degree_sum
                
                if random.random() < prob:
                    node = i 
                    break
            
        return node 
    
    
    def check_node_state(self, node, neighbor):
        
        node_state = ""
         
        if (node.state >=0 and neighbor.state <0) or (node.state <0 and neighbor.state >=0):
                node_state = "different"
        elif (node.state >=0 and neighbor.state >=0) or (node.state <0 and neighbor.state <0):
                node_state = "same"
                
        return node_state
    
    def rewire(self, node_i, neighbour_i):
        
        weight = get_truncated_normal(args["friendship"], args["friendshipSD"], 0, 1).rvs(1)[0]
        
        self.graph.add_edge(node_i, neighbour_i, weight = weight)
        #self.TF_step(node_i, neighbour_i, weight = weight)
         
        return 

    #triad formation
    def TF_step(self, node, node_new, weight):
        
        adjacency = list(self.graph.adj[node_new].keys())
        
        if len(adjacency) > 1:
            #make sure node doesn't conect back to itself
            adjacency.remove(node)
            
            #print(adjacency)
            neighbour_node = random.sample(adjacency, 1)[0]
            
            #print(neighbour_node)
            self.graph.add_edge(node, neighbour_node, weight = weight)
         
            
        return 
    
        
        
    #this takes too long currently
    def property_checks(self, R_G): 
    
        
        #small world property "sigma"
        #sw_property = nx.sigma(self.graph)
        sw_property = self.quick_sigma(self.graph, R_G, args)
        #calculate avg clustering
        clustering = nx.average_clustering(self.graph)
        
        return clustering, sw_property
    
 
    def find_2_steps(self, nodeIndex):
        
        adjacency = list(self.graph.adj[nodeIndex].keys())
        total_neighbours = []
        
        for i in adjacency:
            adjacency_two = list(self.graph.adj[i].keys())
            total_neighbours+= adjacency_two
        
        #removing nodeIndex
        total_neighbours = [x for x in total_neighbours if x != nodeIndex]
        
        assert nodeIndex not in total_neighbours, "index in list"
        
        return total_neighbours 
    
    
    def randomrewiring(self, nodeIndex):
        
               
           # print('starting random rewiring')
            
            #reseting rewired value
            rewired_rand = False 
            
            establishlinkprob = args["establishlinkprob"]
            breaklinkprob = args["breaklinkprob"]
            self.probs  = establishlinkprob, breaklinkprob
                
            non_neighbors = []
            non_neighbors.append([k for k in nx.non_neighbors(self.graph, nodeIndex)])
            non_neighbors = non_neighbors[0]
           
            if(len(non_neighbors) == 0): #the agent is connected to the whole network, hence cannot establish further links
                return nodeIndex
            else:
                establishlinkNeighborIndex = random.sample(non_neighbors, 1)
                if random.random() < establishlinkprob:
                   # print('establishing link')
                    self.graph.add_edge(nodeIndex, establishlinkNeighborIndex[0], weight = get_truncated_normal(args["friendship"], args["friendshipSD"], 0, 1).rvs(1)[0])
                    # print(self.graph.edges[nodeIndex, establishlinkNeighborIndex[0]]['weight'])
                    rewired_rand = True 
                    
            init_neighbours =  list(self.graph.adj[nodeIndex].keys())
            
            if rewired_rand == True:
                
                 if(len(init_neighbours) == 0):
                    return nodeIndex
            
                 else:
                    breaklinkNeighbourIndex = init_neighbours[random.randint(0, len(init_neighbours)-1)]
           
                    if random.random() < breaklinkprob:
                     #  print('breaking link')
                       self.graph.remove_edge(nodeIndex, breaklinkNeighbourIndex)                                
                
            
            # print('ending random rewiring')
            
            
    def biasedrewiring(self, nodeIndex):
        #'realistic' rewiring. Agent can only rewire within small number of network steps and has a bias towards like minded people in the establishing of links
        #here we have an extreme assumption: if agents are of the same opinion a link is established for sure, if not, no link is established -> could be refined by introducing a probability curve 
        #e.g the more alike they are, the higher the probability of establishing a link, vice versa
        #breaking a link only when a link is established is a good idea as the likelyhood of meeting like thinkers changes as the network evolves and it is an easy way to retain the average degree to not get isolated nodes etc
      
        #print('starting biased rewiring')        
        
        potentialfriends = []
        non_neighbors = []
        non_neighbors.append([k for k in nx.non_neighbors(self.graph,nodeIndex)])
        non_neighbors = non_neighbors[0]
        #print("first neighbours: ", non_neighbors)
        
        #finding neighbors
        total_neighbours = self.find_2_steps(nodeIndex)
        #print(total_neighbours) 
        
        #removing duplicates
        res = [*set(total_neighbours)]
        
        #print(res)
        
        potentialfriends = set(non_neighbors).intersection(set(res))
        
        #print(potentialfriends)
        
    
            
            
        
        #replaced this with fastercode above
        # for k in non_neighbors: #! what if somebody has no neighbors!!
        #     #this is probably not the fastest way, maybe shorter way to check if there is a path with max length x
        #     paths = nx.all_simple_paths(self.graph, source=nodeIndex, target=k, cutoff=2)
        #     #cutoff defines the number of network steps to look for possible new friends. cutoff=2 it is possible for agent i to rewire to friends of friends
        #     if len(list(paths)) > 0:
        #         potentialfriends.append(k)
                   
        rewired = False #only if a link is established a link is broken hence we need a variable telling us whether a link has been established
        #print(rewired)
              
        if(len(potentialfriends) == 0): #the agent is connected to the whole network, hence cannot establish further links
           return nodeIndex
        else:
           establishlinkNeighborIndex = self.select_kmax_new(potentialfriends) #here preferential attachment should be implemented
           #print(establishlinkNeighborIndex)
               
           node = self.graph.nodes[nodeIndex]['agent']
           establishlinkNeighbor = self.graph.nodes[establishlinkNeighborIndex]['agent']
          
           node_state = self.check_node_state(node, establishlinkNeighbor)
           
           rewiring_mode = args["rewiringMode"]
           
           if node_state in "different" and rewiring_mode in "diff":
               self.rewire(nodeIndex, establishlinkNeighborIndex) 
               rewired = True
                     
           elif node_state in "same" and rewiring_mode in "same":
               self.rewire(nodeIndex, establishlinkNeighborIndex) 
               rewired = True
             
           else:
               return nodeIndex
               
                   
           if rewired == True: #links are only broken if a link is established
                
                init_neighbours =  list(self.graph.adj[nodeIndex].keys())
                if(len(init_neighbours) == 0):
                    return nodeIndex
            
                else:
                     #print('breaking a link')
                     breaklinkNeighbourIndex = init_neighbours[random.randint(0, len(init_neighbours)-1)]
                     self.graph.remove_edge(nodeIndex, breaklinkNeighbourIndex)


    def bridgerewiring(self, nodeIndex):
        #rewiring outside of ones own (opinion) cluster (we work with louvain clusters (=topological) but these become proxies for opinion clusters over time) -> time efficiency reason (see thesis)
        #we are looking at a deliberate algorithm that promotes faster dissolution of clusters (extreme case), non realistic
        #links are established for sure if agents disagree in their opinion, no link is established if they agree;
        #again this could be refined by introducing probability functions
        #links are only broken is a link is established (see above in biasedrewiring)
              
        if(self.partition == None): #louvain is already used in first step of runSim, this is just to set the variable equal to the partition in the function as well.
            partition = findClusters(self.graph, self.community_detection_with_leidenalg)
        else:
            partition = self.partition
            
        for k, v in partition.items():
            self.graph.nodes[k]["louvain-val"] = v

        
        nodeCluster =  self.graph.nodes[nodeIndex]['louvain-val'] #which cluster does agent i belong to?
        #print(nodeCluster)
        other_cluster = []
        other_cluster.append([k for k in self.graph.nodes if k != nodeIndex and self.graph.nodes[k]['louvain-val'] != nodeCluster]) #a list of all nodes that are not in agent i's cluster
        other_cluster = other_cluster[0]
        #print(other_cluster)
        rewired = False
        
        if(len(other_cluster) == 0): #the agent is connected to the whole network, hence cannot establish further links
            return nodeIndex
            #print('connected to everyone')
        else:
            #print('bridge rewiring')
            partnerOutClusterIndex = self.select_kmax_new(other_cluster) #here preferential attachment should be implemented
            #print(partnerOutClusterIndex)
            #print(self.graph.nodes[partnerOutClusterIndex]['louvain-val'])
                
            node = self.graph.nodes[nodeIndex]['agent']
            partnerOutCluster = self.graph.nodes[partnerOutClusterIndex]['agent']
            
            node_state = self.check_node_state(node, partnerOutCluster)
            
            rewiring_mode = args["rewiringMode"]
        
            
            if node_state in "different" and rewiring_mode in "diff":
                self.rewire(nodeIndex, partnerOutClusterIndex) 
                rewired = True
    
                      
            elif node_state in "same" and rewiring_mode in "same":
                self.rewire(nodeIndex, partnerOutClusterIndex) 
                rewired = True
    
              
            else:
                return nodeIndex
            
            if rewired == True:
                # if random.random() < breaklinkprob:
                    
                init_neighbours =  list(self.graph.adj[nodeIndex].keys())
                if(len(init_neighbours) == 0):
                    return nodeIndex
                 
                else:
                    breaklinkNeighbourIndex = init_neighbours[random.randint(0, len(init_neighbours)-1)]
                        
                    self.graph.remove_edge(nodeIndex, breaklinkNeighbourIndex)
    #------------------------------------------------------------------------------------        
            if "Agent" in self.graph:
                print("stop_3")
          
        return nodeIndex
    
    def train_node2vec(self):
        
        # Generate walks
        node2vec = Node2Vec(self.graph, dimensions=64, walk_length=30, num_walks=200, workers=4)
        
        # Train model
        model = node2vec.fit(window=10, min_count=1, batch_words=4)
        
    
        self.embeddings = {}
        
        for node in self.graph.nodes():
            self.embeddings[node] = model.wv[node]
            
    def node2vec_rewire(self, nodeIndex):
         
        
        def get_similar_agents(nodeIndex, embeddings):
            
            # Extract the target vector and all other vectors
            target_vec = self.embeddings[nodeIndex]
            all_agents = list(self.embeddingss.keys())
            all_vectors = np.array([self.embeddings[agent] for agent in all_agents])
            
            # Compute cosine similarity between the target vector and all vectors
            similarity = np.dot(all_vectors, target_vec) / (np.linalg.norm(all_vectors, axis=1) * np.nlinalg.norm(target_vec))
            
            # Create a list of (agent, similarity) tuples excluding the target agent
            similarities = [(agent, sim) for agent, sim in zip(all_agents, similarity) if agent != nodeIndex]
            
            # Sort the list by similarity in descending order
            similarities.sort(key=lambda x: x[1], reverse=True)
            return similarities
        
        
        self.rewire(nodeIndex, get_similar_agents(nodeIndex)[0])
        
        
             
            
    def wtf1(self, nodeIndex):
        TOPK = 10
        nodes = self.graph.nodes()
        A = nx.to_scipy_sparse_matrix(self.graph, nodes)
        
        
        def _ppr(node_index, A, p, top):
            pp = np.zeros(A.shape[0])
            pp[node_index] = A.shape[0]
            pr = pagerank_power(A, p=p, personalize=pp)
            pr = pr.argsort()[-top-1:][::-1]
            #time.sleep(0.01)
            return pr[pr!=node_index][:top]

        def get_circle_of_trust_per_node(A, p=0.85, top=10, num_cores=40):
            return Parallel(n_jobs=num_cores)(delayed(_ppr)(node_index, A, p, top) for node_index in np.arange(A.shape[0]))
        
        def frequency_by_circle_of_trust(A, cot_per_node=None, p=0.85, top=10, num_cores=40):
            results = cot_per_node if cot_per_node is not None else get_circle_of_trust_per_node(A, p, top, num_cores)
            unique_elements, counts_elements = np.unique(np.concatenate(results), return_counts=True)
            del(results)
            return [ 0 if node_index not in unique_elements else counts_elements[np.argwhere(unique_elements == node_index)[0, 0]] for node_index in np.arange(A.shape[0])]
        
        def _salsa(node_index, cot, A, top=10):
            BG = nx.Graph()
            BG.add_nodes_from(['h{}'.format(vi) for vi in cot], bipartite=0)  # hubs
            edges = [('h{}'.format(vi), int(vj)) for vi in cot for vj in np.argwhere(A[vi,:] != 0 )[:,1]]
            BG.add_nodes_from(set([e[1] for e in edges]), bipartite=1)  # authorities
            BG.add_edges_from(edges)
            centrality = Counter({n: c for n, c in nx.eigenvector_centrality_numpy(BG).items() if type(n) == int
                                                                                               and n not in cot
                                                                                               and n != node_index
                                                                                               and n not in np.argwhere(A[node_index,:] != 0 )[:,1] })
            del(BG)
            #time.sleep(0.01)
            return np.asarray([n for n, pev in centrality.most_common(top)])[:top]
        
        def frequency_by_who_to_follow(A, cot_per_node=None, p=0.85, top=10, num_cores=40):
            cot_per_node = cot_per_node if cot_per_node is not None else get_circle_of_trust_per_node(A, p, top, num_cores)
            results = Parallel(n_jobs=num_cores)(delayed(_salsa)(node_index, cot, A, top) for node_index, cot in enumerate(cot_per_node))
            unique_elements, counts_elements = np.unique(np.concatenate(results), return_counts=True)
            del(results)
            return [0 if node_index not in unique_elements else counts_elements[np.argwhere(unique_elements == node_index)[0, 0]] for node_index in np.arange(A.shape[0])]
        
        def who_to_follow_rank(A, njobs=1):
                return wtf_small(A, njobs)
                
        def wtf_small(A, njobs):
            cot_per_node = get_circle_of_trust_per_node(A, p=0.85, top=TOPK, num_cores=njobs)
        
                  
            wtf = frequency_by_who_to_follow(A, cot_per_node=cot_per_node, p=0.85, top=TOPK, num_cores=njobs)
            return wtf
        
        
        
        
        ranking = wtf_small(A, njobs = 4)        
        neighbour_index = ranking.index(max(ranking))
        self.rewire(nodeIndex, neighbour_index)
        

    
    
    
    
#%%% Statistics functions

    # the following two functions are for statistics
    def findNbAgreeingFriends(self, nodeIdx = None):
        global args
        nbs = []

        if(args["continuous"]):
            for nodeIdx in self.graph.nodes:
                state = self.graph.nodes[nodeIdx]['agent'].state
                neighbours = list(self.graph.adj[nodeIdx])
                neighStates = [self.graph.nodes[n]['agent'].state for n in neighbours ]
                if(len(neighbours) == 0):
                    nbs.append(0)
                    continue
                x = 1-abs((mean(neighStates)-state))/2
                nbs.append(x)
        else:
            for nodeIdx in self.graph.nodes:
                state = self.graph.nodes[nodeIdx]['agent'].state
                neighbours = list(self.graph.adj[nodeIdx])
                neighs = 0
                if(len(neighbours) == 0):
                    nbs.append(0)
                    continue
                for neighbourIdx in neighbours:
                    if(state == self.graph.nodes[neighbourIdx]['agent'].state): neighs+=1
                nbs.append(neighs/len(neighbours))
        self.NbAgreeingFriends= nbs
        return nbs

    def updateAvgNbAgreeingFriends(self, nodeIndex):
        #print(nodeIndex)
        neighbours =  list(self.graph.adj[nodeIndex].keys())
        if(len(neighbours) == 0):
            return self.avgNbAgreeingList[-1]
        nodeState = self.graph.nodes[nodeIndex]['agent'].state


        if(args["continuous"]):
            #TODO: check if this doesn't just repeat itself 
            neighStates = [self.graph.nodes[n]['agent'].state for n in neighbours ]
            x = 1-abs((mean(neighStates)-nodeState))/2
            self.NbAgreeingFriends[nodeIndex] = x
            for node in neighbours:
                nodeState = self.graph.nodes[node]['agent'].state
                neighneigh = list(self.graph.adj[node])
                neighStates = [self.graph.nodes[n]['agent'].state for n in neighneigh]
                print(node)
                x = 1-abs((mean(neighStates)-nodeState))/2
                self.NbAgreeingFriends[node] = x
        else:
            neighbours.append(nodeIndex)

            for n in neighbours:
                try:
                    neighneighs = list(self.graph.adj[n])
                    neighs = 0
                    nState = self.graph.nodes[n]['agent'].state
                    if(len(neighneighs) == 0):
                        self.NbAgreeingFriends[n] = (0)
                        continue
                    for neighbourIdx in neighneighs:
                        if(nState == self.graph.nodes[neighbourIdx]['agent'].state): neighs+=1
                    self.NbAgreeingFriends[n] = neighs/len(neighneighs)  
                except:
                    print("node: ", n)
                    print("neighs: ", neighneighs)      

        return mean(self.NbAgreeingFriends)

    # the influencer is now taken to replace the most extreme agent
    def addInfluencers(self, number = 0, index = None, hub = True, allSame =False):
        if(number == 0):
            return
        if(index == None):
            degrees = nx.degree(self.graph)
            if(hub):
                largest = heapq.nlargest(number, degrees, key=itemgetter(1))
                index = [t[0] for t in largest]


            else:
                index = [p[0]  for p in degrees if p[1] == degree*2]
                if(len(index) == 0 or len(index) < number ):
                    extra = [p[0]  for p in degrees if p[1] == degree*2-1]
                    index = index + extra
        for i in range(number):
            if(allSame):
                self.graph.nodes[index[i]]['agent'].setState(STATES[0])
            else:
                self.graph.nodes[index[i]]['agent'].setState(STATES[i % 2])
            self.graph.nodes[index[i]]['agent'].stubbornness = 1



    def countCooperatorRatio(self):
        count = 0
        
        #print(list(self.graph.nodes))
        for node in self.graph.nodes:
            #print(node)
            if self.graph.nodes[node]['agent'].state > 0:
                count+=1
        return count/len(self.graph)

    def getAvgState(self):
        states = []
        for node in self.graph:
            states.append(self.graph.nodes[node]['agent'].state)
        statearray = np.array(states)
        avg = statearray.mean(axis=0)
        sd = statearray.std()
        return (avg, sd)
    
    def getAvgDegree(self):
        degrees = [val for (node,val) in nx.degree(self.graph)]
        degreearray = np.array(degrees)
        mindegree = np.min(degreearray)
        maxdegree = np.max(degreearray)
        avgdegree = degreearray.mean(axis=0)
        sddegree = degreearray.std()
        return (avgdegree, sddegree, mindegree, maxdegree)
        
    
    def getFriendshipWeight(self):
        weigth = self.friendshipWeightGenerator.rvs(1)
        return weigth[0]

    def getInitialState(self):
        global args
        if(args['continuous'] != True): 
            state = STATES[random.randint(0,1)]
        else:   
            state = self.initialStateGenerator.rvs(1)[0]

        return state
    
    def getInitialState2(self):
        global args
    
        state2 = random.randint(0,1)
    
        return state2
    
    
    # this part actually runs the simulation
    def runSim(self, timesteps, groupInteract=False, drawModel = False, countNeighbours = False, gifname= 'trialtrial', clusters= False):
        if(self.partition ==None):
            if nx.is_directed(self.graph):
                
                
                self.partition = self.community_detection_with_leidenalg(self.graph)
                
            else:
                
                self.partition = self.community_detection_with_leidenalg(self.graph)

        if(drawModel):
            draw_model(self) #this would be supposed to draw the model before any interaction?
           

        filenames = []

        if(countNeighbours):
            (defectorDefectingNeighs,
                    cooperatorDefectingFriends,
                    defectorDefectingNeighsSTD,
                    cooperatorDefectingFriendsSTD) = self.getAvgNumberOfDefectorNeigh()
            print("Defectors: avg: ", defectorDefectingNeighs, " std: ", defectorDefectingNeighsSTD)
            print("Cooperators: avg: ", cooperatorDefectingFriends, " std: ", cooperatorDefectingFriendsSTD)

        #create list of number of agreeing friends
        # self.findNbAgreeingFriends()
        # self.avgNbAgreeingList.append(mean(self.NbAgreeingFriends))

        
        for i in range(timesteps):
            
            #print(list(self.graph))
            
            #if (i == 0) :
            #    reduce_grid(self)

            #print("step: ", i)
            nodeIndex = self.interact()
            ratio = self.countCooperatorRatio()
            self.ratio.append(ratio)
            (state, sd) = self.getAvgState()
            self.states.append(state)
            self.statesds.append(sd)
            (degree, degreeSD, mindegree, maxdegree) = self.getAvgDegree()
            self.degrees.append(degree)
            self.degreesSD.append(degreeSD)
            self.mindegrees_l.append(mindegree)
            self.maxdegrees_l.append(maxdegree)
            # avgFriends = self.updateAvgNbAgreeingFriends(nodeIndex)
            #avgFriends = self.findNbAgreeingFriends(nodeIndex)
            #draw_model(self) #this should draw the model in every timestep! 
            # self.avgNbAgreeingList.append(avgFriends)

            global args 
            # the commented code can be used to change the political climate mid simulation, and to add influencers as well
    
            #if(i == 4695 and (args["newPoliticalClimate"] != args["politicalClimate"])):
            #    self.politicalClimate = args["newPoliticalClimate"]
            #if(i == 7000):
            #    self.addInfluencers(1, index=None, hub=True, allSame=False)
            #self.politicalClimate += (ratio-0.5)*0.001 #change the political climate depending on the ratio of cooperators
           
           # some book keeping
            if(clusters):
                (s, sds, size) = findAvgStateInClusters(self, self.partition)
                self.clusterSD.append(sds)
                self.clusteravg.append(s)

            if(countNeighbours):
                (defectorDefectingNeighs,
                        cooperatorDefectingNeighs,
                        defectorDefectingNeighsSTD,
                        cooperatorDefectingNeighsSTD) = self.getAvgNumberOfDefectorNeigh()
                self.defectorDefectingNeighsList.append(defectorDefectingNeighs)
                self.cooperatorDefectingNeighsList.append(cooperatorDefectingNeighs)
                self.defectorDefectingNeighsSTDList.append(defectorDefectingNeighsSTD)
                self.cooperatorDefectingNeighsSTDList.append(cooperatorDefectingNeighsSTD)
            
            # snapshots = [0,50,100,250,500,750,1000,2000,4000]
            # if(gifname != None and (i in snapshots)):
            #     draw_model(self, True, i, extraTitle = f'  avg state: {self.states[-1]:1.2f} agreement: {self.avgNbAgreeingList[-1]:1.2f}')
            #     filenames.append("plot" + str(i) +".png")
            
        #if(gifname != None):
        #    images = []
        #    for filename in filenames:
        #        images.append(imageio.imread(filename))
        #    #0.08167
        #    imageio.mimsave("network" +gifname+ ".gif", images, duration=0.08167)

        (avgs, sds, sizes) = findAvgStateInClusters(self, self.partition)
        self.clusteravg.append(avgs)

        return self.ratio

    # initial generation of agents in network
    def populateModel(self, n, skew = 0):
        global args
        for n in range (n):
            
            agent1 = Agent(self.getInitialState(), random.choice([-1,1]), args["stubbornness"])
            self.graph.nodes[n]['agent'] = agent1
            
            if args["polarisingNode_f"] > np.random.random():
                self.graph.nodes[n]['agent'].type = "polarising"

        edges = self.graph.edges() 
        for e in edges: 
            weight=self.getFriendshipWeight()
            self.graph[e[0]][e[1]]['weight'] = weight

        if(skew != 0 and not args["continuous"] ): 
            num = round(abs(skew)*len(self.graph.nodes))
            indexes = random.sample(range(len(self.graph.nodes)), num)
            for i in indexes:
                self.graph.nodes[i]['agent'].state = STATES[1]
            #self.pos = nx.kamada_kawai_layout(self.graph)
        #self.pos = forceatlas2.forceatlas2_networkx_layout(self.graph)
        #self.pos = force_atlas2_layout(self.graph)
        #self.pos = nx.spring_layout(self.graph)

    def community_detection_with_leidenalg(self, nx_graph):
        # Convert networkx graph to igraph
        ig_graph = ig.Graph.from_networkx(nx_graph)
        
        # Perform community detection using leidenalg on the igraph graph
        partition = la.find_partition(ig_graph, la.ModularityVertexPartition)
        
        # Map the community detection results back to the networkx graph
        # Creating a dictionary where keys are nodes in the original networkx graph,
        # and values are their community labels
        nx_partition = {}
        for node in nx_graph.nodes():
            #ig_index = ig_graph.vs.find(name=str(node)).index  # Find the igraph index of the node
            nx_partition[node] = partition.membership[node]
        
        return nx_partition
#%% Network topologies 
# network types to use
class GridModel(Model):
    def __init__(self, n, skew=0, doubleDegree=False, **kwargs):
        super().__init__(**kwargs)
        global args
        for i in range(n):
            for j in range (n):
                agent1 = Agent(self.getInitialState(), args["stubbornness"])
                self.graph.add_node(i*n+j, agent=agent1, pos=(i, j))
                self.pos.append((i, j))
                if(i!=0):
                    weight = self.getFriendshipWeight()
                    self.graph.add_edge(i*n+j, (i-1)*n+j, weight = weight)
                if(i==n-1):
                    weight = self.getFriendshipWeight()
                    self.graph.add_edge(i*n+j, j, weight = weight)
                if(j!=0):
                    weight = self.getFriendshipWeight()
                    self.graph.add_edge(i*n+j, i*n+j-1, weight = weight)
                if(j==n-1):
                    weight = self.getFriendshipWeight()
                    self.graph.add_edge(i*n+j, i*n, weight = weight)
        if(doubleDegree): # this includes diagonal neighbours
            for i in range(n):
                for j in range(n):
                    if(i!=0 and j!=0 ):
                        weight = self.getFriendshipWeight()
                        self.graph.add_edge(i*n+j, (i-1)*n+j-1, weight = weight)
                    if(i!=0 and j!=(n-1)):
                        weight = self.getFriendshipWeight()
                        self.graph.add_edge(i*n+j, (i-1)*n+j+1, weight = weight)
                    """
                    if( i != n-1 and j!= n-1):
                        weight = self.getFriendshipWeight()
                        self.graph.add_edge(i*n+j, (i+1)*n+j+1, weight = weight)
                    if(j != 0 and i != n-i):
                        weight = self.getFriendshipWeight()
                        self.graph.add_edge(i*n+j, (i+1)*n+j-1, weight = weight)"""

                    if(j == n-1):
                        if(i == n-1):
                            weight = self.getFriendshipWeight()
                            self.graph.add_edge(i*n+j, 0, weight = weight)
                        else:
                            weight = self.getFriendshipWeight()
                            self.graph.add_edge(i*n+j, (i+1)*n, weight = weight)
                        if(i == 0):
                            weight = self.getFriendshipWeight()
                            self.graph.add_edge(i*n+j, (n-1)*n, weight = weight)
                        else:
                            weight = self.getFriendshipWeight()
                            self.graph.add_edge(i*n+j, (i-1)*n, weight = weight)
                    if( i == n-1):
                        if( j != n-1):
                            weight = self.getFriendshipWeight()
                            self.graph.add_edge(i*n+j, j+1, weight = weight)
                        if(j != 0):
                            weight = self.getFriendshipWeight()
                            self.graph.add_edge(i*n+j, j-1, weight = weight)
                        else: 
                            weight = self.getFriendshipWeight()
                            self.graph.add_edge(i*n+j, (n-1), weight = weight)
        if(skew != 0 and not args["continuous"] ): 
            num = round(abs(skew)*len(self.graph.nodes))
            indexes = random.sample(range(len(self.graph.nodes)), num)
            for i in indexes:
                self.graph.nodes[i]['agent'].state = STATES[1]

class EmpiricalModel(Model):
    def __init__(self,  filename, n, m, skew= 0, **kwargs):
        super().__init__(**kwargs)
        self.graph = nx.read_gpickle(filename)
       # np.savetxt("debug.txt", list(self.graph.nodes))
        self.populateModel(n, skew)

class DPAHModel(Model):
    def __init__(self, n, m, skew= 0, **kwargs):
        super().__init__(**kwargs)
        #TODO: make these not hard-coded 
        self.graph = DPAH(N = n,
             fm=0.5, 
             d=0.1, 
             plo_M=2.5, 
             plo_m=2.5, 
             h_MM=0.5, 
             h_mm=0.5, 
             verbose=False)
        self.populateModel(n, skew)

        
class ScaleFreeModel(Model):
    def __init__(self, n, m, skew= 0, **kwargs):
        super().__init__(**kwargs)
        self.graph = nx.barabasi_albert_graph(n, m)
        self.populateModel(n, skew)

class ClusteredPowerlawModel(Model):
    def __init__(self, n, m, skew = 0, **kwargs):
        super().__init__(**kwargs)

        self.graph = nx.powerlaw_cluster_graph(n, m, clustering)
        self.populateModel(n, skew)

class RandomModel(Model):
    def __init__(self, n, m, skew= 0, **kwargs):
        #m is avg degree/2
        super().__init__(**kwargs)
        p = 2*m/(n-1)

        self.graph =nx.erdos_renyi_graph(n, p)
        self.populateModel(n, skew)

# not exactly sure how this works tbh
def saveModels(models, filename):
    with open(filename, 'wb') as f:
        dill.dump(models, f)

def loadModels(filename):
    with open(filename, 'rb') as f:
        models = dill.load(f)
    return models

def findClusters(G, algo = None):
    
    if nx.is_directed(G):
        
        partition = algo(G)
        
    else:
        
        partition = algo(G)
        
    print(partition)
    return partition

    
def findAvgStateInClusters(model, part):
    states = [[] for i in range(len(set(part.values())))]
   
    for n, v in part.items():
        states[v].append(model.graph.nodes[n]['agent'].state)
    clusters = []
    sd = []
    clsize = []
    for c in range(len(states)):
        clusters.append(mean(states[c]))
        clsize.append(len(states[c]))
        if(len(states[c])>1):
            sd.append(stdev(states[c]))
        else:
            sd.append(0) 
    return (clusters, sd, clsize)

def findAvgSDinClusters(model, part):
    states = [[] for i in range(len(set(part.values())))]
    for n, v in part.items():
        states[v].append(model.graph.nodes[n]['agent'].state)
    
    sd = []
    for c in range(len(states)):
        if(len(states[c])>1):
            sd.append(stdev(states[c]))
        else:
            sd.append(0)
    return sd

# the rest of the file is saving images and data
def drawClusteredModel(model):
    if(model.partition==None):
        partition = findClusters(model)
    else:
        partition = model.partition

    for k, v in partition.items():
        model.graph.nodes[k]["louvain-val"] = v
    degrees = nx.degree(model.graph)

    #colors = [mypalette[G.nodes[node]["louvain-val"] %9 ]  for node in G.nodes()]
#    edge_col = [mypalette[model.graph.nodes[node]["louvain-val"]+1 % 8 ]  for node in model.graph.nodes()]
    edge_col = []
    for node in model.graph.nodes():
        edge_col.append(mypalette[model.graph.nodes[node]["louvain-val"] % 9 ])
    #plt.figure(figsize=(16,16))
    plt.subplot(1, 2, 2, title="Cluster Membership")
    nx.draw(model.graph, model.pos, node_size=[d[1] * 20 for d in degrees], node_color =edge_col)
    (clusters, sd, clsize) = findAvgStateInClusters(model, part= partition)
    text = [f'avg={clusters[c]:5.2f} sd={sd[c]:5.2f} n={clsize[c]}' for c in range(len(clusters))]
    #print(text)
    ax = plt.gca()
    handles = [mpatches.Patch(color=mypalette[c], label=text[c]) for c in range(len(text))]
    ax.legend(handles=handles)
    #plt.title("Snapshot of network with states and clusters")
    draw_model(model)#, outline=edge_col, partition = partition)
    
    
#%% Auxillary and data collection functions
  
#-------- save data functions ---------

def saveavgdata(models, filename, args = args):

    
    states = []
    sds = []
    for i in range(len(models)):
        states.append(models[i].states)
        sds.append(models[i].statesds)
    array = np.array(states)
    avg = array.mean(axis=0)
    std = np.array(sds).mean(axis=0)
    outs = np.column_stack((avg,std))
    hstring = 'avg.std'

    if(gridtype == 'cl'):
        avgSds = []
        for mod in models:
            array = np.array(mod.clusterSD)
            avgSd = array.mean(axis=1)
            avgSds.append(avgSd)
        array = np.array(avgSds)
        avgAvgSd = array.mean(axis=0)
        outs = np.column_stack((outs,avgAvgSd))
        hstring += ',clstd'
        
    degree = []
    degreeSD = []
    mindegree_l = []
    maxdegree_l = []


    num_models = len(models)
    for i in range(num_models):
        degree.append(models[i].degrees)
        degreeSD.append(models[i].degreesSD)
        mindegree_l.append(models[i].mindegrees_l)
        maxdegree_l.append(models[i].maxdegrees_l)
    
    rewiring_a = np.full((timesteps, 1), args["rewiringAlgorithm"])
    scenario_a = np.full((timesteps, 1), args["rewiringMode"])  # Adjust "rewiringMode" as needed
    type_a = np.full((timesteps, 1), args["type"])
    
    array = np.array(degree)
    mindegree_a = np.array(mindegree_l)
    maxdegree_a = np.array(maxdegree_l)
    avg_mindegree = mindegree_a.mean(axis=0)
    avg_maxdegree = maxdegree_a.mean(axis=0)
    avgdegree = array.mean(axis=0)
    degreeSD = np.array(degreeSD).mean(axis=0)
    
    #compiling arrays
    outs = np.column_stack((avg, std, avgdegree, degreeSD, avg_mindegree, avg_maxdegree, rewiring_a, scenario_a, type_a))
    #hstring += ',avgdegree.degreeSD.mindegree.maxdegree.scenario.rewiring.type'
   
    #np.savetxt(filename,outs,delimiter=',',header=hstring) 
    return(outs)

def savesubdata(models,filename):
    
    outs = []

    for i in range(len(models)):
        outs.append(models[i].states)
    
    outs = np.array(outs)
    np.savetxt(filename,outs,delimiter=',')

#-------- drawing functions ---------

def reduce_grid(model):
    n=gridsize
    for i in range(n):
        for j in range(n):
            if(i!=0 and j!=0 ):
                model.graph.remove_edge(i*n+j, (i-1)*n+j-1)
            if(i!=0 and j!=(n-1)):
                model.graph.remove_edge(i*n+j, (i-1)*n+j+1)
            """
            if( i != n-1 and j!= n-1):
                weight = model.getFriendshipWeight()
                model.graph.remove_edge(i*n+j, (i+1)*n+j+1, weight = weight)
            if(j != 0 and i != n-i):
                weight = model.getFriendshipWeight()
                model.graph.remove_edge(i*n+j, (i+1)*n+j-1, weight = weight)"""
            if(j == n-1):
                if(i == n-1):
                    model.graph.remove_edge(i*n+j, 0)
                else:
                    model.graph.remove_edge(i*n+j, (i+1)*n)
                if(i == 0):
                    model.graph.remove_edge(i*n+j, (n-1)*n)
                else:
                    model.graph.remove_edge(i*n+j, (i-1)*n)
            if( i == n-1):
                if( j != n-1):
                    model.graph.remove_edge(i*n+j, j+1)
                if(j != 0):
                    model.graph.remove_edge(i*n+j, j-1)
                else: 
                    model.graph.remove_edge(i*n+j, (n-1))

def draw_model(model, save=True, filenumber = 1, outline=None, partition=None, extraTitle="plotting"):
    
    #plt.figure(figsize=(4, 4))
    #plt.subplot(1, 2, 1, title="State of the Nodes")
    color_map = []
    intensities = []
    #pos = []
    for node in model.graph:
        #pos.append(model.graph.nodes[node]['pos'])
        if model.graph.nodes[node]['agent'].state > 0:
            color_map.append((3/255,164/255,94/255, model.graph.nodes[node]['agent'].state))
            intensities.append(model.graph.nodes[node]['agent'].state)
            #color_map.append('#03a45e')
            #else: color_map.append('#f7796d')
            
        else: 
            color_map.append((247/255,121/255,109/255, -1*model.graph.nodes[node]['agent'].state ))
            intensities.append(model.graph.nodes[node]['agent'].state)
    degrees = nx.degree(model.graph)
    #plt.subplot(121)i
    nx.draw(model.graph, model.pos, node_size=[d[1] * 30 for d in degrees], linewidths=2, node_color =intensities, cmap=plt.cm.RdYlGn,  vmin=-1, vmax=1)  # REMOVE EDGELIST for auto drawing edges
    #nx.draw(model.graph, model.pos, node_size=[d[1] * 30 for d in degrees], linewidths=2, node_color =intensities, cmap=plt.cm.RdYlGn,  vmin=-1, vmax=1, edgelist = [])  # REMOVE EDGELIST for auto drawing edges
    #sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn, norm=plt.Normalize(vmin=-1, vmax=1))
    #sm.set_array([])
    #cbar = plt.colorbar(sm)
    #plt.colorbar(mcp)
    #plt.show()
    
    if(outline !=None):
        #mypalette = ["blue","red","green", "yellow", "orange", "violet", "grey", "magenta","cyan", "cyan", "cyan", "cyan"]
        ax = plt.gca()
        ax.collections[0].set_edgecolor(outline)
        (clusters, sd, clsize) = findAvgStateInClusters(model, part= partition)
        text = [f'x={clusters[c]:5.2f} sd={sd[c]:5.2f} n={clsize[c]}' for c in range(len(clusters))]
        #print(text)
        handles = [mpatches.Patch(color=mypalette[c], label=text[c]) for c in range(len(text))]
        ax.legend(handles=handles)
        plt.title("Snapshot of network with states and clusters")


    if(save):
        plt.title(str(filenumber)+extraTitle)
        plt.savefig("plot" + str(filenumber) +".png", bbox_inches="tight")
        plt.close('all')
        

def drawAvgState(models, avg =False, pltNr=1, title="", clusterSD = False):
    plt.xlabel("timestep")
    plt.ylabel("AVG // STD")
    #mypalette = ["blue","red","green", "yellow", "orange", "violet", "grey", "grey","grey"]
    plt.subplot()
    #plt.subplot(1, 2, 1, title="Average State and SD")
    
    if(not avg):
        plt.ylim((-1, 1))
        for i in range(len(models)):
            plt.plot(models[i].states ,color='#ff7f0e')
            plt.plot(models[i].statesds, alpha=0.5 ,color='#ff7f0e')
            if(clusterSD):
                sds = np.array(models[i].clusterSD)
                avgsd = sds.mean(axis=1)
                plt.plot(avgsd, linestyle=":" ,color='#ff7f0e')
    else:
        states = []
        sds = []
        plt.ylim((-1, 1))
        for i in range(len(models)):
            states.append(models[i].states)
            sds.append(models[i].statesds)
        array = np.array(states)
        avg = array.mean(axis=0)
        std = np.array(sds).mean(axis=0)
        p1 = plt.plot(avg, color='#ff7f0e', label="AVG state")
        p2 = plt.plot(std, color='#ff7f0e', alpha=0.5, label="STD states")
        #plt.plot(avg+std, color=col.to_rgba(mypalette[pltNr-1], 0.5))
        #text = ["rand cont", "cl cont", "rand disc", "cl disc"]
        text =["Clustered"]
        handles = [mpatches.Patch(color=mypalette[c], label=text[c]) for c in range(len(text))]
        plt.legend(handles=handles)
        #print(models[0].states)
        if(clusterSD):
            avgSds = []
            for mod in models:
                array = np.array(mod.clusterSD)
                avgSd = array.mean(axis=1)
                avgSds.append(avgSd)
            array = np.array(avgSds)
            avgAvgSd = array.mean(axis=0)
            plt.plot(avgAvgSd, color='#ff7f0e', linestyle=":", label="STD in clusters")

        #plt.subplot(1, 2, 2)
        #plt.ylim((0, 1))
        #plt.plot(std, color=mypalette[pltNr-1])
        return (p1, p2)

def drawCrossSection(models, pltNr = 1):
    values = []
    #mypalette = ["blue","red","green", "yellow", "orange", "violet", "grey", "grey","grey"]
    for model in models:
        values.append(model.states[-1])
    plt.subplot(1, 2, 2, title="Density Plot of State for Simulations")
    ax = plt.gca()
    #ax.set_xscale('log')
    plt.xlim((0, 5))
    plt.ylim((-1, 1))
    #plt.title('Density Plot of state for simulations')
    #plt.xlabel('avg state of cooperators after all time steps')
    plt.xlabel('Density')
    #plt.scatter(range(len(values)), values.sort(), color = mypalette[pltNr-1])
    try:
        sns.distplot(values, hist=False, kde=True, color = mypalette[pltNr-1], vertical=True)
    except:
        sns.distplot(values, hist=True, kde=False, color = mypalette[pltNr-1], vertical=True)

    #plt.show()

def drawClustersizes(models, pltNr = 1):
    sizes = []
    for model in models:
        part = findClusters(model)
        (avg, sd, size) = findAvgStateInClusters(model, part)
        for s in size:
            sizes.append(s)
    plt.subplot(1, 3, 3, title="Density Plot of clustersize simulations")
    plt.xlabel("Clustersize")
    sns.distplot(sizes, hist=True, kde=True, color = mypalette[pltNr-1])

def drawConvergence(variables, modelsList, pltNr = 1):
    endState = []
    for models in modelsList:
        values = []
        for model in models:
            values.append(model.states[-1])
        endState.append(mean(values))
    plt.subplot(1,2,2)
    plt.xlim((-1, 1))
    plt.ylim((-1, 1))
    plt.scatter(variables, endState, color=mypalette[pltNr-1])

def drawClusterState(models, pltNr = 1, step=-1, subplot=1):
    plt.title("Density of Avg State in Communities")
    if(step < 0):
        plt.subplot(1, 3, 3, title="Avg State after Simulation")
        states = []
        for i in range(len(models)):
            for c in models[i].clusteravg[-1]:
                states.append(c)
    else:
        plt.subplot(1, 3, subplot, title="Avg State at t="+ str(step))
        states = []
        for i in range(len(models)):
            for c in models[i].clusteravg[step]:
                states.append(c)
    ax = plt.gca()
    #ax.set_xscale('log')
    plt.xlim((0, 5))
    plt.ylim((-1, 1))
    #plt.title('Density Plot of state for simulations')
    #plt.xlabel('avg state of cooperators after all time steps')
    plt.xlabel('Density')
    plt.ylabel('State')
    try:
        sns.distplot(states, hist=True, kde=True, color = mypalette[pltNr-1], vertical=True)
    except:
        sns.distplot(states, hist=True, kde=False, color = mypalette[pltNr-1], vertical=True)

def drawAvgNumberOfAgreeingFriends(models, pltNr = 1):
    avgNbAgreeingFriends = [model.avgNbAgreeingList for model in models]
    avgAvg = np.array(avgNbAgreeingFriends).mean(axis=0)
    plt.title("Average Agreement of Neighbours")
    plt.ylim((0, 1))
    plt.xlabel("Timesteps")
    plt.ylabel("Agreement")
    plt.plot(avgAvg, color=mypalette[pltNr-1])


#%%
#for testing only

def timer(func, arg):
    start = time.time()
    func(arg)
    end = time.time()
    mins = (end - start) / 60
    sec = (end - start) % 60
    print(f'Runtime was complete: {mins:5.0f} mins {sec}s\n')

    return

# # #for i in range(10):
# # args.update({"type": "DPAH"})
# model = simulate(50, args)

# G = nx.barabasi_albert_graph(100, 4)
# # nx.draw(model.graph)
# model.graph = G

# n = 20 

# out = model.wtf1(n)
# i = out.index(max(out))

# # labels=nx.draw_networkx_labels(model.graph,pos=nx.spring_layout(model.graph))


















