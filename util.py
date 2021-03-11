import networkx as nx
import numpy as np
import random
import torch
from sklearn.model_selection import StratifiedKFold
from argparse import Namespace
from typing import List, Tuple, Union
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
import json
# Atom feature sizes
MAX_ATOMIC_NUM = 685
ATOM_FEATURES = {
    'atomic_num': list(range(MAX_ATOMIC_NUM)),
    'degree': [0, 1, 2, 3, 4, 5],
    'formal_charge': [-1, -2, 1, 2, 0],
    'chiral_tag': [0, 1, 2, 3],
    'num_Hs': [0, 1, 2, 3, 4],
    'hybridization': [
        Chem.rdchem.HybridizationType.SP,
        Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3,
        Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2
    ],
}

# Distance feature sizes
PATH_DISTANCE_BINS = list(range(10))
THREE_D_DISTANCE_MAX = 20
THREE_D_DISTANCE_STEP = 1
THREE_D_DISTANCE_BINS = list(range(0, THREE_D_DISTANCE_MAX + 1, THREE_D_DISTANCE_STEP))

# len(choices) + 1 to include room for uncommon values; + 2 at end for IsAromatic and mass
ATOM_FDIM = sum(len(choices) + 1 for choices in ATOM_FEATURES.values()) + 2
BOND_FDIM = 14

# Memoization
SMILES_TO_GRAPH = {}
def onek_encoding_unk(value: int, choices: List[int]) -> List[int]:
    """
    Creates a one-hot encoding.

    :param value: The value for which the encoding should be one.
    :param choices: A list of possible values.
    :return: A one-hot encoding of the value in a list of length len(choices) + 1.
    If value is not in the list of choices, then the final element in the encoding is 1.
    """
    encoding = [0] * (len(choices) + 1)
    index = choices.index(value) if value in choices else -1
    encoding[index] = 1

    return encoding
def atom_features(atom: Chem.rdchem.Atom, functional_groups: List[int] = None) -> List[Union[bool, int, float]]:
    """
    Builds a feature vector for an atom.

    :param atom: An RDKit atom.
    :param functional_groups: A k-hot vector indicating the functional groups the atom belongs to.
    :return: A list containing the atom features.
    """
    features = onek_encoding_unk(atom.GetAtomicNum() - 1, ATOM_FEATURES['atomic_num']) + \
               onek_encoding_unk(atom.GetTotalDegree(), ATOM_FEATURES['degree']) + \
               onek_encoding_unk(atom.GetFormalCharge(), ATOM_FEATURES['formal_charge']) + \
               onek_encoding_unk(int(atom.GetChiralTag()), ATOM_FEATURES['chiral_tag']) + \
               onek_encoding_unk(int(atom.GetTotalNumHs()), ATOM_FEATURES['num_Hs']) + \
               onek_encoding_unk(int(atom.GetHybridization()), ATOM_FEATURES['hybridization']) + \
               [1 if atom.GetIsAromatic() else 0] + \
               [atom.GetMass() * 0.01]  # scaled to about the same range as other features
    if functional_groups is not None:
        features += functional_groups
    return features

class S2VGraph(object):
    def __init__(self, g, label, node_features=None):
        '''
            g: a networkx graph图神经网络
            label: an integer graph label整图的label
            node_tags: a list of integer node tags整节点的label的list
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
                            用于神经网络输入的tag的one-hot表示
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
                            edge的list，用于创建torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        '''
        self.label = label
        self.g = g
        self.neighbors = []
        self.node_features = 0
        self.edge_mat = 0

        self.max_neighbor = 0


def load_data(n, dataset):
    '''
            dataset: name of dataset
            test_proportion: ratio of test train split
            seed: random seed for random splitting of dataset
    '''

    G = []
    for p in range(500):
        df = pd.read_csv('dataset/dataset/%s/%s.csv'% (dataset, n), engine='python')
        g_list = []
        nodes = []

        g = nx.Graph()
        g.f_atoms = []
        g.smiles = df.loc[p][0]
        g.mol = Chem.MolFromSmiles(g.smiles)
        label = df.loc[p][1]
        g.atoms_n = g.mol.GetNumAtoms()
        for a1, atom in enumerate(g.mol.GetAtoms()):
            g.f_atoms.append(atom_features(atom))
            g.add_node(a1)
            for a2 in range(a1 + 1, g.atoms_n):
                g.bond = g.mol.GetBondBetweenAtoms(a1, a2)
                if g.bond is not None:
                    g.add_edge(a1, a2)
        g.f_atoms = [g.f_atoms[i] for i in range(g.atoms_n)]
        nodes.append(g.atoms_n)
        g_list.append(S2VGraph(g, label))


        for g in g_list:
            g.neighbors = [[] for i in range(len(g.g))]
            for i, j in g.g.edges():
                g.neighbors[i].append(j)
                g.neighbors[j].append(i)
            degree_list = []
            for i in range(len(g.g)):
                g.neighbors[i] = g.neighbors[i]
                degree_list.append(len(g.neighbors[i]))
            g.max_neighbor = max(degree_list)
            edges = [list(pair) for pair in g.g.edges()]
            edges.extend([[i, j] for j, i in edges])
            deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
            g.edge_mat = torch.LongTensor(edges).transpose(0, 1)


        for g in g_list:
            tagset = []
            for i in range(len(g.g.f_atoms[0])):
                tagset.append(i)
            tag2index = {tagset[i]: i for i in range(len(tagset))}
            g.node_features = torch.zeros(g.g.atoms_n, len(tagset))
            for j in range(g.g.atoms_n):
                g.node_features[j] = torch.LongTensor(g.g.f_atoms[j])

        G.append(g_list[0])

    return G


def valtest_load_data(setlist, shots, num_shots, dataset, retrain = False):
    '''
            dataset: name of dataset
            test_proportion: ratio of test train split
            seed: random seed for random splitting of dataset
    '''

    G = []
    for v, n in enumerate(setlist):
        df = pd.read_csv('dataset/dataset/%s/%s.csv' % (dataset, n), engine='python')
        p = shots[v]
        g_list = []
        #nodes = []
        for i in p:
            g = nx.Graph()
            g.f_atoms = []
            g.smiles = df.loc[i][0]
            g.mol = Chem.MolFromSmiles(g.smiles)
            label = df.loc[i][1]
            g.atoms_n = g.mol.GetNumAtoms()
            for a1, atom in enumerate(g.mol.GetAtoms()):
                g.f_atoms.append(atom_features(atom))
                g.add_node(a1)
                for a2 in range(a1 + 1, g.atoms_n):
                    g.bond = g.mol.GetBondBetweenAtoms(a1, a2)
                    if g.bond is not None:
                        g.add_edge(a1, a2)
            #g.f_atoms = [g.f_atoms[i] for i in range(g.atoms_n)]
            #assert len(g) == g.atoms_n
            #nodes.append(g.atoms_n)
            g_list.append(S2VGraph(g, label))

            #scaffold
            # if retrain:
            #     g1 = nx.Graph()
            #     g1.f_atoms = []
            #     g1.mol = MurckoScaffold.GetScaffoldForMol(g.mol)
            #     g1.atoms_n = g1.mol.GetNumAtoms()
            #     for a1, atom in enumerate(g1.mol.GetAtoms()):
            #         g1.f_atoms.append(atom_features(atom))
            #         g1.add_node(a1)
            #         for a2 in range(a1 + 1, g1.atoms_n):
            #             g1.bond = g1.mol.GetBondBetweenAtoms(a1, a2)
            #             if g1.bond is not None:
            #                 g1.add_edge(a1, a2)
			#
            # if list(g1.edges) == []:
            #     g1 = g
            # g_list.append(S2VGraph(g1, label))

            #sub_segregate
            if retrain:
                x = int(g.atoms_n / 2)
                g1 = nx.Graph()
                g2 = nx.Graph()
                g1.f_atoms = []
                g2.f_atoms = []
                g1.atoms_n = x
                g2.atoms_n = g.atoms_n - x
                for a1, atom in enumerate(g.mol.GetAtoms()):
                    if a1 < x:
                        g1.f_atoms.append(atom_features(atom))
                        g1.add_node(a1)
                        for a2 in range(a1 + 1, g1.atoms_n):
                            g1.bond = g.mol.GetBondBetweenAtoms(a1, a2)
                            if g1.bond is not None:
                                g1.add_edge(a1, a2)
                    else:
                        g2.f_atoms.append(atom_features(atom))
                        g2.add_node(a1-x)
                        for a2 in range(a1 + 1, g.atoms_n):
                            g2.bond = g.mol.GetBondBetweenAtoms(a1, a2)
                            if g2.bond is not None:
                                g2.add_edge(a1-x, a2-x)


                if list(g1.edges) == []:
                    g1 = g2
                if list(g2.edges) == []:
                    g2 = g1
                g_list.append(S2VGraph(g1, label))
                g_list.append(S2VGraph(g2, label))


        for g in g_list:

            edges = [list(pair) for pair in g.g.edges()]
            edges.extend([[i, j] for j, i in edges])
            deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
            g.edge_mat = torch.LongTensor(edges).transpose(0, 1)

        for g in g_list:
            tagset = []
            for i in range(len(g.g.f_atoms[0])):
                tagset.append(i)
            tag2index = {tagset[i]: i for i in range(len(tagset))}
            g.node_features = torch.zeros(g.g.atoms_n, len(tagset))
            for j in range(g.g.atoms_n):
                g.node_features[j] = torch.LongTensor(g.g.f_atoms[j])

        G.append(g_list)

    return G

def data_load_data(dataset, degree_as_tag):
	'''
		dataset: name of dataset
		test_proportion: ratio of test train split
		seed: random seed for random splitting of dataset
	'''

	print('loading data')
	g_list = []
	label_dict = {}
	feat_dict = {}

	with open('dataset/dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
		n_g = int(f.readline().strip())
		for i in range(n_g):
			row = f.readline().strip().split()
			n, l = [int(w) for w in row]
			if not l in label_dict:
				mapped = len(label_dict)
				label_dict[l] = mapped
			g = nx.Graph()
			node_tags = []
			node_features = []
			n_edges = 0
			for j in range(n):
				g.add_node(j)
				row = f.readline().strip().split()
				tmp = int(row[1]) + 2
				if tmp == len(row):
					# no node attributes
					row = [int(w) for w in row]
					attr = None
				else:
					row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
				if not row[0] in feat_dict:
					mapped = len(feat_dict)
					feat_dict[row[0]] = mapped
				node_tags.append(feat_dict[row[0]])

				if tmp > len(row):
					node_features.append(attr)

				n_edges += row[1]
				for k in range(2, len(row)):
					g.add_edge(j, row[k])

			if node_features != []:
				node_features = np.stack(node_features)
				node_feature_flag = True
			else:
				node_features = None
				node_feature_flag = False

			assert len(g) == n

			g_list.append(S2VGraph(g, l, node_tags))

	#add labels and edge_mat
	for g in g_list:
		g.neighbors = [[] for i in range(len(g.g))]
		for i, j in g.g.edges():
			g.neighbors[i].append(j)
			g.neighbors[j].append(i)
		degree_list = []
		for i in range(len(g.g)):
			g.neighbors[i] = g.neighbors[i]
			degree_list.append(len(g.neighbors[i]))
		g.max_neighbor = max(degree_list)

		# g.label = label_dict[g.label]

		edges = [list(pair) for pair in g.g.edges()]
		edges.extend([[i, j] for j, i in edges])

		deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
		g.edge_mat = torch.LongTensor(edges).transpose(0,1)

	if degree_as_tag:
		for g in g_list:
			g.node_tags = list(dict(g.g.degree).values())

	#Extracting unique tag labels
	tagset = set([])
	for g in g_list:
		tagset = tagset.union(set(g.node_tags))

	tagset = list(tagset)
	tag2index = {tagset[i]:i for i in range(len(tagset))}

	for g in g_list:
		g.node_features = torch.zeros(len(g.node_tags), len(tagset))
		g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


	print('# classes: %d' % len(label_dict))
	print('# maximum node tag: %d' % len(tagset))

	print("# data: %d" % len(g_list), "\n")

	return g_list, label_dict

def segregate(args, all_graphs, label_dict):
    all_classes = list(label_dict.keys())

    with open("dataset/dataset/{}/train_test_classes.json".format(args.dataset), "r") as f:
        all_class_splits = json.load(f)
        train_classes = all_class_splits["train"]
        test_classes = all_class_splits["test"]


    train_graph = [[] for _ in range(len(train_classes))]
    test_graph = [[] for _ in range(len(test_classes))]

    for i in range(len(all_graphs)):
        if all_graphs[i].label in train_classes:
            n = train_classes.index(all_graphs[i].label)
            train_graph[n].append(all_graphs[i])

        if all_graphs[i].label in test_classes:
            m = test_classes.index(all_graphs[i].label)
            test_graph[m].append(all_graphs[i])

    return train_graph, test_graph

def sub_segregate(args, all_graphs, label_dict):


    with open("dataset/dataset/{}/train_test_classes.json".format(args.dataset), "r") as f:
        all_class_splits = json.load(f)
        train_classes = all_class_splits["train"]
        test_classes = all_class_splits["test"]


    train_graph = [[] for _ in range(len(train_classes))]
    test_graph = [[] for _ in range(len(test_classes))]

    for i in range(len(all_graphs)):
        if all_graphs[i][0].label in train_classes:
            n = train_classes.index(all_graphs[i][0].label)
            train_graph[n].append(all_graphs[i])

        if all_graphs[i][0].label in test_classes:
            m = test_classes.index(all_graphs[i][0].label)
            test_graph[m].append(all_graphs[i])

    return train_graph, test_graph

def datasub_load_data(dataset, degree_as_tag):
	'''
		dataset: name of dataset
		test_proportion: ratio of test train split
		seed: random seed for random splitting of dataset
	'''

	print('loading data')
	g_list = []
	label_dict = {}
	feat_dict = {}

	with open('dataset/dataset/%s/%s.txt' % (dataset, dataset), 'r') as f:
		n_g = int(f.readline().strip())
		for i in range(n_g):
			G_list = []
			row = f.readline().strip().split()
			n, l = [int(w) for w in row]
			if not l in label_dict:
				mapped = len(label_dict)
				label_dict[l] = mapped
			g = nx.Graph()
			g1 = nx.Graph()
			g2 = nx.Graph()
			node_tags = []
			node_features = []
			n_edges = 0
			for j in range(n):
				g.add_node(j)
				if j < int(n/2):
					g1.add_node(j)
				else:
					g2.add_node(j - int(n/2))
				row = f.readline().strip().split()
				tmp = int(row[1]) + 2
				if tmp == len(row):
					# no node attributes
					row = [int(w) for w in row]
					attr = None
				else:
					row, attr = [int(w) for w in row[:tmp]], np.array([float(w) for w in row[tmp:]])
				if not row[0] in feat_dict:
					mapped = len(feat_dict)
					feat_dict[row[0]] = mapped
				node_tags.append(feat_dict[row[0]])

				if tmp > len(row):
					node_features.append(attr)

				n_edges += row[1]
				for k in range(2, len(row)):
					g.add_edge(j, row[k])
					if j < int(n/2) and row[k]< int(n/2):
						g1.add_edge(j, row[k])
					if j >= int(n/2) and row[k] >= int(n/2):
						g2.add_edge(j-int(n/2), row[k]-int(n/2))

			if node_features != []:
				node_features = np.stack(node_features)
				node_feature_flag = True
			else:
				node_features = None
				node_feature_flag = False

			assert len(g) == n

			if list(g1.edges) == []:
				g1 = g2
			if list(g2.edges) == []:
				g2 = g1

			G_list.append(S2VGraph(g, l, node_tags))
			G_list.append(S2VGraph(g1, l, node_tags))
			G_list.append(S2VGraph(g2, l, node_tags))

			g_list.append(G_list)

	#add labels and edge_mat
	for graph in g_list:
		for g in graph:
			g.neighbors = [[] for i in range(len(g.g))]
			for i, j in g.g.edges():
				g.neighbors[i].append(j)
				g.neighbors[j].append(i)
			degree_list = []
			for i in range(len(g.g)):
				g.neighbors[i] = g.neighbors[i]
				degree_list.append(len(g.neighbors[i]))
			g.max_neighbor = max(degree_list)

			# g.label = label_dict[g.label]

			edges = [list(pair) for pair in g.g.edges()]
			edges.extend([[i, j] for j, i in edges])

			deg_list = list(dict(g.g.degree(range(len(g.g)))).values())
			g.edge_mat = torch.LongTensor(edges).transpose(0,1)

	if degree_as_tag:
		for graph in g_list:
			for g in graph:
				g.node_tags = list(dict(g.g.degree).values())

	#Extracting unique tag labels
	tagset = set([])
	for graph in g_list:
		for g in graph:
			tagset = tagset.union(set(g.node_tags))

	tagset = list(tagset)
	tag2index = {tagset[i]:i for i in range(len(tagset))}

	for graph in g_list:
		for g in graph:
			g.node_features = torch.zeros(len(g.node_tags), len(tagset))
			g.node_features[range(len(g.node_tags)), [tag2index[tag] for tag in g.node_tags]] = 1


	print('# classes: %d' % len(label_dict))
	print('# maximum node tag: %d' % len(tagset))

	print("# data: %d" % len(g_list), "\n")

	return g_list, label_dict


