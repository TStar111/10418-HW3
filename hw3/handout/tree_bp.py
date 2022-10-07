import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from collections import defaultdict
import copy
import numpy as np
import nltk.tree

EMBEDDING_DIM = 256
HIDDEN_DIM = 128

tag_dict = defaultdict(lambda: len(tag_dict) )

class OverallIndex():
  def __init__(self,val):
    self.val = val

'''
Class that represents individual nodes
'''
class Node():
  def __init__(self, treeNode, parent, idx, idx_set):
    if isinstance(treeNode, tuple):
      self.true_label = np.array([tag_dict[treeNode[1]]])
      self.word = treeNode[0]
    else:
      self.true_label = np.array([tag_dict[treeNode._label]])

    self.true_label = torch.tensor(self.true_label).long()

    self.children = []
    self.unary_potential = self.belief = 1
    self.parent = parent
    self.idx = idx  # give each node in the tree a unique index
    idx_set.add(idx)

    for child in treeNode:
      if not isinstance(child, str):
        self.children.append(Node(child, self, max(idx_set) + 1, idx_set))

'''
Builds a tree using the node class above
Outputs a Node object along with the size of the tree
'''
def build_tree(tree):
  idx_set = set([])
  my_tree = Node(tree, None, 0, idx_set)
  return my_tree, len(idx_set)

'''
Helper function that returns the appropriate leaves in node
'''
def get_leaves(node, leaves):
    if len(node.children) == 0:
        leaves += [node.idx]
    for child in node.children:
        leaves = get_leaves(child, leaves)
    return leaves

'''
Class representing dataset structure
Important: Please read through and understand what this code is doing in detail,
           this will make implementation much easier for you!
'''
class Dataset():
  def __init__(self, path_to_file, w2i=None):
    self.trees = []
    with open(path_to_file) as f:
      lines = [l.strip() for l in f.readlines()]

    for line in lines:
      try:
        self.trees.append(nltk.tree.Tree.fromstring(line))
      except:
        continue

    self.sentences = list([tree.leaves() for tree in self.trees])
    self.tree_size = list([len(tree.treepositions()) for tree in self.trees])
    self.len = len(self.trees)

    self.tree_tups = list([build_tree(tree) for tree in self.trees])
    self.my_trees = list([t[0] for t in self.tree_tups])
    self.tree_lens = list([t[1] for t in self.tree_tups])

    self.w2i = w2i
    self.sentences = self.build_vocab()
    if w2i is None:  # initialized vocab for the first time
      self.w2i['<UNK>'] = 0
      self.w2i.default_factory = lambda: 0  # all future unknown tokens will be mapped to this index

    self.vocab_size = max(list(self.w2i.values())) + 1
    self.tag2idx = tag_dict

    self.tag_size = len(self.tag2idx)
    self.batch_size = 1

    self.ptr = 0
    self.reverse_dict = {v: k for k, v in self.tag2idx.items()}

  def reset(self):
    self.ptr = 0

  def build_vocab(self):
    if self.w2i is None:
            self.w2i = defaultdict(lambda: len(self.w2i) + 1)
    sentence_idxs = [[self.w2i[x.lower()] for x in sent] for sent in self.sentences]
    return sentence_idxs

  def get_next_batch(self):
    current_batch = (torch.LongTensor(np.array(self.sentences[self.ptr])),
                     self.tree_size[self.ptr],
                     self.my_trees[self.ptr],
                     self.trees[self.ptr],
                     self.tree_lens[self.ptr])
    self.ptr += 1
    return self.ptr == self.len, current_batch


'''
Modularized Tree class that will be used to represent both LSTM + CRF Tree
'''
class TreeModule(nn.Module):

  def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags, model):
    super(TreeModule, self).__init__()
    self.model = model
    self.embedding = nn.Embedding(vocab_size, embedding_dim)
    self.lstm = nn.LSTM(embedding_dim, hidden_dim)
    self.build_unary_potentials = nn.Linear(hidden_dim, num_tags)
    self.num_tags = num_tags
    if model == 'baseline':
      self.build_intermediate_hidden = nn.Linear(2 * hidden_dim, hidden_dim)
      self.build_unary_potentials_intermediate_node = nn.Linear(2 * hidden_dim, num_tags)
      self.criterion = nn.NLLLoss()
      self.logsoftmax = nn.LogSoftmax(dim=-1)
    else:
      self.build_edge_potentials = nn.Linear(2 * hidden_dim, num_tags * num_tags)
      self.hidden_linear = nn.Linear(2 * hidden_dim, hidden_dim)
      self.build_unary_potentials_intermediate_node = nn.Linear(2 * hidden_dim, num_tags)
      self.criterion = TreeCrfLoss()
    
  def compute_potentials(self, tree_structure, overall_index, leaves):
    
    if len(tree_structure.children) == 0:
      tree_structure.unary_potential = self.unary_pot[overall_index.val]
      tree_structure.lstm = self.lstm_op[overall_index.val]
      tree_structure.belief = tree_structure.unary_potential

      leaves.append(tree_structure)
      overall_index.val+=1
      return 

    for child in tree_structure.children:
      self.compute_potentials(child, overall_index, leaves)
 
    if self.model == 'baseline':
      children_cat = torch.cat((tree_structure.children[0].lstm, tree_structure.children[-1].lstm), dim=-1)
      tree_structure.lstm = self.build_intermediate_hidden(children_cat)
      tree_structure.unary_potential = self.build_unary_potentials_intermediate_node(children_cat)
    else:
      lstm_left = tree_structure.children[0].lstm
      lstm_right = tree_structure.children[-1].lstm
      tree_structure.unary_potential = (self.build_unary_potentials_intermediate_node(
          torch.cat((lstm_left, lstm_right), dim=-1)))
      tree_structure.lstm = self.hidden_linear(torch.cat((lstm_left, lstm_right), dim=-1))

    tree_structure.belief = tree_structure.unary_potential

  ####################################################################
  ## Helper functions for LSTM + CRF
  ####################################################################
  def compute_edge_potentials(self, tree_structure):
    for child in tree_structure.children:
      child.edge_potential = self.build_edge_potentials(
          torch.cat((child.lstm, tree_structure.lstm), dim=-1)).view(-1, self.num_tags)
      self.compute_edge_potentials(child)

  @classmethod
  def _get_true_labels(cls, node, d):
    d[node.idx] = node.true_label
    for child in node.children:
        d = cls._get_true_labels(child, d=d)
    return d

  @classmethod
  def _get_parents_dict(cls, node, parent_idx, d):
    d[node.idx] = parent_idx
    for child in node.children:
        d = cls._get_parents_dict(child, node.idx, d=d)
    return d

  @classmethod
  def _get_children_dict(cls, node, d):
    for child in node.children:
        d[node.idx] = d.get(node.idx, []) + [child.idx]
        d = cls._get_children_dict(child, d=d)
    return d

  @classmethod
  def _get_unary_potential_dict(cls, node, d={}):
    d[node.idx] = node.unary_potential
    for child in node.children:
        d = cls._get_unary_potential_dict(child, d=d)
    return d

  @classmethod
  def _get_edge_potential_dict(cls, node, d={}):
    # edge potential between node and its parent
    for child in node.children:
        d[child.idx] = child.edge_potential
        d = cls._get_edge_potential_dict(child, d=d)
    return d

  @classmethod
  def _get_tensor_from_dict(cls, d, tree_len):
    tensors = []
    example = None
    for i in range(tree_len):  # order the tensors
        if i in d:
            tensors.append(d[i].unsqueeze(0))
            example = tensors[-1]
        else:
            tensors.append(None)
    tensors = [t if t is not None else torch.ones_like(example) for t in tensors]
    tensor = torch.cat(tensors, axis=0)
    return tensor

  @classmethod
  def get_structure_dicts(cls, my_structure):
    parents_dict = cls._get_parents_dict(my_structure, None, {})
    del parents_dict[0]

    children_dict = cls._get_children_dict(my_structure, {})
    true_labels = cls._get_true_labels(my_structure, {})
    leaves_idx = get_leaves(my_structure, [])
    leaves_idx = list(set(leaves_idx))
    return parents_dict, children_dict, leaves_idx, true_labels

  ####################################################################
  ## End of helper functions for LSTM + CRF
  ####################################################################

  # forward function for LSTM + CRF
  # Output:
  #   For model=='baseline':
  #     Outputs a list of size three that contains the following elements
  #         in the exact order [average loss, average accuracy, average leaf accuracy]
  #   For model=='crf:
  #     Outputs a list of size two that contains the following elements
  #      in the exact order [unary potentials, edge potentials] where 
  #      both elements are tensors

  def forward(self, inp, tree_len, my_structure):
    lstm_inp = self.embedding(inp).unsqueeze(0)
    self.lstm_op, _ = self.lstm(lstm_inp)

    #^ 1 x L x D
    self.lstm_op = self.lstm_op.squeeze(0)
    self.unary_pot = self.build_unary_potentials(self.lstm_op)
    
    leaves =[]
    overall_index =  OverallIndex(0)
    self.compute_potentials(my_structure,overall_index, leaves)

    if self.model == 'baseline':
      loss, accuracy, leaf_acc = self.TreeLSTMLoss(my_structure, leaves)
  
      return [loss/tree_len, accuracy/tree_len, leaf_acc/len(inp)]
    else:
      self.compute_edge_potentials(my_structure)
      my_structure.edge_potential = torch.zeros(self.num_tags * self.num_tags).view(-1, self.num_tags)

      unary_potentials = TreeModule._get_tensor_from_dict(TreeModule._get_unary_potential_dict(my_structure), tree_len)
      edge_potentials = TreeModule._get_tensor_from_dict(TreeModule._get_edge_potential_dict(my_structure), tree_len)

      return [unary_potentials, edge_potentials]

  # Calculates metrics for Tree LSTM
  def TreeLSTMLoss(self, node, leaves):
    ce_loss = torch.nn.CrossEntropyLoss()
    loss = ce_loss((node.belief - torch.min(node.belief)).unsqueeze(0), node.true_label)

    pred = torch.argmax(node.belief).item()
    acc = (pred == node.true_label.item())

    leaf_acc = 0
    if node in leaves:
      leaf_acc = (pred == node.true_label.item())

    for child in node.children:
      child_loss, child_acc, child_leaf_acc = self.TreeLSTMLoss(child, leaves)
      loss += child_loss
      acc += child_acc
      leaf_acc += child_leaf_acc
    return loss, acc, leaf_acc


####################################################################
## Evaluation
####################################################################

class TreeCrfLoss(nn.Module):
    def __init__(self):
        super(TreeCrfLoss, self).__init__()

    # TODO: implement the loss function given the potentials, beliefs, children dictionary, and true labels
    # Output: loss value
    # Hint: Useful functions to look into are torch.logsumexp, nn.NLLLoss(),and nn.LogSoftmax
    def forward(self, unary_potentials, edge_potentials, beliefs, children_dict, true_labels):
        raise NotImplementedError("Please implement the TODO here!")

# Returns metrics from decoding tree
def treeDecoder(beliefs, true_labels, leaves_idxs):
    correct = 0
    total = 0
    leaf_correct = 0
    leaf_total = 0
    for idx in true_labels:
        if torch.argmax(beliefs[idx]).item() == true_labels[idx]:
            correct += 1
            if idx in leaves_idxs:
                leaf_correct += 1
        total += 1
        if idx in leaves_idxs:
            leaf_total += 1
    return correct / float(total), leaf_correct / float(leaf_total)


####################################################################
## Belief propagation
####################################################################

# In the below functions,
# unary_potentials is a tensor of shape (tree_len, tag_size)
#   such that unary_potentials[i] is the unary potentials at node i
# edge_potentials is a tensor of shape (tree_len, tag_size, tag_size)
#   such that edge_potentials[i] gives the edge potentials for the edge connecting node i to it's parent,
#     and edge_potential[0] is ignored, since 0 is the root node, which does not have a parent
# parents_dict is a Dict[int, int] such that parents_dict[i] gives the parent of node i
# children_dict is a Dict[int, List[int]] such that children_dict[i] gives a list of nodes that are the children of node i
# leaves_idx is a List[int], a list of all the leaf node indices from the tree

# TODO: complete this function which is a helper function for belief propogation
#       and does message passing from leaves to root given the potentials, dictionaries
#       and leaves index
# Output: return messages (variable is called msgs) which is a dictionary as specified below
def leaves_to_root(unary_potentials, edge_potentials, parents_dict, children_dict, leaves_idx):
    to_explore = set(copy.copy(leaves_idx))
    completed = set([])
    msgs = {}  # {node_idx: [message to send to parent, message received from nodes below, message received from parent]}
    iters = 0

    # Initializes some shape values
    tree_len, tag_size = unary_potentials.shape

    # Initializes a list to be copied for next round of belief prop
    next_explore = []

    while len(to_explore) != 0:
      for node in to_explore:

        # Initialization of the lists in the msgs dictionary
        if node != 0:
          parent = parents_dict[node]
        if node != 0 and parent not in msgs:
          msgs[parent] = [None, None, None]
        if node not in msgs:
          msgs[node] = [None, None, None]

        # Identify relevant child messages
        if msgs[node][1] is not None:
          if len(msgs[node][1]) == 1:
            child_message = msgs[node][1]
          else:
            assert(len(msgs[node[1]]) == 2)
            child_message = torch.multiply(msgs[node][1][0], msgs[node][1][1])

        # Incorporate unary_potential
        if msgs[node][1] is None:
          message_with_unary = unary_potentials[node]
        else:
          message_with_unary = torch.add(unary_potentials[node], child_message)

        # Incorporate edge potential, and considers root
        # TODO
        if node != 0:
          complete_message = torch.matmul(edge_potentials[node], message_with_unary)
        else:
          complete_message = message_with_unary

        # Copy to relvant areas
        msgs[node][0] = complete_message
        if node != 0:
          if msgs[parent][1] is None:
            msgs[parent][1] = [complete_message]
          else:
            assert isinstance(msgs[parent][1], list)
            msgs[parent][1].append(complete_message)

        # Increment iter and completed
        iters += 1
        completed.append(node)

        # Check if parent is ready to propogate message
        if node != 0:
          if completed.issubset(set(children_dict[parent])):
            next_explore.append(parent)

      # Copy over the next round to explore
      to_explore = set(next_explore)
      next_explore = []

    return msgs

# TODO: complete this function which is a helper function for belief propogation
#       and does message passing from root to leaves given the potentials, dictionaries
#       and messages from leaves to root
# Output: return updates messages (same variable called msgs) which is a dictionary as specified previously
def root_to_leaves(unary_potentials, edge_potentials, children_dict, msgs):
    print(msgs)
    raise NotImplementedError("Please implement the TODO here!")

# TODO: implement belief propogation given the potentials along with parent and children dictionary and leaf index
# Output: A dictionary representing the beliefs where keys correspond to leaves_idx
# Hint: It could be useful to look at the train loop first to get further understanding of the inputs/ how the 
#       function fits into the training process
# Hint: For modularity and easier debugging it could be useful to utilize the two functions above that passes messages leaves to root
#       and root to leaves
def belief_propagation(unary_potentials, edge_potentials, parents_dict, children_dict, leaves_idx):
    raise NotImplementedError("Please implement the TODO here!")


####################################################################
## Train loop
####################################################################

def train(model, train_dataset, val_dataset, num_epochs=1):
  myTagger = TreeModule(train_dataset.vocab_size, EMBEDDING_DIM, HIDDEN_DIM, train_dataset.tag_size, model)
  
  optim = torch.optim.Adam(myTagger.parameters(),lr=0.00001)
  if model != 'baseline':
    loss_func = TreeCrfLoss()

  train_losses = []
  train_accs = []
  train_leaf_accs = []
  val_losses = []
  val_accs = []
  val_leaf_accs = []

  for epoch in range(num_epochs):
    print('-------------- Epoch {} ---------------'.format(epoch))
    done = False

    sq = 0
    print(train_dataset.len)
    start = time.time()
    while not done:
      done, train_example = train_dataset.get_next_batch()
      sentence, _, my_tree, _, tree_len = train_example

      optim.zero_grad()
      
      forwardOutput = myTagger(sentence, tree_len, my_tree)
      if model == 'baseline':
        loss = forwardOutput[0]
        accuracy = forwardOutput[1]
        leaf_acc = forwardOutput[2]
      else:
        unary_potentials = forwardOutput[0]
        edge_potentials = forwardOutput[1]

        parents_dict, children_dict, leaves_idx, true_labels = myTagger.get_structure_dicts(my_tree)
        beliefs = belief_propagation(unary_potentials, edge_potentials, parents_dict, children_dict, leaves_idx)
        loss = loss_func(unary_potentials, edge_potentials, beliefs, children_dict, true_labels)

      loss.backward()
      if model != 'baseline':
        nn.utils.clip_grad_value_(myTagger.parameters(), 5)
      optim.step()

      train_losses.append(loss.item())
      if model == 'baseline':
        train_accs.append(accuracy)
        train_leaf_accs.append(leaf_acc)

      if sq % 1000 == 0:
        with torch.no_grad():
            if model != 'baseline':
              accuracy, leaf_acc = treeDecoder(beliefs, true_labels, leaves_idx)
              train_accs.append(accuracy)
              train_leaf_accs.append(leaf_acc)

            dev_done = False
            ct = 0
            val_dataset.reset()
            while (not dev_done):
              ct += 1
              if ct % 100 == 0:
                print('ct: ', ct)
              dev_done, dev_example = val_dataset.get_next_batch()
              dev_sentence = dev_example[0]
              dev_my_tree = dev_example[2]
              dev_tree_len = dev_example[4]
              forwardOutput = myTagger(dev_sentence, dev_tree_len, dev_my_tree)

              if model == 'baseline':
                val_loss = forwardOutput[0]
                val_accuracy = forwardOutput[1]
                val_leaf_acc = forwardOutput[2]
              else:
                unary_potentials = forwardOutput[0]
                edge_potentials = forwardOutput[1]
                parents_dict, children_dict, leaves_idx, true_labels = myTagger.get_structure_dicts(dev_my_tree)
                beliefs = belief_propagation(unary_potentials, edge_potentials, parents_dict, children_dict,
                                              leaves_idx)
                val_loss = loss_func(unary_potentials, edge_potentials, beliefs, children_dict, true_labels)
                val_accuracy, val_leaf_acc = treeDecoder(beliefs, true_labels, leaves_idx)
              
              val_losses.append(val_loss)
              val_accs.append(val_accuracy)
              val_leaf_accs.append(val_leaf_acc)

            print('dev loss: ', val_losses[-1])
            print('dev acc: ', val_accs[-1])
            print('dev leaf acc: ', val_leaf_accs[-1])

      if sq % 100 == 0:
        print('sq: ', sq)
        print('loss: ', train_losses[-1])
        print('acc: ', sum(train_accs[-100:]) / len(train_accs[-100:]))
        print('leaf acc: ', sum(train_leaf_accs[-100:]) / len(train_leaf_accs[-100:]))

        print('time elapsed: ', time.time() - start)

      sq += 1

    train_dataset.reset()
    print ("done")


if __name__ == '__main__':
  print('loading data...')
  train_dataset = Dataset('data/ptb-train-5000.txt')
  val_dataset = Dataset('data/ptb-dev-500.txt', w2i=train_dataset.w2i)
  test_dataset = Dataset('data/ptb-test-1000.txt', w2i=train_dataset.w2i)

  print('train len: ', train_dataset.len)
  print('val len: ', val_dataset.len)
  print('test len: ', test_dataset.len)

  print('start training baseline...')
  train('baseline', train_dataset, test_dataset)

  print('start training LSTM + CRF...')
  train('crf', train_dataset, val_dataset)


