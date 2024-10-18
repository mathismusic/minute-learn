import numpy as np
import networkx
import matplotlib.pyplot as plt
from networkx.drawing.nx_pydot import graphviz_layout
np.random.seed(0)

class DecisionTreeClassifier:
    def __init__(self) -> None:
        self.root = None
        self.children = None
        self.counts = None # counts of each class at this node

    @staticmethod
    def _entropy(p: np.ndarray):
        p = p.astype(np.float32)/p.sum()
        return -(p * np.log2(p)).sum()
    
    def fit(self, X: np.ndarray, y: np.ndarray, attrs=None):
        if attrs is None:
            attrs = set(range(X.shape[1]))
       
        if len(attrs) == 0:
            self.children = None
            classes, self.counts = np.unique(y, return_counts=True)
            self.prediction = classes[np.argmax(self.counts)]
            return self
        
        # if all examples have the same class, return a leaf node
        if len(np.unique(y)) == 1:
            self.children = None
            self.prediction = y[0]
            self.counts = np.array([len(y)])
            return self
        
        best, bestval = None, -np.inf
        for attr in attrs:
            _, Sv = np.unique(X[:, attr], return_counts=True)
            vy_pairs = np.stack((X[:, attr], y)).T
            _, Svy = np.unique(vy_pairs, return_counts=True, axis=0)
            ig = self._entropy(Svy) - self._entropy(Sv)
            print(ig)
            if ig > bestval: best, bestval = attr, ig
        attr = best
        attrs2 = attrs.copy()
        attrs2.discard(attr)
        self.root = attr
        vals = np.unique(X[:, attr])
        dataset_masks = [(X[:, attr] == v) for v in vals]
        print(dataset_masks)
        # exit()
        self.bucketing_rule = lambda val: val
        self.children = {val:DecisionTreeClassifier().fit(X[mask], y[mask], attrs2) for val, mask in zip(vals, dataset_masks)}
        
        # what to classify if we must stop here (mandatory for leaf nodes, and for nonleaf if the attribute value was never seen during training)
        classes, self.counts = np.unique(y, return_counts=True)
        self.prediction = classes[np.argmax(self.counts)]
        return self

    def _predict(self, x):
        if self.root is None: return None
        if self.children is None: # leaf node
            return self.prediction
        val = x[self.root]
        print(self.bucketing_rule(val))
        print(len(self.children))
        bucket = self.bucketing_rule(val)
        if bucket not in self.children:
            # unseen data
            return self.prediction
        return self.children[self.bucketing_rule(val)]._predict(x)
    
    def predict(self, X):
        return np.array([self._predict(x) for x in X])
    
    @property
    def _data(self):
        return self.root, self.prediction, tuple(self.counts.tolist())
    
    def asnetworkx(self, id):
        G = networkx.DiGraph()
        root_data = (*self._data, id[0]); id[0] += 1
        G.add_node(root_data, size=25)
        if self.children is None: 
            return G
        for child, childtree in self.children.items():
            G.add_edge(root_data, (*childtree._data, id[0]), label=child)
            G = networkx.compose(G, childtree.asnetworkx(id))
            id[0] += 1
        return G


    def print(self):
        G = self.asnetworkx([0])

        plt.figure(figsize=(20, 20))
        pos = graphviz_layout(G, prog='dot') # to render nicely
        
        networkx.draw(G, pos, with_labels=True, labels={node: f'{node[0]}{node[2]}' for node in G.nodes})
        networkx.draw_networkx_edge_labels(G, pos, edge_labels=networkx.get_edge_attributes(G, 'label'))
        
        # save
        plt.savefig('tree.png')

        
if __name__ == '__main__':
    import gen
    feats = 4
    featrange = 2
    X = gen.discrete(500, feats, featrange)
    print(X)
    # y = ((X[:, 0] < featrange/3) == (X[:, 1] < featrange/2)).astype(np.int32)
    bitstrings = ['0110', '1010']
    y = np.zeros(X.shape[0], dtype=np.int32)
    for bitstr in bitstrings:
        y += (sum([X[:, i] == int(bitstr[i]) for i in range(feats)]) == feats).astype(np.int32)
    y = (y > 0).astype(np.int32)
    # plt.xticks(np.arange(0, featrange, 5))
    # plt.yticks(np.arange(0, featrange, 5))
    # plt.xticks(np.arange(featrange), minor=True)
    # plt.yticks(np.arange(featrange), minor=True)
    # plt.grid(visible=True, which='both')
    # plt.scatter(X[:, 0], X[:, 1], c=y)
    # plt.show()
    d = DecisionTreeClassifier()
    d.fit(X, y)
    d.print()
    # x = np.array([2, 2])
    # pred = d.predict(x.reshape(1, 2))
    # print(pred)
    # print(y)
    # print(np.unique(y, return_index=True))