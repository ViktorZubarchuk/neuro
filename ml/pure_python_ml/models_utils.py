import numpy as np
from scipy.stats import mode

class LinearRegressionOneFeature:
    def __init__(self, learning_rate, epochs):
        self.w = np.random.rand()
        self.b = np.random.rand()
        self.learning_rate = learning_rate
        self.epochs = epochs
    
    def predict(self, x):
        y_pred = x*self.w + self.b
        return y_pred
    
    def calc_mse(self, y, y_pred):
        mse = np.mean((y_pred-y)**2)
        return mse
    
    def gradient(self, x, y, y_pred):
        dw = 2*np.mean(x*(y_pred-y)) 
        db = 2*np.mean(y_pred-y)
        return dw, db
    
    def fit(self, x, y):
        for i in range(self.epochs):
            y_pred = self.predict(x)
            dw, db = self.gradient(x, y, y_pred)
            self.w -= dw*self.learning_rate
            self.b -= db*self.learning_rate
            if i%100==0:
                print(f"{i}: MSE = {self.calc_mse(y, y_pred)}, w = {self.w}, b = {self.b}")

class LinearRegression:
    def __init__(self, learning_rate, epochs):
        self.learning_rate = learning_rate
        self.epochs = epochs

        self.mean = None
        self.std = None
    
    def predict(self, x):
        x_norm = self.normalize(x)
        y_pred = x_norm.dot(self.w) + self.b
        return y_pred
    
    def normalize(self, x, fit = False):
        if fit:
            self.mean = np.mean(x, axis=0)
            self.std = np.std(x, axis=0)
        return (x - self.mean) / (self.std + 1e-8)

    def calc_mse(self, y, y_pred):
        mse = np.mean((y_pred-y)**2)
        return mse
    
    def gradient(self, x, y, y_pred):
        n = len(y)
        dw = (2/n) * x.T.dot(y_pred - y)
        db = 2*np.mean(y_pred-y)
        return dw, db
    
    def fit(self, x, y):
        self.w = np.random.rand(x.shape[1])
        self.b = np.random.rand()
        x_norm = self.normalize(x, fit=True)
        for i in range(self.epochs):
            y_pred = x_norm.dot(self.w) + self.b
            dw, db = self.gradient(x_norm, y, y_pred)
            self.w -= dw*self.learning_rate
            self.b -= db*self.learning_rate
            if i%100==0:
                print(f"{i}: MSE = {self.calc_mse(y, y_pred)}, w = {self.w}, b = {self.b}")

class LogisticRegression:
    def __init__(self):
        self.learning_rate = 0.01
        self.epochs = 1000
        self.mean = None
        self.std = None

    def sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def normalize(self, x, fit = False):
        if fit:
            self.mean = np.mean(x, axis=0)
            self.std = np.std(x, axis=0)
        return (x - self.mean) / (self.std + 1e-8)

    def predict(self, x):
        x_norm = self.normalize(x)
        y_pred = self.sigmoid(x_norm.dot(self.w)+self.b)
        return (y_pred > 0.5).astype(int)
    
    def calc_cross_entropy(self, y, y_pred):
        loss = -np.mean(y*np.log(y_pred)+(1-y)*np.log(1-y_pred))
        return loss

    def gradient(self, x, y, y_pred):
        m = x.shape[0]
        dw = x.T.dot(y_pred-y)/m
        db = np.sum(y_pred-y)/m
        return dw,db
    
    def fit(self, x, y):
        self.w = np.random.rand(x.shape[1])
        self.b = np.random.rand()
        x_norm = self.normalize(x, fit=True)
        for i in range(self.epochs):
            y_pred = self.sigmoid(x_norm.dot(self.w)+self.b)
            dw, db = self.gradient(x_norm, y, y_pred)
            self.w -= self.learning_rate * dw
            self.b -= self.learning_rate * db
            if i%100==0:
                print(f"{i}: cross_entropy = {self.calc_cross_entropy(y, y_pred)}, w = {self.w}, b = {self.b}")

class KNN():
    def __init__(self):
        self.k = 3

    def euclid_dist(self, X_train, X_test):
        dist = np.sqrt(np.sum((X_train-X_test)**2, axis=1))
        return dist
    
    def predict(self, X_train, X_test, y_train):
        dist = self.euclid_dist(X_train, X_test)
        indices = np.argsort(dist)[:self.k]
        neighbors = y_train[indices]
        return mode(neighbors)
    
class DesicionTree():
    def __init__(self):
        self.max_depth = None
        self.min_samples_split = None
        self.root = None

    def predict_single(self, x, node):
        if node.value is not None: 
            return node.value
        feature_value = x[node.feature_idx]
        if feature_value < node.threshold: 
            return self.predict_single(x, node.left)
        else: 
            return self.predict_single(x, node.right)
        
    def gini(self, y):
        classes = np.unique(y)
        res = 1
        for cls in classes:
            p = np.sum(y==cls)/len(y)
            res -= p**2
        return res
    
    def split(self, X_colomn, threshold):
        left_idxs = np.argwhere(X_colomn < threshold).flatten()
        right_idxs = np.argwhere(X_colomn >= threshold).flatten()
        return left_idxs, right_idxs
    
    def split_gini(self, y, X_colomn, threshold):
        left_idxs, right_idxs = self.split(X_colomn, threshold)
        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 999
        left_gini = self.gini(y[left_idxs])
        right_gini = self.gini(y[right_idxs])
        n = len(y)
        weighted_gini = len(left_idxs) / n * left_gini + len(right_idxs) / n * right_gini
        return weighted_gini

    def best_split(self, X, y):
        best_gini = 999
        best_feature = None
        best_threshold = None
        n_features = X.shape[1]
        for feature_idx in range(n_features):
            n_threshold = np.unique(X[:, feature_idx])
            for thr in n_threshold:
                split_gini = self.split_gini(y, X[:, feature_idx], thr)
                if split_gini < best_gini:
                    best_gini = split_gini
                    best_feature = feature_idx
                    best_threshold = thr
        return best_feature, best_threshold
    
    def build_tree(self, X, y):
        if len(np.unique(y)) == 1:
            return Node(value = y[0]) 
        feature_idx, threshold = self.best_split(X, y)
        left_idxs, right_idxs = self.split(X[:, feature_idx], threshold)
        X_left = X[left_idxs]
        y_left = y[left_idxs]
        X_right = X[right_idxs]
        y_right = y[right_idxs]
        left_tree = self.build_tree(X_left, y_left)
        right_tree = self.build_tree(X_right, y_right)
        return Node(
            feature_idx=feature_idx,
            threshold=threshold,
            left=left_tree,
            right=right_tree)
    
    def fit(self, X, y):
        self.root = self.build_tree(X, y)

    def predict(self, X):
        predictions = [self.predict_single(x, self.root) for x in X]
        return np.array(predictions)


class Node:
    def __init__(self, feature_idx=None, threshold=None, left=None, right=None, value=None):
        self.feature_idx = feature_idx  # индекс признака, по которому делим
        self.threshold = threshold      # порог, по которому делим
        self.left = left                # левое поддерево
        self.right = right              # правое поддерево
        self.value = value              # если это лист, здесь класс
    

class RandomForest():
    def __init__(self, n_trees):
        self.n_trees = n_trees
        self.trees = []

    def bootstrap(self, X, y):
        X_sample = []
        y_sample = []
        n = len(X)
        for i in range(n):
            random_idx = np.random.randint(0,n)
            X_sample.append(X[random_idx]) 
            y_sample.append(y[random_idx])
        X_sample = np.array(X_sample)
        y_sample = np.array(y_sample)
        return X_sample, y_sample
    
    def build_tree(self, X, y):
        if len(np.unique(y)) == 1:
            return Node(value = y[0]) 
        feature_idx, threshold = self.best_split(X, y)
        left_idxs, right_idxs = self.split(X[:, feature_idx], threshold)
        X_left = X[left_idxs]
        y_left = y[left_idxs]
        X_right = X[right_idxs]
        y_right = y[right_idxs]
        left_tree = self.build_tree(X_left, y_left)
        right_tree = self.build_tree(X_right, y_right)
        return Node(
            feature_idx=feature_idx,
            threshold=threshold,
            left=left_tree,
            right=right_tree)
    
    def fit(self, X, y):
        for i in range(self.n_trees):
            X_sample, y_sample = self.bootstrap(X, y)
            tree = DesicionTree()
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        preds = np.array(tree.predict(X) for tree in self.trees)
        preds = preds.T
        return mode(preds, axis=1, keepdims=False).mode 


            
    
    

