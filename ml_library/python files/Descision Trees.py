def count_recurrence(x):
        strata = {}
        for y in x:
            if y not in strata.keys():
                strata[y] = 0
            strata[y] += 1
        return strata
def entropy(x):
    y = count_recurrence(x)
    prob = []
    for keys in y.keys():
        prob.append(y[keys]/len(x))
    logprob = np.log2(prob)
    gini = sum(-x*y for x,y in zip(prob,logprob))
    return gini
class Node:
        def __init__(self, values :list,threshold = None, left = None, right = None):
            self.values = values
            self.left = left
            self.right = right
            self.threshold = threshold
            self.loss = entropy(self.values) - entropy(left.values) - entropy(right.values) if left and right else 0
        def __repr__(self):
            return f"(values = {self.values},\n left = {self.left},\n right = {self.right}) \n"
def find_best_threshold(x):
    strata = count_recurrence(x).keys()
    strata = list(strata)
    threshold = strata[0]
    left_values = [value for value in x if value <= threshold]
    right_values = [value for value in x if value > threshold]
    loss = entropy(x) - ((len(left_values)*entropy(left_values) + len(right_values)*entropy(right_values))/len(x))
    for y in strata:
        threshold_ = y
        left_values_ = [value for value in x if value <= threshold_]
        right_values_ = [value for value in x if value > threshold_]
        loss_ = entropy(x) - ((len(left_values_)*entropy(left_values_) + len(right_values_)*entropy(right_values_))/len(x))
        if loss_ > loss:
            loss = loss_
            threshold = threshold_
    return threshold


class DecisionTree:

    def build_tree(self,x):
        if len(np.unique(x)) == 1:
            return Node(values = x[0])
        threshold = find_best_threshold(x)
        left_values = [value for value in x if value <= threshold]
        right_values = [value for value in x if value > threshold]

        node = Node(x)
        node.threshold = threshold

        if left_values:
            node.left = self.build_tree(left_values)
        if right_values:
            node.right = self.build_tree(right_values)
        return node




