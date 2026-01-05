class KNearestNeighbours:
    def __init__(self, k):
        self.k = k
    def euclidean_distances(self,point1,point2):
        return (sum(((x1 - x2)**2 for x1,x2 in zip(point1,point2))))**(1/2)
    def predict(self,training_points,training_labels,test_point):
        distances = [(self.euclidean_distances(test_point,x),label) for x,label in zip(training_points,training_labels)]
        distances.sort(key = lambda x: x[1],reverse = True)
        distances = distances[:self.k]
        return self.most_common(distances)
    def most_common(self,distances):
        def count_recurrence(x):
            strata = {}
            for y in x:
                if y[1] not in strata.keys():
                    strata[y[1]] = 0
                strata[y[1]] += 1
            return strata
        counter = [(x,y) for x,y in zip(count_recurrence(distances).keys(),count_recurrence(distances).values())]
        counter.sort(key = lambda x: x[1])
        return counter[0][0]

