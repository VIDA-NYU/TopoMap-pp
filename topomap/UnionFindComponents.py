import numpy as np
from .utils import get_hull, AuxHull

class Component:
    def __init__(self, 
                 point_id):
        self.points_ids = [point_id]
        self.hull_ids = [point_id]
        self.hull_simplices = [[point_id,point_id]]

class UnionFind:
    def __init__(self, n, 
                 compute_hulls=True):
        self.parent = [i for i in range(n)]
        self.size = [1]*n
        self.count = n
        self.component = self.make_components(n)
        self.compute_hulls = compute_hulls

    def make_components(self, n):
        component = []
        for i in range(n):
            c = Component(i)
            component.append(c)
        return component

    def find(self, node):
        while node != self.parent[node]:
            self.parent[node] = self.parent[self.parent[node]]
            node = self.parent[node]
        return node
    
    def find_component(self, node):
        node = self.find_id(node)
        return self.component[node]
    
    def compute_union_hull(self, root1, root2, projections):
        if self.size[root1]+self.size[root2] == 2:
            return [root1, root2]
        else:
            h1 = self.component[root1].hull_ids
            h2 = self.component[root2].hull_ids
            h1.extend(h2)

            if projections[h1,0].max() == projections[h1,0].min():
                min_id = projections[h1,1].argmin() 
                max_id = projections[h1,1].argmax() 
                return [h1[min_id], h1[max_id]]
            
            union_hull = get_hull(projections[h1,:])
            union_hull_ids = [h1[i] for i in union_hull.vertices]

            return union_hull_ids
    
    def union(self, node1, node2, projections=None):
        root1 = self.find(node1)
        root2 = self.find(node2)

        if root1 == root2:
            return

        if self.size[root1] >= self.size[root2]:
            self.parent[root2] = root1
            self.size[root1] += self.parent[root2]
            self.component[root1].points_ids.extend(self.component[root2].points_ids)

            if self.compute_hulls:
                new_hull_ids = self.compute_union_hull(root1, root2, projections)
                self.component[root1].hull_ids = new_hull_ids
                m = len(new_hull_ids)
                simplices = [(new_hull_ids[i], new_hull_ids[(i+1)%m]) for i in range(m)]
                self.component[root1].hull_simplices = simplices

            self.component[root2] = None
        else:
            self.parent[root1] = root2
            self.size[root2] += self.parent[root1]
            self.component[root2].points_ids.extend(self.component[root1].points_ids)

            if self.compute_hulls:
                new_hull_ids = self.compute_union_hull(root2, root1, projections)
                self.component[root2].hull_ids = new_hull_ids
                m = len(new_hull_ids)
                simplices = [(new_hull_ids[i], new_hull_ids[(i+1)%m]) for i in range(m)]
                self.component[root2].hull_simplices = simplices

            self.component[root1] = None
        
        self.count -= 1