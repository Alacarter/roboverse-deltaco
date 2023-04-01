import numpy as np
from itertools import combinations

class OrderedVerticesWeightedGraph:
    def __init__(self, vertices_list, edge_array=None, clusters_dict=None):
        self.vertices = list(vertices_list)
        self.num_vertices = len(self.vertices)
        if edge_array is None:
            assert isinstance(clusters_dict, dict)
            # clusters_dict maps from cluster name (idf) to member vertices.
            self.create_edge_array(clusters_dict)
        else:
            assert isinstance(edge_array, np.ndarray)
            assert np.all(edge_array >= 0)
            self.edge_array = np.array(edge_array)

    def create_edge_array(self, clusters_dict):
        self.edge_array = np.zeros(
            (self.num_vertices, self.num_vertices), dtype=int)
        for cluster, cluster_vertices in clusters_dict.items():
            for pair in combinations(cluster_vertices, 2):
                self.add_edge(*pair)

    def add_edge(self, u, v):
        assert u != v, "No Self-edges allowed"
        u_idx = self.vertices.index(u)
        v_idx = self.vertices.index(v)
        self.edge_array[u_idx][v_idx] += 1
        self.edge_array[v_idx][u_idx] += 1

    def get_random_neighbors(self, u, num_neighbors):
        """returns list of `num_neighbors` random neighbors of vertex."""
        all_neighbors, selection_probs = self.get_all_neighbors(u)

        if len(all_neighbors) > num_neighbors:
            random_neighbors = np.random.choice(
                all_neighbors, size=num_neighbors, p=selection_probs,
                replace=False)
            random_neighbors = list(random_neighbors)
        else:
            random_neighbors = all_neighbors
        return random_neighbors

    def get_all_neighbors(self, u):
        """returns list of all neighbors of vertex"""
        u_idx = self.vertices.index(u)
        v_indices = np.where(self.edge_array[u_idx] > 0)[0]
        # use_edge_weights_as_selection_probs
        selection_probs = []
        if np.sum(self.edge_array[u_idx]) > 0:
            selection_probs = (
                self.edge_array[u_idx] / np.sum(self.edge_array[u_idx]))
            selection_probs = selection_probs[v_indices]
            # print("self.edge_array[u_idx]", self.edge_array[u_idx])
            # print("selection_probs", selection_probs)
            # print("v_indices", v_indices)
            all_neighbors = [self.vertices[v] for v in v_indices]
        else:
            # No outgoing edges from u
            all_neighbors = []
        return list(all_neighbors), selection_probs

    def get_random_maybe_neighbors(self, u, num_maybe_neighbors):
        """
        Returns list of neighbors of vertex.
        If not enough valid ones to get to `num_maybe_neighbors`, return
        randomly selected self.vertices names
        """
        random_neighbors = self.get_random_neighbors(u, num_maybe_neighbors)
        num_non_neighbors_to_select = (
            num_maybe_neighbors - len(random_neighbors))

        if num_non_neighbors_to_select > 0:
            # Add other random vertices to fulfill num_maybe_neighbors
            ineligible = random_neighbors + [u]
            non_neighbors = [v for v in self.vertices if v not in ineligible]
            random_non_neighbors = np.random.choice(
                non_neighbors, size=num_non_neighbors_to_select, replace=False)
            random_maybe_neighbors = (
                random_neighbors + list(random_non_neighbors))
        else:
            random_maybe_neighbors = random_neighbors

        # Check no duplicates
        assert (len(random_maybe_neighbors + [u])
                == len(set(random_maybe_neighbors + [u]))
                == (num_maybe_neighbors + 1))

        return random_maybe_neighbors

    def __add__(self, other, multiplier=1):
        assert isinstance(other, OrderedVerticesWeightedGraph)
        assert self.vertices == other.vertices
        new_edge_array = self.edge_array + multiplier * other.edge_array
        return OrderedVerticesWeightedGraph(
            self.vertices, edge_array=new_edge_array)

    def __sub__(self, other):
        new_graph = self.__add__(other, multiplier=-1)
        return new_graph

    def __eq__(self, other):
        if not isinstance(other, OrderedVerticesWeightedGraph):
            return False
        vertices_match = self.vertices == other.vertices
        edges_match = np.all(self.edge_array - other.edge_array == 0)
        return vertices_match and edges_match


if __name__ == "__main__":
    from roboverse.assets.meta_env_object_lists import (
        PICK_PLACE_TRAIN_TASK_OBJECTS_BY_COLOR_MAP,
        PICK_PLACE_TRAIN_TASK_OBJECTS_BY_SHAPE_MAP,
        PICK_PLACE_TRAIN_TASK_OBJECTS
    )
    possible_objs_flattened = [
        item for pair in PICK_PLACE_TRAIN_TASK_OBJECTS for item in pair]
    g_color = OrderedVerticesWeightedGraph(
        possible_objs_flattened,
        clusters_dict=PICK_PLACE_TRAIN_TASK_OBJECTS_BY_COLOR_MAP)
    g_shape = OrderedVerticesWeightedGraph(
        possible_objs_flattened,
        clusters_dict=PICK_PLACE_TRAIN_TASK_OBJECTS_BY_SHAPE_MAP)
    g_total = g_color + g_shape
    print(g_color.edge_array)
    print(g_shape.edge_array)
    print(g_total.edge_array)
    np.savetxt("20220824.csv", g_total.edge_array, fmt="%.0e", delimiter=",")
    assert g_total - g_color == g_shape
    assert g_total - g_shape == g_color
    # import ipdb; ipdb.set_trace()
