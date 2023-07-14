"""
 Copyright 2019 Xilinx Inc.
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

"""
VF2 Algorithm for Graph-Subgraph Isomorphism
"""

import sys
from typing import Dict, Generator, List, NoReturn, Optional, Set, Tuple

from xnnc.ir.xmodel import XModel
from xnnc.ir.xnode import XModelNode

__all__ = ["DiGraphMatcher"]


class DiGraphMatcher(object):
    """Implementation of VF2 algorithm for finding subgraph matchings in a (directed) graph (G1), given a (directed) subgraph (G2).
    """

    def __init__(self, G1: XModel, G2: XModel):
        """Initialize DiGraphMatcher instance.
        
        Parameters
        ----------
        G1 : XModel
            XModel instance on which to search subgraph matching.
        G2 : XModel
            XModel instance as subgraph pattern.
        """
        self.G1: XModel = G1
        self.G2: XModel = G2
        self.G1_nodes: Set[XModelNode] = set(G1.xnodes)
        self.G2_nodes: Set[XModelNode] = set(G2.xnodes)
        self.G2_node_order: Dict[XModelNode, int] = {
            n: i for i, n in enumerate(G2.xnodes)
        }

        # Set recursion limit.
        self.old_recursion_limit = sys.getrecursionlimit()
        expected_max_recursion_level = len(self.G2.xnodes)
        if self.old_recursion_limit < 1.5 * expected_max_recursion_level:
            # Give some breathing room.
            sys.setrecursionlimit(int(1.5 * expected_max_recursion_level))

    def search_matching(self,) -> Optional[List[Dict[XModelNode, XModelNode]]]:
        """Find all subgraph matches on G1, which are isomorphic to G2.
        
        Returns
        -------
        Optional[List[Dict[XModelNode, XModelNode]]]
            List of subgraph matches.
        """
        self.__initialize()

        matches = []
        # search matches recursively
        self.__match(matches)

        if len(matches) > 0:
            values, res = [], []
            for match in matches:
                hash_val = hash(str(sorted([x.op_name for x in list(match.keys())])))
                if hash_val in values:
                    continue
                values.append(hash_val)
                res.append(match)
            return res
        return None

    def __initialize(self) -> NoReturn:
        """Reinitializes the state of the algorithm.
        """
        # core_1[n] contains the index of the node paired with n, which is m,
        #           provided n is in the mapping.
        # core_2[m] contains the index of the node paired with m, which is n,
        #           provided m is in the mapping.
        self.core_1: Dict[XModelNode, XModelNode] = {}
        self.core_2: Dict[XModelNode, XModelNode] = {}

        # See the paper for definitions of M_x and T_x^{y}

        # in_1[n]  is non-zero if n is in M_1 or in T_1^{in}
        # out_1[n] is non-zero if n is in M_1 or in T_1^{out}
        #
        # in_2[m]  is non-zero if m is in M_2 or in T_2^{in}
        # out_2[m] is non-zero if m is in M_2 or in T_2^{out}
        #
        # The value stored is the depth of the search tree when the node became
        # part of the corresponding set.
        self.in_1 = {}
        self.in_2 = {}
        self.out_1 = {}
        self.out_2 = {}

        self.state = DiGMState(self)

        # Provide a convenient way to access the isomorphism mapping.
        self.mapping = self.core_1.copy()

    def __match(self, matches: List[Dict[XModelNode, XModelNode]]) -> NoReturn:
        """Search all matches recursively.
        
        Parameters
        ----------
        matches : List[Dict[XModelNode, XModelNode]]
            Graph-subgraph isomorphic matches
        """
        assert matches is not None, "'matches' should not be None."
        assert isinstance(matches, list), "'matches' should be of list type."

        if len(self.core_1) == len(self.G2.xnodes):
            self.mapping = self.core_1.copy()
            matches.append(self.mapping)
        else:
            candidate_pairs = list(self.__get_candidate_pairs())
            for G1_node, G2_node in candidate_pairs:
                if self.__syntactic_feasibility(G1_node, G2_node):
                    if self.__semantic_feasibility(G1_node, G2_node):
                        newstate = self.state.__class__(self, G1_node, G2_node)
                        self.__match(matches)

                        # restore state variables
                        newstate.restore()

    def __get_candidate_pairs(self) -> Generator[XModelNode, XModelNode, None]:
        """Get candidate node pairs for matching.
        """

        # All computations are done using the current state!

        G1_nodes = self.G1_nodes
        G2_nodes = self.G2_nodes
        min_key = self.G2_node_order.__getitem__

        # First we compute the out-terminal sets.
        T1_out = [node for node in self.out_1 if node not in self.core_1]
        T2_out = [node for node in self.out_2 if node not in self.core_2]

        # If T1_out and T2_out are both nonempty.
        # P(s) = T1_out x {min T2_out}
        if T1_out and T2_out:
            node_2 = min(T2_out, key=min_key)
            for node_1 in T1_out:
                yield node_1, node_2

        # If T1_out and T2_out were both empty....
        # We compute the in-terminal sets.

        # elif not (T1_out or T2_out):   # as suggested by [2], incorrect
        else:  # as suggested by [1], correct
            T1_in = [node for node in self.in_1 if node not in self.core_1]
            T2_in = [node for node in self.in_2 if node not in self.core_2]

            # If T1_in and T2_in are both nonempty.
            # P(s) = T1_out x {min T2_out}
            if T1_in and T2_in:
                node_2 = min(T2_in, key=min_key)
                for node_1 in T1_in:
                    yield node_1, node_2

            # If all terminal sets are empty...
            # P(s) = (N_1 - M_1) x {min (N_2 - M_2)}

            # elif not (T1_in or T2_in):   # as suggested by  [2], incorrect
            else:  # as inferred from [1], correct
                node_2 = min(G2_nodes - set(self.core_2), key=min_key)
                for node_1 in G1_nodes:
                    if node_1 not in self.core_1:
                        yield node_1, node_2

        # For all other cases, we don't have any candidate pairs.

    def __syntactic_feasibility(self, G1_node: XModelNode, G2_node: XModelNode) -> bool:
        """Check if two nodes are matched syntactically.
        
        Parameters
        ----------
        G1_node : XModelNode
            node from graph
        G2_node : XModelNode
            node from subgraph
        
        Returns
        -------
        bool
            True if the two nodes are matched syntactically; otherwise, False.
        """
        # Look ahead 0

        # * R_self

        # The number of selfloops for G1_node must equal the number of
        # self-loops for G2_node. Without this check, we would fail on R_pred
        # at the next recursion level. This should prune the tree even further.
        if self.G1.number_of_edges(G1_node, G1_node) != self.G2.number_of_edges(
            G2_node, G2_node
        ):
            return False

        # * R_pred

        # For each predecessor n' of n in the partial mapping, the
        # corresponding node m' is a predecessor of m, and vice versa. Also,
        # the number of edges must be equal
        for predecessor in self.G1.predecessors(G1_node):
            if predecessor in self.core_1:
                if not (self.core_1[predecessor] in self.G2.predecessors(G2_node)):
                    return False
                elif self.G1.number_of_edges(
                    predecessor, G1_node
                ) != self.G2.number_of_edges(self.core_1[predecessor], G2_node):
                    return False

        for predecessor in self.G2.predecessors(G2_node):
            if predecessor in self.core_2:
                if not (self.core_2[predecessor] in self.G1.predecessors(G1_node)):
                    return False
                else:
                    if self.G1.number_of_edges(
                        self.core_2[predecessor], G1_node
                    ) != self.G2.number_of_edges(predecessor, G2_node):
                        return False

        # * R_succ

        # For each successor n' of n in the partial mapping, the corresponding
        # node m' is a successor of m, and vice versa. Also, the number of
        # edges must be equal.
        for successor in self.G1.successors(G1_node):
            if successor in self.core_1:
                if not (self.core_1[successor] in self.G2.successors(G2_node)):
                    return False
                elif self.G1.number_of_edges(
                    G1_node, successor
                ) != self.G2.number_of_edges(G2_node, self.core_1[successor]):
                    return False

        for successor in self.G2.successors(G2_node):
            if successor in self.core_2:
                if not (self.core_2[successor] in self.G1.successors(G1_node)):
                    return False
                else:
                    if self.G1.number_of_edges(
                        G1_node, self.core_2[successor]
                    ) != self.G2.number_of_edges(G2_node, successor):
                        return False

        # Look ahead 1

        # * R_termin
        # The number of predecessors of n that are in T_1^{in} is equal to the
        # number of predecessors of m that are in T_2^{in}.
        num1 = 0
        for predecessor in self.G1.predecessors(G1_node):
            if (predecessor in self.in_1) and (predecessor not in self.core_1):
                num1 += 1
        num2 = 0
        for predecessor in self.G2.predecessors(G2_node):
            if (predecessor in self.in_2) and (predecessor not in self.core_2):
                num2 += 1
        if not (num1 >= num2):
            return False

        # The number of successors of n that are in T_1^{in} is equal to the
        # number of successors of m that are in T_2^{in}.
        num1 = 0
        for successor in self.G1.successors(G1_node):
            if (successor in self.in_1) and (successor not in self.core_1):
                num1 += 1
        num2 = 0
        for successor in self.G2.successors(G2_node):
            if (successor in self.in_2) and (successor not in self.core_2):
                num2 += 1
        if not (num1 >= num2):
            return False

        # * R_termout

        # The number of predecessors of n that are in T_1^{out} is equal to the
        # number of predecessors of m that are in T_2^{out}.
        num1 = 0
        for predecessor in self.G1.predecessors(G1_node):
            if (predecessor in self.out_1) and (predecessor not in self.core_1):
                num1 += 1
        num2 = 0
        for predecessor in self.G2.predecessors(G2_node):
            if (predecessor in self.out_2) and (predecessor not in self.core_2):
                num2 += 1
        if not (num1 >= num2):
            return False

        # The number of successors of n that are in T_1^{out} is equal to the
        # number of successors of m that are in T_2^{out}.
        num1 = 0
        for successor in self.G1.successors(G1_node):
            if (successor in self.out_1) and (successor not in self.core_1):
                num1 += 1
        num2 = 0
        for successor in self.G2.successors(G2_node):
            if (successor in self.out_2) and (successor not in self.core_2):
                num2 += 1
        if not (num1 >= num2):
            return False

        # Look ahead 2

        # * R_new

        # The number of predecessors of n that are neither in the core_1 nor
        # T_1^{in} nor T_1^{out} is equal to the number of predecessors of m
        # that are neither in core_2 nor T_2^{in} nor T_2^{out}.
        num1 = 0
        for predecessor in self.G1.predecessors(G1_node):
            if (predecessor not in self.in_1) and (predecessor not in self.out_1):
                num1 += 1
        num2 = 0
        for predecessor in self.G2.predecessors(G2_node):
            if (predecessor not in self.in_2) and (predecessor not in self.out_2):
                num2 += 1
        if not (num1 >= num2):
            return False

        # The number of successors of n that are neither in the core_1 nor
        # T_1^{in} nor T_1^{out} is equal to the number of successors of m
        # that are neither in core_2 nor T_2^{in} nor T_2^{out}.
        num1 = 0
        for successor in self.G1.successors(G1_node):
            if (successor not in self.in_1) and (successor not in self.out_1):
                num1 += 1
        num2 = 0
        for successor in self.G2.successors(G2_node):
            if (successor not in self.in_2) and (successor not in self.out_2):
                num2 += 1
        if not (num1 >= num2):
            return False

        return True

    def __semantic_feasibility(self, G1_node: XModelNode, G2_node: XModelNode) -> bool:
        """Check if two nodes are matched semantically.
        
        Parameters
        ----------
        G1_node : XModelNode
            node from graph
        G2_node : XModelNode
            node from subgraph
        
        Returns
        -------
        bool
            True if the two nodes are matched in semantics; otherwise, False.
        """
        assert G1_node is not None, "'G1_node' should not be None."
        assert G2_node is not None, "'G2_node' should not be None."
        assert isinstance(
            G1_node, XModelNode
        ), "'G1_node' should be of XModelNode type."
        assert isinstance(
            G2_node, XModelNode
        ), "'G2_node' should be of XModelNode type."

        # op_type
        if "," in G2_node.op_type:
            op_types = [x.strip() for x in G2_node.op_type.split(",")]
            if G1_node.op_type not in op_types:
                return False
        elif G1_node.op_type != G2_node.op_type and G2_node.op_type != "any":
            return False

        return True


class DiGMState(object):
    """Internal representation of state for the DiGraphMatcher class.

    This class is used internally by the DiGraphMatcher class.  It is used
    only to store state specific data. There will be at most G2.order() of
    these objects in memory at a time, due to the depth-first search
    strategy employed by the VF2 algorithm.

    """

    def __init__(
        self, GM: DiGraphMatcher, G1_node: XModelNode = None, G2_node: XModelNode = None
    ):
        """Initializes DiGMState object.

        Pass in the DiGraphMatcher to which this DiGMState belongs and the
        new node pair that will be added to the GraphMatcher's current
        isomorphism mapping.
        """
        self.GM = GM

        # Initialize the last stored node pair.
        self.G1_node = None
        self.G2_node = None
        self.depth = len(GM.core_1)

        if G1_node is None or G2_node is None:
            # Then we reset the class variables
            GM.core_1 = {}
            GM.core_2 = {}
            GM.in_1 = {}
            GM.in_2 = {}
            GM.out_1 = {}
            GM.out_2 = {}

        # Watch out! G1_node == 0 should evaluate to True.
        if G1_node is not None and G2_node is not None:
            # Add the node pair to the isomorphism mapping.
            GM.core_1[G1_node] = G2_node
            GM.core_2[G2_node] = G1_node

            # Store the node that was added last.
            self.G1_node = G1_node
            self.G2_node = G2_node

            # Now we must update the other four vectors.
            # We will add only if it is not in there already!
            self.depth = len(GM.core_1)

            # First we add the new nodes...
            for vector in (GM.in_1, GM.out_1):
                if G1_node not in vector:
                    vector[G1_node] = self.depth
            for vector in (GM.in_2, GM.out_2):
                if G2_node not in vector:
                    vector[G2_node] = self.depth

            # Now we add every other node...

            # Updates for T_1^{in}
            new_nodes = set()
            for node in GM.core_1:
                new_nodes.update(
                    [
                        predecessor
                        for predecessor in GM.G1.predecessors(node)
                        if predecessor not in GM.core_1
                    ]
                )
            for node in new_nodes:
                if node not in GM.in_1:
                    GM.in_1[node] = self.depth

            # Updates for T_2^{in}
            new_nodes = set()
            for node in GM.core_2:
                new_nodes.update(
                    [
                        predecessor
                        for predecessor in GM.G2.predecessors(node)
                        if predecessor not in GM.core_2
                    ]
                )
            for node in new_nodes:
                if node not in GM.in_2:
                    GM.in_2[node] = self.depth

            # Updates for T_1^{out}
            new_nodes = set()
            for node in GM.core_1:
                new_nodes.update(
                    [
                        successor
                        for successor in GM.G1.successors(node)
                        if successor not in GM.core_1
                    ]
                )
            for node in new_nodes:
                if node not in GM.out_1:
                    GM.out_1[node] = self.depth

            # Updates for T_2^{out}
            new_nodes = set()
            for node in GM.core_2:
                new_nodes.update(
                    [
                        successor
                        for successor in GM.G2.successors(node)
                        if successor not in GM.core_2
                    ]
                )
            for node in new_nodes:
                if node not in GM.out_2:
                    GM.out_2[node] = self.depth

    def restore(self) -> NoReturn:
        """Deletes the DiGMState object and restores the class variables."""

        # First we remove the node that was added from the core vectors.
        # Watch out! G1_node == 0 should evaluate to True.
        if self.G1_node is not None and self.G2_node is not None:
            del self.GM.core_1[self.G1_node]
            del self.GM.core_2[self.G2_node]

        # Now we revert the other four vectors.
        # Thus, we delete all entries which have this depth level.
        for vector in (self.GM.in_1, self.GM.in_2, self.GM.out_1, self.GM.out_2):
            for node in list(vector.keys()):
                if vector[node] == self.depth:
                    del vector[node]
