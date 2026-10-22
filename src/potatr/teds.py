# Copyright 2020 IBM
# Author: peter.zhong@au1.ibm.com
# Modified from the original PubTabNet source (https://github.com/ibm-aur-nlp/PubTabNet/blob/master/src/metric.py).
#
# This is free software; you can redistribute it and/or modify
# it under the terms of the Apache 2.0 License.
#
# This software is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# Apache 2.0 License for more details.

from collections import deque

import numpy as np
from apted import APTED, Config
from apted.helpers import Tree
from lxml import etree, html
from rapidfuzz.distance import Levenshtein
from scipy.optimize import linear_sum_assignment


class TableTree(Tree):
    def __init__(self, tag, colspan=None, rowspan=None, content=None, *children):
        self.tag = tag
        self.colspan = colspan
        self.rowspan = rowspan
        self.content = content
        self.children = list(children)

    def bracket(self):
        """Show tree using brackets notation"""
        if self.tag == "td":
            result = f'"tag": {self.tag}, "colspan": {self.colspan}, "rowspan": {self.rowspan}, "text": {self.content}'
        else:
            result = f'"tag": {self.tag}'
        for child in self.children:
            result += child.bracket()
        return f"{{{result}}}"


class CustomConfig(Config):
    @staticmethod
    def maximum(*sequences):
        """Get maximum possible value"""
        return max(map(len, sequences))

    def normalized_distance(self, *sequences):
        """Get distance from 0 to 1"""
        return float(Levenshtein.distance(*sequences)) / self.maximum(*sequences)

    def rename(self, node1, node2):
        """Compares attributes of trees"""
        if (node1.tag != node2.tag) or (node1.colspan != node2.colspan) or (node1.rowspan != node2.rowspan):
            return 1.0
        if node1.tag == "td":
            if node1.content or node2.content:
                return self.normalized_distance(node1.content, node2.content)
        return 0.0


class TEDS:
    """Tree Edit Distance basead Similarity"""

    def __init__(self, structure_only=False, n_jobs=1, ignore_nodes=None):
        assert isinstance(n_jobs, int) and (n_jobs >= 1), "n_jobs must be an integer greather than 1"
        self.structure_only = structure_only
        self.n_jobs = n_jobs
        self.ignore_nodes = ignore_nodes
        self.__tokens__ = []

    def tokenize(self, node):
        """Tokenizes table cells"""
        self.__tokens__.append(f"<{node.tag}>")
        if node.text is not None:
            self.__tokens__ += list(node.text)
        for n in node.getchildren():
            self.tokenize(n)
        if node.tag != "unk":
            self.__tokens__.append(f"</{node.tag}>")
        if node.tag != "td" and node.tail is not None:
            self.__tokens__ += list(node.tail)

    def load_html_tree(self, node, parent=None):
        """Converts HTML tree to the format required by apted"""
        if node.tag == "td":
            if self.structure_only:
                cell = []
            else:
                self.__tokens__ = []
                self.tokenize(node)
                cell = self.__tokens__[1:-1].copy()
            new_node = TableTree(
                node.tag, int(node.attrib.get("colspan", "1")), int(node.attrib.get("rowspan", "1")), cell, *deque()
            )
        else:
            new_node = TableTree(node.tag, None, None, None, *deque())
        if parent is not None:
            parent.children.append(new_node)
        if node.tag != "td":
            for n in node.getchildren():
                self.load_html_tree(n, new_node)
        if parent is None:
            return new_node

    def evaluate(self, pred, true):
        """Computes TEDS score between the prediction and the ground truth of a
        given sample
        """
        if (not pred) or (not true):
            return 0.0
        parser = html.HTMLParser(remove_comments=True, encoding="utf-8")
        pred = html.fromstring(pred, parser=parser)
        true = html.fromstring(true, parser=parser)
        if pred.xpath("body/table") and true.xpath("body/table"):
            pred = pred.xpath("body/table")[0]
            true = true.xpath("body/table")[0]
            if self.ignore_nodes:
                etree.strip_tags(pred, *self.ignore_nodes)
                etree.strip_tags(true, *self.ignore_nodes)
            n_nodes_pred = len(pred.xpath(".//*"))
            n_nodes_true = len(true.xpath(".//*"))
            n_nodes = max(n_nodes_pred, n_nodes_true)
            tree_pred = self.load_html_tree(pred)
            tree_true = self.load_html_tree(true)
            distance = APTED(tree_pred, tree_true, CustomConfig()).compute_edit_distance()
            return 1.0 - (float(distance) / n_nodes)
        else:
            return 0.0


def hungarian_teds(true_htmls, pred_htmls, structure_only=False):
    teds_evaluator = TEDS(structure_only=structure_only)
    num_true = len(true_htmls)
    num_pred = len(pred_htmls)

    teds_matrix = np.zeros((num_true, num_pred))

    for true_idx, true_html in enumerate(true_htmls):
        for pred_idx, pred_html in enumerate(pred_htmls):
            try:
                score = teds_evaluator.evaluate(true_html, pred_html)
            except ValueError:
                print("Error computing TEDS")
                score = 0

            teds_matrix[true_idx, pred_idx] = score

    col_ind, row_ind = linear_sum_assignment(teds_matrix, maximize=True)
    scores = teds_matrix[col_ind, row_ind]
    total_score = scores.sum()
    num_exact_match = (scores == 1).sum()

    return num_true, num_pred, total_score, num_exact_match
