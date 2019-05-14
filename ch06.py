from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, export_graphviz, DecisionTreeRegressor
import graphviz
import numpy as np

#################################################################
# iris = load_iris()
# X = iris.data[:, 2:]
# y = iris.target
#
# tree_clf = DecisionTreeClassifier(max_depth=2, random_state=42)
# tree_clf.fit(X, y)
#
# export_graphviz(
#     tree_clf,
#     out_file="iris_tree.dot",
#     feature_names=iris.feature_names[2:],
#     class_names=iris.target_names,
#     rounded=True,
#     filled=True
# )
#
# with open("iris_tree.dot") as f:
#     dot_graph = f.read()
#
# dot = graphviz.Source(dot_graph)
# dot.format = "png"
# dot.render(filename="iris_tree", cleanup=True)
###################################################################

np.random.seed(42)
m = 200
X = np.random.rand(m, 1)
y = 4 * (X - 0.5) ** 2
y = y + np.random.randn(m, 1) / 10

tree_reg = DecisionTreeRegressor(max_depth=2)
tree_reg.fit(X, y)

export_graphviz(
    tree_reg,
    out_file="reg_tree.dot",
    feature_names=["x1"],
    rounded=True,
    filled=True
)

with open("reg_tree.dot") as f:
    dot_graph = f.read()

dot = graphviz.Source(dot_graph)
dot.format = "png"
dot.render(filename="reg_tree", cleanup=True)
