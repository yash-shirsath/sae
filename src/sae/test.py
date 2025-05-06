# %%
import torch

# %%
print(torch.__version__)


# %%
class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right


# %%
a = TreeNode(1)
a.__dict__["parent"] = TreeNode(2)

if "parent" in a.__dict__:
    print("cool")
else:
    print("a.left is not TreeNode")

# %%
