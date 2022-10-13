# https://github.com/topepo/caret

from rpy2 import robjects

pi = robjects.r["pi"]
print(pi)
