import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from matplotlib.colors import ListedColormap
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

h = .02  # step size in the mesh

from features import mcr, ns

# logger.info(mcr.get_ratio("asd5122jsd"))
ns.get_score("facebook")