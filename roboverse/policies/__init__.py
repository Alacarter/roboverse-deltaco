from .pick_place import PickPlace
from .grasp import (
    Grasp,
    GraspContinuous
)
from .place import Place

policies = dict(
    grasp=Grasp,
    grasp_continuous=GraspContinuous,
    pickplace=PickPlace,
    place=Place,
)
