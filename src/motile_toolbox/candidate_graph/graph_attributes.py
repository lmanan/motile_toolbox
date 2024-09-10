from enum import Enum


class NodeAttr(Enum):
    """Node attributes that can be added to candidate graph using the toolbox.
    Note: Motile can flexibly support any custom attributes. The toolbox provides
    implementations of commonly used ones, listed here.
    """

    POS = "pos"
    TIME = "time"
    SEG_ID = "seg_id"
    SEG_HYPO = "seg_hypo"
    IGNORE_APPEAR_COST = "ignore_appear_cost"
    IGNORE_DISAPPEAR_COST = "ignore_disappear_cost"


class EdgeAttr(Enum):
    """Edge attributes that can be added to candidate graph using the toolbox.
    Note: Motile can flexibly support any custom attributes. The toolbox provides
    implementations of commonly used ones, listed here.
    """

    IOU = "iou"
