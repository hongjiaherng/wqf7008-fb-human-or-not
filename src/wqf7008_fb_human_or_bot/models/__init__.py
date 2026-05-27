"""Model subpackage. Import directly from submodules to avoid the train <-> base
circular import (`train` imports `BidderClassifier` from `models.base`, so pulling
torch-heavy submodules into this __init__ would create a cycle).
"""
