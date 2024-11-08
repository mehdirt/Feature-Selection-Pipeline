from functools import reduce

def extract_metas(metas_dict, base_metas):
    """Return the final metabolites."""
    
    differences = []

    for metas in metas_dict.values():
        differences.append(metas.difference(base_metas))

    SEF = reduce(set.intersection, differences)
    SBP = base_metas.union(SEF)

    # print(metas_dict)
    # print("Output Metabolites:\n", SBP)