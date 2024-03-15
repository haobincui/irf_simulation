import numpy as np


def compare_results(B1: dict, B2: dict) -> bool:
    # length of B1 and B2 should be the same
    if len(B1) != len(B2):
        print(f'length are not equal, ')
        return False

    for i in range(len(B1)):
        keys1 = B1[i].keys()
        keys2 = B2[i].keys()

        # keys should be the same
        if not keys1 == keys2:
            print('keys are not equal')
            return False

        # values for each key should be the same
        for key in keys1:
            if not np.array_equal(B1[i][key], B2[i][key]):
                print(f'values are not equal [{i}]')
                return False

    # if all checks pass, return True
    return True



