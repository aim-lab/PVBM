def far(i,j,set_):
    for element in set_:
        if ((i-element[0])**2 + (j-element[1])**2)**0.5 < 20:
            return False
    return True