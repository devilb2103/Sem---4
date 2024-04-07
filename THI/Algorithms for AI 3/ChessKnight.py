currPos = [4, 4]
targetPos = [8, 8]
deltaPos = [[-1, 2], [1, 2], [2, 1], [2, -1], [1, -2], [-1, -2], [-2, -1], [-2, 1]]

minSteps = None
minPath = None

def addPos(pos1: list, pos2: list) -> list:
    return [pos1[0] + pos2[0], pos1[1] + pos2[1]]

def moveKnight(currSteps: int, currPath: list, currPos: list) -> None:
    global minSteps, minPath
    
    path = currPath
    pos = currPos
    steps = currSteps
    # print(type(steps), type(path), type(pos))
    # return
    # stop recursion branch if exceeded current best record already
    if(minSteps != None and currSteps >= minSteps):
        return
    
    # if target is reached then check for new step record and overwrite old step record
    elif(pos == targetPos):
        if(minSteps == None or steps < minSteps):
            minSteps = steps
            minPath = path
            return
    
    # spawn new step branch
    else:
        for i in deltaPos:
            nextPos = addPos(pos, i)
            # check for valid pos not out of board else return
            if(not(0 < nextPos[0] < 9 and 0 < nextPos[1] < 9)): continue
            # print(steps + 1, path, nextPos)
            path.append(nextPos)
            nextPath = path
            moveKnight(steps + 1, nextPath, nextPos)


stepInit = 0
pathInit = list([currPos])
posInit = currPos
moveKnight(0, pathInit, posInit)

print(minSteps, minPath)
print(len(minPath))