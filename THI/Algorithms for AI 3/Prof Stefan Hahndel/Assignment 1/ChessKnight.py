deltaPos = [[-1, 2], [1, 2], [2, 1], [2, -1], [1, -2], [-1, -2], [-2, -1], [-2, 1]]

def addPos(pos1: list, pos2: list) -> list:
    return [pos1[0] + pos2[0], pos1[1] + pos2[1]]

board_size = 8
max_step_limit = ((2 * board_size) / 3) + 1

def moveKnight(currSteps: int, currPath: list, currPos: list, targetPos: list) -> None:
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
            # print(minSteps, minPath)
            return
    
    # spawn new recursion step branch
    else:
        for i in deltaPos:
            nextPos = addPos(pos, i)
            # check for valid pos that is not out of chess board else return
            if(not(0 < nextPos[0] < 9 and 0 < nextPos[1] < 9)): continue
            elif(minSteps != None and steps >= minSteps): continue
            nextPath = path.copy() + [nextPos]
            moveKnight(steps + 1, nextPath, nextPos, targetPos)

while(True):
    curr_pos = eval(str(input("Start Pos (format [x <int>, y <int>]): ")))
    targetPos = eval(str(input("Target Pos (format [x <int>, y <int>]): ")))

    if(1 <= curr_pos[0] <= 8 and 1 <= curr_pos[1] <= 8 and 1 <= targetPos[0] <= 8 and 1 <= targetPos[1] <= 8):
        pass
    elif(type(curr_pos) != list or type(targetPos) != list or len(curr_pos) != len(targetPos) != 2 ):
        print("Invalid board locations entered")
    else:
        print("Invalid board locations entered")
        continue

    minSteps = max_step_limit
    minPath = None
    moveKnight(0, [curr_pos], curr_pos, targetPos)
    if(minPath != None):
        print(f"{targetPos} can be reached from {curr_pos} in {minSteps} steps\npath: {minPath}", "\n\n\n")
    else:
        print(f"Path from {curr_pos} to {targetPos} cannot be found.")

    end = False
    while True:
        state = input("End program?? (y/n)")
        if(state.lower() == "y"):
            end = True
            break
        elif(state.lower() == "n"):
            break
        else:
            print("invalid option")
            continue
    
    if(end):
        break