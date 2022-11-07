import numpy as np
import tkinter as tk
import random
import time
from PIL import Image, ImageTk

#二維迷宮  -1為起點 0為路 1為障礙 2為終點 
maze = np.array([
  [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1],
  [1, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
  [1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1],
  [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
  [1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1],
  [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1],
  [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1],
  [1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1],
  [1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1],
  [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1],
  [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 1],
  [1, 0, 0, 0, 0, 0, 0, 2, 1, 0, 1, 0, 1],
  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
])
#繪製迷宮遊戲
class MazeWindow:
    def __init__(self, maze):
        self.root = tk.Tk()
        self.root.title('Maze Q-learning')
        self.maze = maze
        self.labels = np.zeros(self.maze.shape).tolist()
        self.plotBackground()
    #繪製
    def plotBackground(self):
        for i, row in enumerate(self.maze.tolist()):
            for j, element in enumerate(row):
                bg = 'black' if element == 1 else 'red' if element == 2 else 'blue' if element == -1 else 'white'
                self.labels[i][j] = tk.Label(self.root, foreground='blue', background=bg, width=2, height=1, relief='ridge', font='? 10 bold')
                self.labels[i][j].grid(row=i, column=j)
    #延遲
    def mainloop(self, func):
        self.root.after(1000, func)
        self.root.mainloop()
        
    #玩家
    def target(self, indexes):
        for label in [item for row in self.labels for item in row]:
            label.config(text='')
        self.labels[indexes[0]][indexes[1]].config(text = 'Q')

        self.root.update()
#代理人
class Agent:
    def __init__(self, maze, initState):
        self.state = initState
        self.maze = maze
        self.initQTable()
        self.actionList = ['up', 'down', 'left', 'right']
        self.actionDict = {element : index for index, element in enumerate(self.actionList)}
        #定義Q Table
    def initQTable(self):
        Q = np.zeros(self.maze.shape).tolist()
        for i, row in enumerate(Q):
            for j, _ in enumerate(row):
                Q[i][j] = [0, 0, 0, 0] #上,下,左,右
        self.QTable = np.array(Q, dtype='f')
    
    def showQTable(self):
        for i, row in enumerate(self.QTable):
            for j, element in enumerate(row):
                print(f'({i}, {j}){element}')

    def showBestAction(self):
        for i, row in enumerate(self.QTable):
            for j, element in enumerate(row):
                Qa = element.tolist()
                action = self.actionList[Qa.index(max(Qa))] if max(Qa) != 0 else '??'
                print(f'({i}, {j}){action}', end=" ")
            print()
    #eGreddy = 貪婪策略  讓代理人有機率不使用Q Table而是隨機執行行為，防止代理人只會循環某些策略
    def getAction(self, eGreddy=0.8):
        if random.random() > eGreddy:
            return random.choice(self.actionList)
        else:
            Qsa = self.QTable[self.state].tolist()
            return self.actionList[Qsa.index(max(Qsa))]
    
    def getNextMaxQ(self, state):
        return max(np.array(self.QTable[state]))
    #更新Q Table , lr = 學習率 , gamma = 折扣因子 
    def updateQTable(self, action, nextState, reward, lr=0.7, gamma=0.9):
        Qs = self.QTable[self.state]
        Qsa = Qs[self.actionDict[action]]
        Qs[self.actionDict[action]] = (1 - lr) * Qsa + lr * (reward + gamma *(self.getNextMaxQ(nextState)))
#建立遊戲環境
class Environment:
    def __init__(self):
        pass
    #定義動作
    def getNextState(self, state, action):
        row = state[0]
        column = state[1]
        if action == 'up':
            row -= 1
        elif action == 'down':
            row += 1
        elif action == 'left':
            column -= 1
        elif action == 'right':
            column += 1
        nextState = (row, column)
        try:
            #碰到邊界或是牆壁
            if row < 0 or column < 0 or maze[row, column] == 1:
                return [state, False]
            #抵達終點
            elif maze[row, column] == 2:
                return [nextState, True]
            #普通前進
            else:
                return [nextState, False]
        except IndexError as e:
            #碰到邊界
            return [state, False]

    #定義獎勵
    def doAction(self, state, action):
        nextState, result = self.getNextState(state, action)
        
        #下個的狀態等於當前狀態(未移動)
        if nextState == state:
            reward = -20
        #抵達終點
        elif result:
            reward = 5000
        #有移動但還未抵達終點
        else:
            reward = -2
        return [reward, nextState, result]
    
def main():    
    initState = (np.where(maze==-1)[0][0], np.where(maze==-1)[1][0])
    #創建代理人
    agent = Agent(maze, initState)
    #創建遊戲環境
    environment = Environment()
    for j in range(0, 1000):
        agent.state = initState
        m.target(agent.state)
        time.sleep(0.1)
        i = 0
        while True:
            i += 1
            # 代理人執行下一步
            action = agent.getAction(0.9)
            # Give the action to the Environment to execute
            reward, nextState, result = environment.doAction(agent.state, action)
            # 更新Q Table
            agent.updateQTable(action, nextState, reward)
            # 代理人改變狀態
            agent.state = nextState
            m.target(agent.state)
            if result:
                print(f' {j+1:2d} : {i} 步到達終點.')
                break
    agent.showQTable()
    agent.showBestAction()
m = MazeWindow(maze)
m.mainloop(main)
