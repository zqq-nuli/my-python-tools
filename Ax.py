from queue import PriorityQueue

# 定义一个类来表示带权重的网格地图
class GridWithWeights:
    def __init__(self, width, height):
        self.width = width  # 网格的宽度
        self.height = height  # 网格的高度
        self.walls = []  # 存储障碍物的位置
        self.weights = {}  # 存储每个节点的权重

    # 判断一个坐标是否在网格范围内
    def in_bounds(self, id):
        (x, y) = id
        return 0 <= x < self.width and 0 <= y < self.height

    # 判断一个坐标是否是可通过的（不是障碍物）
    def passable(self, id):
        return id not in self.walls

    # 获取一个节点的邻居节点
    def neighbors(self, id):
        (x, y) = id
        results = [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]  # 右、左、上、下四个方向
        results = filter(self.in_bounds, results)  # 过滤掉不在网格范围内的坐标
        results = filter(self.passable, results)  # 过滤掉障碍物坐标
        return results

    # 获取从一个节点到另一个节点的移动成本
    def cost(self, from_node, to_node):
        return self.weights.get(to_node, 1)  # 默认移动成本为1

# 启发式函数，使用曼哈顿距离
def heuristic(a, b):
    (x1, y1) = a
    (x2, y2) = b
    return abs(x1 - x2) + abs(y1 - y2)

# A*算法的实现
def a_star_search(graph, start, goal):
    frontier = PriorityQueue()  # 优先队列用于存储待处理节点
    frontier.put(start, 0)  # 将起点加入队列，优先级为0
    came_from = {}  # 存储每个节点的前一个节点
    cost_so_far = {}  # 存储从起点到每个节点的成本

    came_from[start] = None  # 起点没有前驱节点
    cost_so_far[start] = 0  # 起点到自身的成本为0

    while not frontier.empty():
        current = frontier.get()  # 取出优先级最高的节点

        if current == goal:
            break  # 如果当前节点是目标节点，则退出

        for next in graph.neighbors(current):
            new_cost = cost_so_far[current] + graph.cost(current, next)  # 计算到下一个节点的成本
            if next not in cost_so_far or new_cost < cost_so_far[next]:
                cost_so_far[next] = new_cost  # 更新下一个节点的成本
                priority = new_cost + heuristic(goal, next)  # 计算优先级
                frontier.put(next, priority)  # 将下一个节点加入队列
                came_from[next] = current  # 记录下一个节点的前驱节点

    return came_from, cost_so_far  # 返回前驱节点和成本信息

# 根据前驱节点重建路径
def reconstruct_path(came_from, start, goal):
    current = goal
    path = []
    while current != start:
        path.append(current)
        current = came_from[current]  # 追溯前驱节点
    path.append(start)
    path.reverse()  # 反转路径使其从起点到终点
    return path

# 将路径转换为方向
def path_to_directions(path):
    directions = []
    for i in range(1, len(path)):
        (x1, y1) = path[i-1]
        (x2, y2) = path[i]
        if x2 == x1 + 1:
            directions.append("向右")
        elif x2 == x1 - 1:
            directions.append("向左")
        elif y2 == y1 + 1:
            directions.append("向上")
        elif y2 == y1 - 1:
            directions.append("向下")
    return directions


# 示例使用:
width, height = 10, 10  # 定义网格大小
grid = GridWithWeights(width, height)  # 创建网格对象
start = (1, 4)  # 定义起点
goal = (7, 8)  # 定义终点

# 定义障碍物位置
grid.walls = [(3, 4), (3, 5), (3, 6), (3, 7), (4, 7)]
# grid.walls = []

# 运行A*算法
came_from, cost_so_far = a_star_search(grid, start, goal)
# 重建路径
path = reconstruct_path(came_from, start, goal)

# 将路径转换为方向
directions = path_to_directions(path)

# 输出路径和方向
print("Path:", path)
print("Directions:", directions)
