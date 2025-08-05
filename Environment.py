import random
import numpy as np

## AMBIENTE DE INVENTARIO

class InventoryEnvironment:
    def __init__(self, max_stock=10):
        self.products = ['product_A', 'product_B']
        self.max_stock = max_stock
        self.demand = {p: [0,1,2] for p in self.products}
        self.restock_cost = {'product_A': 5, 'product_B': 7}
        self.sell_price = {'product_A': 10, 'product_B': 15}
        self.state = None

    def reset(self):
        self.state = {p: random.randint(0, self.max_stock) for p in self.products}
        return self.state

    def step(self, action):
        reward = 0
        for p in self.products:
            stock = self.state[p]
            restock = action[p]
            self.state[p] = min(self.max_stock, stock + restock)
            demand = random.choice(self.demand[p])
            sales = min(demand, self.state[p])
            self.state[p] -= sales
            reward += sales * self.sell_price[p] - restock * self.restock_cost[p]
        return self.state, reward
