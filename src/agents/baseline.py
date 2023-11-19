import copy
import datetime
from collections import defaultdict
from typing import DefaultDict, Dict, Tuple

import numpy as np
from interface import Agent
from pyspark.sql import DataFrame

from custom_typings import STOCK_LIST, Action, Portfolio, Price, Stock
from envs.market import StockMarket


class KOSPIFollower(Agent):
    def observe(self, env: StockMarket, timestep: datetime.date) -> DataFrame:
        return env.observed(stocks=["kospi"], start=timestep, end=timestep)

    def struct_state(self, observation: DataFrame) -> Tuple[int, Price]:
        # 1. holding quantity of KOSPI
        holding_quantity = self.portfolio["kospi"]

        # 2. today's KOSPI open price
        open_price = float(observation.first().Open)

        return holding_quantity, open_price

    def policy(self, state: Tuple[int, Price]) -> Action:
        kospi_holding_quantity, kospi_open_price = state
        action = defaultdict(lambda: defaultdict(int))

        if kospi_holding_quantity * kospi_open_price > 10_000_000:
            action["kospi"]["sell"] = action["kospi"]["buy"] = kospi_holding_quantity
        else:
            action["kospi"]["buy"] = int(self.portfolio["capital"] / kospi_open_price)

        return action


class DiversifyingRandomTrader(Agent):
    def __init__(self, upper_bound: float = 0.1, lower_bound: float = 0.1):
        self.portfolio = Portfolio(
            celltrion=0,
            hyundai_motors=0,
            kakao=0,
            kospi=0,
            lg_chem=0,
            lg_hnh=0,
            naver=0,
            samsung_bio=0,
            samsung_elec=0,
            samsung_elec2=0,
            samsung_sdi=0,
            sk_hynix=0,
            capital=100_000_000,
            value=100_000_000,
        )
        self.rewards = []
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.inventory = defaultdict(lambda: defaultdict(int))

    def reset(self):
        self.portfolio = Portfolio(
            celltrion=0,
            hyundai_motors=0,
            kakao=0,
            kospi=0,
            lg_chem=0,
            lg_hnh=0,
            naver=0,
            samsung_bio=0,
            samsung_elec=0,
            samsung_elec2=0,
            samsung_sdi=0,
            sk_hynix=0,
            capital=100_000_000,
            value=100_000_000,
        )
        self.rewards = []
        self.inventory = defaultdict(lambda: defaultdict(int))

    def observe(self, env: StockMarket, timestep: datetime.date) -> DataFrame:
        return env.observed(stocks=list(STOCK_LIST), start=timestep, end=timestep)

    def struct_state(
        self, observation: DataFrame
    ) -> Tuple[Dict[Stock, Price], DefaultDict[Stock, DefaultDict[Price, int]]]:
        # 1. Today's open prices
        open_prices = {
            row.Stock: float(row.Open)
            for row in observation.select("Stock", "Open").collect()
        }

        # 2. inventory

        return open_prices, copy.deepcopy(self.inventory)

    def place_random_order(
        self, open_price_book: Dict[Stock, Price]
    ) -> Tuple[DefaultDict[str, int], Price]:
        subtotal = 0
        order_book = defaultdict(int)
        stocks = list(open_price_book.keys())

        while subtotal < 10_000_000:
            for idx in np.random.choice(len(stocks), len(stocks), replace=False):
                if subtotal >= 10_000_000:
                    break
                pick = stocks[idx]
                subtotal += open_price_book[pick]
                order_book[pick] += 1

        return order_book, subtotal

    def search_most_profitable_stock_in_inventory(
        self,
        open_price_book: Dict[Stock, Price],
        inventory: DefaultDict[Stock, DefaultDict[Price, int]],
    ) -> Tuple[Stock, Price]:
        max_unit_profit = -100_000_000
        most_profitable_stock = ""
        bought_price = 0
        for stock in [
            s
            for s in set(open_price_book.keys()) & set(inventory.keys())
            if len(inventory[s].keys()) > 0
        ]:
            open_price = open_price_book[stock]
            buying_history = inventory[stock]
            min_bought_price = min(buying_history.keys())
            unit_profit = open_price - min_bought_price

            if unit_profit > max_unit_profit:
                most_profitable_stock = stock
                bought_price = min_bought_price

        return most_profitable_stock, bought_price

    def policy(
        self,
        state: Tuple[Dict[Stock, Price], DefaultDict[Stock, DefaultDict[Price, int]]],
    ) -> Action:
        open_price_book, inventory = state
        action = defaultdict(lambda: defaultdict(int))

        # 1. Buy 10_000_000 + alpha
        random_orders, subtotal = self.place_random_order(
            open_price_book=open_price_book
        )
        for stock, quantity in random_orders.items():
            action[stock]["buy"] += quantity

            self.inventory[stock][open_price_book[stock]] += quantity

        # 2. Sell alpha
        while subtotal > self.portfolio["capital"]:
            (
                most_profitable_stock,
                bought_price,
            ) = self.search_most_profitable_stock_in_inventory(
                open_price_book=open_price_book, inventory=inventory
            )

            if not most_profitable_stock:
                print(self.portfolio)
                print(subtotal)
                print(inventory)
                print(action)

            # update action
            action[most_profitable_stock]["sell"] += 1
            # update subtotal
            subtotal -= open_price_book[most_profitable_stock]
            # update inventory
            buying_history = inventory[most_profitable_stock]
            if buying_history[bought_price] > 1:
                inventory[most_profitable_stock][bought_price] -= 1
                self.inventory[most_profitable_stock][bought_price] -= 1
            elif len(buying_history.keys()) > 1:
                del inventory[most_profitable_stock][bought_price]
                del self.inventory[most_profitable_stock][bought_price]
            else:
                del inventory[most_profitable_stock]
                del self.inventory[most_profitable_stock]

        # 3. Sell off-boundaries
        off_boundaries = []
        for stock in [s for s in set(open_price_book.keys()) & set(inventory.keys())]:
            open_price = open_price_book[stock]
            buying_history = inventory[stock]
            for bought_price, quantity in buying_history.items():
                stop_loss = open_price < bought_price * (1 - self.lower_bound)
                confirm_profit = open_price > bought_price * (1 + self.upper_bound)
                if stop_loss or confirm_profit:
                    off_boundaries.append((stock, bought_price, quantity))

        for stock, bought_price, quantity in off_boundaries:
            # update action
            action[stock]["sell"] += quantity
            # update inventory
            buying_history = inventory[stock]
            if len(buying_history.keys()) > 1:
                del self.inventory[stock][bought_price]
                del inventory[stock][bought_price]
            else:
                del self.inventory[stock]
                del inventory[stock]

        return action
