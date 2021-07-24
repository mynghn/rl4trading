import copy
import datetime
from collections import defaultdict
from typing import DefaultDict, Tuple

import numpy as np
from custom_typings import STOCK_LIST, Action, Inventory, OpenPriceBook, Order, Price
from environment.market import StockMarket
from interface import Agent
from pyspark.sql import DataFrame


class KOSPIFollower(Agent):
    def observe(self, env: StockMarket, timestep: datetime.date) -> DataFrame:
        return env.observed(stocks=["kospi"], start=timestep, end=timestep)

    def observation2state(self, observation: DataFrame) -> Price:
        return observation.first().Open

    def policy(self, state: Price) -> Action:
        kospi_open_price = state
        portfolio_value = (
            self.portfolio.capital + self.portfolio.kospi * kospi_open_price
        )

        return Action(
            kospi=Order(
                buy=int(portfolio_value / kospi_open_price),
                sell=self.portfolio.kopsi,
            ),
        )


class DiversifyingRandomInvestor(Agent):
    def __init__(self, percentage_bound: int = 10):
        super().__init__()
        self.percentage_bound = percentage_bound
        self.inventory = Inventory()

    def observe(self, env: StockMarket, timestep: datetime.date) -> DataFrame:
        return env.observed(stocks=STOCK_LIST, start=timestep, end=timestep)

    def observation2state(self, observation: DataFrame) -> OpenPriceBook:
        return OpenPriceBook(
            **{
                row.Stock: row.Open
                for row in observation.select("Stock", "Open").collect()
            }
        )

    def random_pick(
        self, open_prices: OpenPriceBook
    ) -> Tuple[DefaultDict[str, int], int]:
        subtotal = 0
        order_book = defaultdict(int)
        stocks = [attr for attr in dir(open_prices) if not attr.startswith("_")]

        while subtotal < 10_000_000:
            for idx in np.random.choice(len(stocks), len(stocks), replace=False):
                if subtotal >= 10_000_000:
                    break
                pick = stocks[idx]
                subtotal += getattr(open_prices, pick)
                order_book[pick] += 1

        return order_book, subtotal

    def search_most_profitable_stock_in_inventory(
        self, open_prices: OpenPriceBook, inventory: Inventory
    ) -> str:
        stocks_in_hand = [
            attr
            for attr in dir(self.portfolio)
            if not attr.startswith("_")
            and getattr(self.portfolio, attr) > 0
            and attr != "capital"
        ]

        most_unit_profitable_stock = stocks_in_hand[
            np.random.randint(0, len(stocks_in_hand))
        ]
        max_unit_profit = -100_000_000
        for stock in stocks_in_hand:
            min_bought_price = min(getattr(inventory, stock))
            unit_profit = getattr(open_prices, stock) - min_bought_price
            if unit_profit > max_unit_profit:
                max_unit_profit = unit_profit
                most_unit_profitable_stock = stock

        return most_unit_profitable_stock

    def policy(self, state: OpenPriceBook) -> int:
        open_prices = state
        stocks = [attr for attr in dir(self.portfolio) if not attr.startswith("_")]
        action = Action()

        # 1. Sell
        for stock in stocks:
            buying_history = getattr(self.portfolio_history, stock)
            open_price = getattr(open_prices, stock)
            for bought_price in buying_history.keys():
                stop_loss = open_price < bought_price * (
                    1 - self.percentage_bound / 100
                )
                confirm_profit = open_price > bought_price * (
                    1 + self.percentage_bound / 100
                )
                if stop_loss or confirm_profit:
                    order = getattr(action, stock)
                    curr = getattr(order, "sell")
                    setattr(order, "sell", curr + buying_history[bought_price])

        # 2. Buy
        random_orders, subtotal = self.random_pick(open_prices=open_prices)

        if self.portfolio.capital >= subtotal:
            for stock, quantity in random_orders.items():
                order = getattr(action, stock)
                curr = getattr(order, "buy")
                setattr(order, "buy", curr + quantity)
        else:
            portfolio_value = self.portfolio.capital
            for stock in stocks:
                portfolio_value += getattr(self.portfolio, stock) * getattr(
                    open_prices, stock
                )

            assert portfolio_value >= subtotal

            curr_capital = self.portfolio.capital
            curr_inventory = copy.deepcopy(self.inventory)
            while curr_capital < subtotal:
                most_unit_profitable_stock = (
                    self.search_most_profitable_stock_in_inventory(
                        open_prices=open_prices, inventory=curr_inventory
                    )
                )

                order = getattr(action, most_unit_profitable_stock)
                curr = getattr(order, "sell")
                setattr(order, "sell", curr + 1)

                curr_capital += getattr(open_prices, most_unit_profitable_stock)
                stock_inventory = getattr(curr_inventory, most_unit_profitable_stock)
                stock_inventory.remove(min(stock_inventory))

            for stock, quantity in random_orders.items():
                order = getattr(action, stock)
                curr = getattr(order, "buy")
                setattr(order, "buy", curr + quantity)

        return action
