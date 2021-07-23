from collections import defaultdict
from typing import DefaultDict, Tuple

import numpy as np
from typings import OpenPriceBook, PortfolioHistory

from agents.agent import Agent


class DiversifyingRandomInvestor(Agent):
    def __init__(self, percentage_bound: int = 10):
        super().__init__()
        self.percentage_bound = percentage_bound
        self.portfolio_history = PortfolioHistory()

    def sell(self, stock: str, quantity: int, bought_price: int, selling_price: int):
        quantity_now = getattr(self.portfolio, stock)
        assert quantity_now >= quantity

        # portfolio
        setattr(self.portfolio, stock, quantity_now - quantity)
        self.portfolio.capital += quantity * selling_price
        # history
        getattr(self.portfolio_history, stock)[bought_price] -= quantity

    def buy(self, stock: str, quantity: int, buying_price: int):
        assert self.portfolio.capital >= quantity * buying_price

        # portfolio
        self.portfolio.capital -= quantity * buying_price
        setattr(self.portfolio, stock, getattr(self.portfolio, stock) + quantity)
        # history
        getattr(self.portfolio_history, stock)[buying_price] += quantity

    def random_pick(
        self, open_prices: OpenPriceBook
    ) -> Tuple[DefaultDict[str, int], int]:
        subtotal = 0
        orders = defaultdict(int)
        stocks = [attr for attr in dir(open_prices) if not attr.startswith("_")]

        while subtotal < 10_000_000:
            for idx in np.random.choice(len(stocks), len(stocks), replace=False):
                if subtotal >= 10_000_000:
                    break
                pick = stocks[idx]
                subtotal += getattr(open_prices, pick)
                orders[pick] += 1

        return orders, subtotal

    def search_most_profitable_stock_in_hand(
        self, open_prices: OpenPriceBook
    ) -> Tuple[str, int]:
        max_unit_profit = -100_000_000
        stocks_in_hand = [
            attr
            for attr in dir(self.portfolio)
            if not attr.startswith("_") and getattr(self.portfolio, attr) > 0
        ]
        most_unit_profitable_stock = stocks_in_hand[
            np.random.randint(0, len(stocks_in_hand))
        ]
        bought_price = -1
        for stock in stocks_in_hand:
            buying_history = getattr(self.portfolio_history, stock)
            for _bought_price in buying_history.keys():
                unit_profit = getattr(open_prices, stock) - _bought_price
                if unit_profit > max_unit_profit:
                    max_unit_profit = unit_profit
                    most_unit_profitable_stock = stock
                    bought_price = _bought_price

        return most_unit_profitable_stock, bought_price

    def action(self, open_prices: OpenPriceBook) -> int:
        stocks = [
            attr for attr in dir(self.portfolio_history) if not attr.startswith("_")
        ]
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
                    self.sell(
                        stock=stock,
                        quantity=buying_history[bought_price],
                        bought_price=bought_price,
                        selling_price=open_price,
                    )
        # 2. Buy
        random_orders, subtotal = self.random_pick(open_prices=open_prices)

        if self.portfolio.capital >= subtotal:
            for s, q in random_orders.items():
                self.buy(stock=s, quantity=q, buying_price=getattr(open_prices, s))
        else:
            portfolio_value = self.portfolio.capital
            for stock in stocks:
                portfolio_value += getattr(self.portfolio, stock) * getattr(
                    open_prices, stock
                )

            assert portfolio_value >= subtotal

            while self.portfolio.capital < subtotal:
                (
                    most_unit_profitable_stock,
                    bought_price,
                ) = self.search_most_profitable_stock_in_hand(open_prices=open_prices)

                if bought_price != -1:
                    self.sell(
                        stock=most_unit_profitable_stock,
                        quantity=1,
                        bought_price=bought_price,
                        selling_price=getattr(open_prices, most_unit_profitable_stock),
                    )
                else:
                    bought_prices = list(
                        getattr(
                            self.portfolio_history, most_unit_profitable_stock
                        ).keys()
                    )
                    random_bought_price = bought_prices[
                        np.random.randint(0, len(bought_price))
                    ]
                    self.sell(
                        stock=most_unit_profitable_stock,
                        quantity=1,
                        bought_price=random_bought_price,
                        selling_price=getattr(open_prices, most_unit_profitable_stock),
                    )

            for s, q in random_orders.items():
                self.buy(stock=s, quantity=q, buying_price=getattr(open_prices, s))


class KOSPIFollower(Agent):
    def sell(self, quantity: int, price: int):
        assert self.portfolio.kospi >= quantity

        # portfolio
        self.portfolio.kospi -= quantity
        self.portfolio.capital += quantity * price

    def buy(self, quantity: int, price: int):
        assert self.portfolio.capital >= quantity * price

        # portfolio
        self.portfolio.capital -= quantity * price
        self.portfolio.kospi += quantity

    def action(self, kospi_open_price: int) -> int:
        # 풀매도
        self.sell(quantity=self.portfolio.kospi, price=kospi_open_price)

        # 다시 풀매수
        self.buy(
            quantity=int(self.portfolio.capital / kospi_open_price),
            price=kospi_open_price,
        )
