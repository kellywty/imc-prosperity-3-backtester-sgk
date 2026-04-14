import json
from typing import Any, Optional

# from data import LIMITS
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

LIMITS = {
    "EMERALDS": 50,
    "TOMATOES": 50,
}

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[listing.symbol, listing.product, listing.denomination] for listing in listings.values()]

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {
            symbol: [order_depth.buy_orders, order_depth.sell_orders]
            for symbol, order_depth in order_depths.items()
        }

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [trade.symbol, trade.price, trade.quantity, trade.buyer, trade.seller, trade.timestamp]
                )
        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()


class Trader:
    EMERALDS_DEFAULT_FAIR_VALUE = 10000
    EMERALDS_POSITION_STEP = 8
    EMERALDS_PASSIVE_SIZE = 10
    EMERALDS_TAKE_SIZE = 16

    def get_best_bid(self, order_depth: OrderDepth) -> Optional[int]:
        return max(order_depth.buy_orders) if order_depth.buy_orders else None

    def get_best_ask(self, order_depth: OrderDepth) -> Optional[int]:
        return min(order_depth.sell_orders) if order_depth.sell_orders else None

    def get_mid_price(self, order_depth: OrderDepth) -> int:
        best_bid = self.get_best_bid(order_depth)
        best_ask = self.get_best_ask(order_depth)

        if best_bid is not None and best_ask is not None:
            return round((best_bid + best_ask) / 2)
        if best_bid is not None:
            return best_bid
        if best_ask is not None:
            return best_ask

        return self.EMERALDS_DEFAULT_FAIR_VALUE

    def trade_tomatoes(self, order_depth: OrderDepth, position: int, limit: int) -> list[Order]:
        orders: list[Order] = []
        best_bid = self.get_best_bid(order_depth)

        if best_bid is None:
            return orders

        sell_capacity = max(0, limit + position)
        if sell_capacity == 0:
            return orders

        bid_volume = order_depth.buy_orders[best_bid]
        sell_quantity = min(bid_volume, sell_capacity)
        if sell_quantity > 0:
            logger.print("TOMATOES SELL", sell_quantity, "@", best_bid)
            orders.append(Order("TOMATOES", best_bid, -sell_quantity))

        return orders

    def trade_emeralds(self, order_depth: OrderDepth, position: int, limit: int) -> list[Order]:
        orders: list[Order] = []
        best_bid = self.get_best_bid(order_depth)
        best_ask = self.get_best_ask(order_depth)
        fair_value = self.get_mid_price(order_depth)

        buy_capacity = max(0, limit - position)
        sell_capacity = max(0, limit + position)

        inventory_skew = position // self.EMERALDS_POSITION_STEP
        passive_bid = fair_value - 1 - inventory_skew
        passive_ask = fair_value + 1 - inventory_skew

        logger.print(
            f"EMERALDS pos={position} fair={fair_value} best_bid={best_bid} best_ask={best_ask} "
            f"passive_bid={passive_bid} passive_ask={passive_ask}"
        )

        if best_ask is not None and buy_capacity > 0 and best_ask <= fair_value - 1:
            ask_volume = -order_depth.sell_orders[best_ask]
            buy_quantity = min(ask_volume, buy_capacity, self.EMERALDS_TAKE_SIZE)
            if buy_quantity > 0:
                logger.print("EMERALDS BUY", buy_quantity, "@", best_ask)
                orders.append(Order("EMERALDS", best_ask, buy_quantity))
                buy_capacity -= buy_quantity

        if best_bid is not None and sell_capacity > 0 and best_bid >= fair_value + 1:
            bid_volume = order_depth.buy_orders[best_bid]
            sell_quantity = min(bid_volume, sell_capacity, self.EMERALDS_TAKE_SIZE)
            if sell_quantity > 0:
                logger.print("EMERALDS SELL", sell_quantity, "@", best_bid)
                orders.append(Order("EMERALDS", best_bid, -sell_quantity))
                sell_capacity -= sell_quantity

        if best_bid is not None and best_ask is not None and buy_capacity > 0 and passive_bid > best_bid and passive_bid < best_ask:
            buy_quantity = min(self.EMERALDS_PASSIVE_SIZE, buy_capacity)
            if buy_quantity > 0:
                logger.print("EMERALDS BID", buy_quantity, "@", passive_bid)
                orders.append(Order("EMERALDS", passive_bid, buy_quantity))

        if best_bid is not None and best_ask is not None and sell_capacity > 0 and passive_ask < best_ask and passive_ask > best_bid:
            sell_quantity = min(self.EMERALDS_PASSIVE_SIZE, sell_capacity)
            if sell_quantity > 0:
                logger.print("EMERALDS ASK", sell_quantity, "@", passive_ask)
                orders.append(Order("EMERALDS", passive_ask, -sell_quantity))

        return orders

    def run(self, state: TradingState):
        result: dict[Symbol, list[Order]] = {}

        for product, order_depth in state.order_depths.items():
            position = state.position.get(product, 0)
            limit = LIMITS.get(product, 0)

            if product == "TOMATOES":
                result[product] = self.trade_tomatoes(order_depth, position, limit)
            elif product == "EMERALDS":
                result[product] = self.trade_emeralds(order_depth, position, limit)
            else:
                result[product] = []

        conversions = 0
        trader_data = ""
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
