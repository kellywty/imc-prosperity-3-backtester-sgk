import json
from typing import Any, Optional

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


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
    ASH_COATED_OSMIUM_LIMIT = 80
    ASH_COATED_OSMIUM_DEFAULT_FAIR_VALUE = 10000

    ASH_COATED_OSMIUM_POSITION_STEP = 20

    ASH_COATED_OSMIUM_WIDE_OFFSET = 7
    ASH_COATED_OSMIUM_WIDE_SIZE = 30

    ASH_COATED_OSMIUM_TAKE_SIZE = 15
    ASH_COATED_OSMIUM_BASE_BUY_EDGE = 3
    ASH_COATED_OSMIUM_BASE_SELL_EDGE = 3

    ASH_COATED_OSMIUM_FAST_FAIR_WINDOW = 20
    ASH_COATED_OSMIUM_SLOW_FAIR_WINDOW = 80
    ASH_COATED_OSMIUM_BUY_SLOW_BUFFER = -1
    ASH_COATED_OSMIUM_SELL_SLOW_BUFFER = 0

    INTARIAN_PEPPER_ROOT_LIMIT = 80
    INTARIAN_PEPPER_ROOT_DEFAULT_FAIR_VALUE = 12000
    INTARIAN_PEPPER_ROOT_WINDOW = 100
    INTARIAN_PEPPER_ROOT_TREND_THRESHOLD = 6
    INTARIAN_PEPPER_ROOT_TAKE_SIZE = 10
    INTARIAN_PEPPER_ROOT_EXIT_BUFFER = 2

    def get_best_bid(self, order_depth: OrderDepth) -> Optional[int]:
        return max(order_depth.buy_orders) if order_depth.buy_orders else None

    def get_best_ask(self, order_depth: OrderDepth) -> Optional[int]:
        return min(order_depth.sell_orders) if order_depth.sell_orders else None

    def get_mid_price(self, order_depth: OrderDepth, default_fair_value: int) -> int:
        best_bid = self.get_best_bid(order_depth)
        best_ask = self.get_best_ask(order_depth)

        if best_bid is not None and best_ask is not None:
            return round((best_bid + best_ask) / 2)
        if best_bid is not None:
            return best_bid
        if best_ask is not None:
            return best_ask
        return default_fair_value

    def update_history(
        self,
        trader_data: dict,
        product: str,
        timestamp: int,
        mid_price: int,
        window: int,
    ) -> None:
        if "history" not in trader_data:
            trader_data["history"] = {}
        if product not in trader_data["history"]:
            trader_data["history"][product] = {"t": [], "p": []}

        trader_data["history"][product]["t"].append(timestamp)
        trader_data["history"][product]["p"].append(mid_price)

        trader_data["history"][product]["t"] = trader_data["history"][product]["t"][-window:]
        trader_data["history"][product]["p"] = trader_data["history"][product]["p"][-window:]

    def get_ash_fast_slow_fair_values(self, trader_data: dict) -> tuple[int, int]:
        history = trader_data.get("history", {}).get("ASH_COATED_OSMIUM", {})
        prices = history.get("p", [])

        if not prices:
            base = self.ASH_COATED_OSMIUM_DEFAULT_FAIR_VALUE
            return base, base

        fast_prices = prices[-self.ASH_COATED_OSMIUM_FAST_FAIR_WINDOW:]
        slow_prices = prices[-self.ASH_COATED_OSMIUM_SLOW_FAIR_WINDOW:]

        fast_fair = round(sum(fast_prices) / len(fast_prices))
        slow_fair = round(sum(slow_prices) / len(slow_prices))
        return fast_fair, slow_fair

    def trade_intarian_pepper_root(
        self,
        order_depth: OrderDepth,
        position: int,
        limit: int,
        timestamp: int,
        trader_data: dict,
    ) -> list[Order]:
        orders: list[Order] = []

        buy_capacity = max(0, limit - position)

        if order_depth.sell_orders and buy_capacity > 0:
            worst_ask = max(order_depth.sell_orders)
            orders.append(Order("INTARIAN_PEPPER_ROOT", worst_ask, buy_capacity))

        return orders

    def trade_ash_coated_osmium(
        self,
        order_depth: OrderDepth,
        position: int,
        limit: int,
        trader_data: dict,
    ) -> list[Order]:
        orders: list[Order] = []
        product = "ASH_COATED_OSMIUM"

        best_bid = self.get_best_bid(order_depth)
        best_ask = self.get_best_ask(order_depth)

        fast_fair, slow_fair = self.get_ash_fast_slow_fair_values(trader_data)
        fair_value = fast_fair

        buy_capacity = max(0, limit - position)
        sell_capacity = max(0, limit + position)

        inventory_skew = int(position / self.ASH_COATED_OSMIUM_POSITION_STEP)

        wide_bid = fair_value - self.ASH_COATED_OSMIUM_WIDE_OFFSET - inventory_skew
        wide_ask = fair_value + self.ASH_COATED_OSMIUM_WIDE_OFFSET - inventory_skew

        buy_edge = self.ASH_COATED_OSMIUM_BASE_BUY_EDGE
        sell_edge = self.ASH_COATED_OSMIUM_BASE_SELL_EDGE

        if position > 40:
            buy_edge = 4
            sell_edge = 1
        elif position < -40:
            buy_edge = 1
            sell_edge = 4

        if (
            best_ask is not None
            and buy_capacity > 0
            and best_ask <= fast_fair - buy_edge
            and best_ask <= slow_fair + self.ASH_COATED_OSMIUM_BUY_SLOW_BUFFER
        ):
            ask_volume = -order_depth.sell_orders[best_ask]
            qty = min(ask_volume, buy_capacity, self.ASH_COATED_OSMIUM_TAKE_SIZE)
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
                buy_capacity -= qty

        if (
            best_bid is not None
            and sell_capacity > 0
            and best_bid >= fast_fair + sell_edge
            and best_bid >= slow_fair - self.ASH_COATED_OSMIUM_SELL_SLOW_BUFFER
        ):
            bid_volume = order_depth.buy_orders[best_bid]
            qty = min(bid_volume, sell_capacity, self.ASH_COATED_OSMIUM_TAKE_SIZE)
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
                sell_capacity -= qty

        wide_buy_size = self.ASH_COATED_OSMIUM_WIDE_SIZE
        wide_sell_size = self.ASH_COATED_OSMIUM_WIDE_SIZE

        if position > 60:
            wide_buy_size = 0
        elif position < -60:
            wide_sell_size = 0

        if buy_capacity > 0 and wide_buy_size > 0:
            qty = min(wide_buy_size, buy_capacity)
            if qty > 0:
                orders.append(Order(product, wide_bid, qty))
                buy_capacity -= qty

        if sell_capacity > 0 and wide_sell_size > 0:
            qty = min(wide_sell_size, sell_capacity)
            if qty > 0:
                orders.append(Order(product, wide_ask, -qty))

        return orders

    def run(self, state: TradingState):
        result: dict[Symbol, list[Order]] = {}

        if state.traderData:
            try:
                trader_data = json.loads(state.traderData)
            except Exception:
                trader_data = {}
        else:
            trader_data = {}

        for product, order_depth in state.order_depths.items():
            position = state.position.get(product, 0)

            if product == "INTARIAN_PEPPER_ROOT":
                limit = self.INTARIAN_PEPPER_ROOT_LIMIT
            elif product == "ASH_COATED_OSMIUM":
                limit = self.ASH_COATED_OSMIUM_LIMIT
            else:
                limit = 0

            if product == "ASH_COATED_OSMIUM":
                mid_price = self.get_mid_price(order_depth, self.ASH_COATED_OSMIUM_DEFAULT_FAIR_VALUE)
                self.update_history(
                    trader_data,
                    product,
                    state.timestamp,
                    mid_price,
                    self.ASH_COATED_OSMIUM_SLOW_FAIR_WINDOW,
                )

            if product == "INTARIAN_PEPPER_ROOT":
                result[product] = self.trade_intarian_pepper_root(
                    order_depth,
                    position,
                    limit,
                    state.timestamp,
                    trader_data,
                )
            elif product == "ASH_COATED_OSMIUM":
                result[product] = self.trade_ash_coated_osmium(
                    order_depth,
                    position,
                    limit,
                    trader_data,
                )
            else:
                result[product] = []

        conversions = 0
        trader_data_str = json.dumps(trader_data)
        logger.flush(state, result, conversions, trader_data_str)
        return result, conversions, trader_data_str