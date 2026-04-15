import json
from typing import Any, Optional

# from data import LIMITS
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState

LIMITS = {
    "ASH_COATED_OSMIUM": 30,
    "INTARIAN_PEPPER_ROOT": 30,
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
    ASH_COATED_OSMIUM_DEFAULT_FAIR_VALUE = 10000
    ASH_COATED_OSMIUM_POSITION_STEP = 8
    ASH_COATED_OSMIUM_PASSIVE_SIZE = 10
    ASH_COATED_OSMIUM_TAKE_SIZE = 16

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


    def get_linear_fair_value(
        self,
        trader_data: dict,
        product: str,
        timestamp: int,
        default_fair_value: int,
    ) -> int:
        history = trader_data.get("history", {}).get(product, {})
        t = history.get("t", [])
        p = history.get("p", [])

        if len(t) < 5:
            return default_fair_value

        n = len(t)
        mean_t = sum(t) / n
        mean_p = sum(p) / n

        denom = sum((ti - mean_t) ** 2 for ti in t)
        if denom == 0:
            return round(mean_p)

        slope = sum((ti - mean_t) * (pi - mean_p) for ti, pi in zip(t, p)) / denom
        intercept = mean_p - slope * mean_t

        fair_value = intercept + slope * timestamp
        return round(fair_value)

    def update_price_history(self, trader_data: dict, product: str, price: int, window: int) -> list[int]:
        if "price_history" not in trader_data:
            trader_data["price_history"] = {}
        if product not in trader_data["price_history"]:
            trader_data["price_history"][product] = []

        trader_data["price_history"][product].append(price)
        trader_data["price_history"][product] = trader_data["price_history"][product][-window:]
        return trader_data["price_history"][product]

    def trade_intarian_pepper_root(
        self,
        order_depth: OrderDepth,
        position: int,
        limit: int,
        timestamp: int,
        trader_data: dict,
    ) -> list[Order]:
        orders: list[Order] = []

        best_bid = self.get_best_bid(order_depth)
        best_ask = self.get_best_ask(order_depth)
        mid_price = self.get_mid_price(order_depth, self.INTARIAN_PEPPER_ROOT_DEFAULT_FAIR_VALUE)

        self.update_history(
            trader_data,
            "INTARIAN_PEPPER_ROOT",
            timestamp,
            mid_price,
            self.INTARIAN_PEPPER_ROOT_WINDOW,
        )

        fair_value = self.get_linear_fair_value(
            trader_data,
            "INTARIAN_PEPPER_ROOT",
            timestamp,
            self.INTARIAN_PEPPER_ROOT_DEFAULT_FAIR_VALUE,
        )

        buy_capacity = max(0, limit - position)
        sell_capacity = max(0, limit + position)

        logger.print(
            f"INTARIAN_PEPPER_ROOT t={timestamp} pos={position} mid={mid_price} fair={fair_value} "
            f"best_bid={best_bid} best_ask={best_ask}"
        )

        # Buy dips below trend fair value
        if best_ask is not None and buy_capacity > 0 and best_ask <= fair_value - 3:
            ask_volume = -order_depth.sell_orders[best_ask]
            qty = min(ask_volume, buy_capacity, self.INTARIAN_PEPPER_ROOT_TAKE_SIZE)
            if qty > 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", best_ask, qty))

        # Only sell to reduce long inventory, not to open shorts
        elif position > 0 and best_bid is not None and best_bid >= fair_value:
            bid_volume = order_depth.buy_orders[best_bid]
            qty = min(bid_volume, position, self.INTARIAN_PEPPER_ROOT_TAKE_SIZE)
            if qty > 0:
                orders.append(Order("INTARIAN_PEPPER_ROOT", best_bid, -qty))

        return orders

    def trade_ash_coated_osmium(
        self,
        order_depth: OrderDepth,
        position: int,
        limit: int,
    ) -> list[Order]:
        orders: list[Order] = []
        best_bid = self.get_best_bid(order_depth)
        best_ask = self.get_best_ask(order_depth)
        fair_value = self.ASH_COATED_OSMIUM_DEFAULT_FAIR_VALUE

        buy_capacity = max(0, limit - position)
        sell_capacity = max(0, limit + position)

        inventory_skew = position // self.ASH_COATED_OSMIUM_POSITION_STEP
        passive_bid = fair_value - 1 - inventory_skew
        passive_ask = fair_value + 1 - inventory_skew

        if best_ask is not None and buy_capacity > 0 and best_ask <= fair_value - 1:
            ask_volume = -order_depth.sell_orders[best_ask]
            qty = min(ask_volume, buy_capacity, self.ASH_COATED_OSMIUM_TAKE_SIZE)
            if qty > 0:
                orders.append(Order("ASH_COATED_OSMIUM", best_ask, qty))
                buy_capacity -= qty

        if best_bid is not None and sell_capacity > 0 and best_bid >= fair_value + 1:
            bid_volume = order_depth.buy_orders[best_bid]
            qty = min(bid_volume, sell_capacity, self.ASH_COATED_OSMIUM_TAKE_SIZE)
            if qty > 0:
                orders.append(Order("ASH_COATED_OSMIUM", best_bid, -qty))
                sell_capacity -= qty

        if best_bid is not None and best_ask is not None and buy_capacity > 0 and passive_bid > best_bid and passive_bid < best_ask:
            qty = min(self.ASH_COATED_OSMIUM_PASSIVE_SIZE, buy_capacity)
            if qty > 0:
                orders.append(Order("ASH_COATED_OSMIUM", passive_bid, qty))

        if best_bid is not None and best_ask is not None and sell_capacity > 0 and passive_ask < best_ask and passive_ask > best_bid:
            qty = min(self.ASH_COATED_OSMIUM_PASSIVE_SIZE, sell_capacity)
            if qty > 0:
                orders.append(Order("ASH_COATED_OSMIUM", passive_ask, -qty))

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
            limit = LIMITS.get(product, 0)

            if product == "INTARIAN_PEPPER_ROOT":
                result[product] = self.trade_intarian_pepper_root(
                    order_depth,
                    position,
                    limit,
                    state.timestamp,
                    trader_data,
                )
            elif product == "ASH_COATED_OSMIUM":
                result[product] = self.trade_ash_coated_osmium(order_depth, position, limit)
            else:
                result[product] = []

        conversions = 0
        trader_data_str = json.dumps(trader_data)
        logger.flush(state, result, conversions, trader_data_str)
        return result, conversions, trader_data_str