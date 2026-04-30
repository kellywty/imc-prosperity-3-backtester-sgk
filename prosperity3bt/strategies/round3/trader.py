import json
import math
from json import JSONEncoder
from statistics import NormalDist
from typing import Any, Dict, List, Optional

# ============================================================
# Inlined datamodel (single-file submission compatibility)
# ============================================================
Time = int
Symbol = str
Product = str
Position = int
UserId = str
ObservationValue = int


class Listing:
    def __init__(self, symbol: Symbol, product: Product, denomination: int):
        self.symbol = symbol
        self.product = product
        self.denomination = denomination


class ConversionObservation:
    def __init__(
        self,
        bidPrice: float,
        askPrice: float,
        transportFees: float,
        exportTariff: float,
        importTariff: float,
        sugarPrice: float,
        sunlightIndex: float,
    ):
        self.bidPrice = bidPrice
        self.askPrice = askPrice
        self.transportFees = transportFees
        self.exportTariff = exportTariff
        self.importTariff = importTariff
        self.sugarPrice = sugarPrice
        self.sunlightIndex = sunlightIndex


class Observation:
    def __init__(
        self,
        plainValueObservations: Dict[Product, ObservationValue],
        conversionObservations: Dict[Product, ConversionObservation],
    ) -> None:
        self.plainValueObservations = plainValueObservations
        self.conversionObservations = conversionObservations

    def __str__(self) -> str:
        return json.dumps(
            {
                "plainValueObservations": self.plainValueObservations,
                "conversionObservations": {
                    k: v.__dict__ for k, v in self.conversionObservations.items()
                },
            },
            separators=(",", ":"),
        )


class Order:
    def __init__(self, symbol: Symbol, price: int, quantity: int) -> None:
        self.symbol = symbol
        self.price = price
        self.quantity = quantity

    def __str__(self) -> str:
        return "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"

    def __repr__(self) -> str:
        return "(" + self.symbol + ", " + str(self.price) + ", " + str(self.quantity) + ")"


class OrderDepth:
    def __init__(self):
        self.buy_orders: Dict[int, int] = {}
        self.sell_orders: Dict[int, int] = {}


class Trade:
    def __init__(
        self,
        symbol: Symbol,
        price: int,
        quantity: int,
        buyer: UserId = None,
        seller: UserId = None,
        timestamp: int = 0,
    ) -> None:
        self.symbol = symbol
        self.price: int = price
        self.quantity: int = quantity
        self.buyer = buyer
        self.seller = seller
        self.timestamp = timestamp

    def __str__(self) -> str:
        return (
            "("
            + self.symbol
            + ", "
            + str(self.buyer)
            + " << "
            + str(self.seller)
            + ", "
            + str(self.price)
            + ", "
            + str(self.quantity)
            + ", "
            + str(self.timestamp)
            + ")"
        )

    def __repr__(self) -> str:
        return self.__str__()


class TradingState(object):
    def __init__(
        self,
        traderData: str,
        timestamp: Time,
        listings: Dict[Symbol, Listing],
        order_depths: Dict[Symbol, OrderDepth],
        own_trades: Dict[Symbol, List[Trade]],
        market_trades: Dict[Symbol, List[Trade]],
        position: Dict[Product, Position],
        observations: Observation,
    ):
        self.traderData = traderData
        self.timestamp = timestamp
        self.listings = listings
        self.order_depths = order_depths
        self.own_trades = own_trades
        self.market_trades = market_trades
        self.position = position
        self.observations = observations

    def toJSON(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True)


class ProsperityEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__

_N = NormalDist()


# ============================================================
# Product universe & limits
# ============================================================
HYDROGEL_SYMBOL = "HYDROGEL_PACK"
EXTRACT_SYMBOL = "VELVETFRUIT_EXTRACT"

VOUCHER_STRIKES: dict[str, int] = {
    "VEV_4000": 4000,
    "VEV_4500": 4500,
    "VEV_5000": 5000,
    "VEV_5100": 5100,
    "VEV_5200": 5200,
    "VEV_5300": 5300,
    "VEV_5400": 5400,
    "VEV_5500": 5500,
    "VEV_6000": 6000,
    "VEV_6500": 6500,
}
VOUCHER_SYMBOLS = list(VOUCHER_STRIKES.keys())

POS_LIMITS: dict[str, int] = {
    HYDROGEL_SYMBOL: 200,
    EXTRACT_SYMBOL: 200,
    **{v: 300 for v in VOUCHER_SYMBOLS},
}


# ============================================================
# Options config (vol smile calibrated offline from round 3 day 0-2 data)
# ============================================================
DAYS_PER_YEAR = 365

# iv = a * m^2 + b * m + c, where m = log(K/S) / sqrt(T)
SMILE_A = 0.02972997
SMILE_B = 0.00225217
SMILE_C = 0.23947520

# Anchor-EMA window (for BS theoretical-vs-market mid bias absorption).
# Tuned via grid search: 20 outperforms 10/15/25/30/40/50 on the historical
# round 3 days (peak total profit ~+61.4k vs +57.9k at the default).
THEO_BIAS_WINDOW = 20

# NB: A one-tick mean-reversion bias on `fair` (negative lag-1 autocorr ~-0.14)
# was tried and removed. It looked +3.8% in backtest but lost -2,736 on
# HYDROGEL_PACK in live (run 452625) because it FIGHTS the inventory skew during
# trending regimes: each down-tick raised fair, encouraging us to chase dips
# longer just as skew was trying to flatten longs into the trend. The signal is
# real on average but loses in directionally-driven sessions.

# ---- HYDROGEL_PACK long-horizon mean-reversion overlay ----
# When best mid is far from the all-time HYDROGEL fair, take aggressively in
# the reverting direction. This is structurally different from the failed
# 1-tick MR bias: it only fires on REGIME-LEVEL dislocations (multi-tick
# moves away from a stable historical mean), so it doesn't fight skew on
# every noisy tick. See `HydrogelOverlayTrader` below.
#
# Parameters tuned via grid search (LONG_FAIR x DISLOC x VOL) over 3 backtest
# days. Chosen point optimises worst-day PnL while staying near the total
# maximum: at this setting all 3 days are strongly positive (+52k worst,
# +68k best, +183k total vs +58k baseline). EXTRACT mean drifts ~9 ticks
# day-over-day (vs HYDROGEL's ~1.3) so it gets no overlay - only HYDROGEL.
LONG_FAIR_HYDRO = 9988        # all-time mean ~9990.8, optimum at 9988
DISLOC_TICKS = 40             # only fire on >= ~5*1tick-std deviations from fair
MAX_OVERLAY_VOL = 20          # per-tick clip; small clip = patient accumulation


def _tte_days_at_day_start() -> int:
    """Submission-safe TTE for round 3 final simulation."""
    return 5


# ============================================================
# Logger (visualizer-compatible)
# ============================================================
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


# ============================================================
# Base ProductTrader
# ============================================================
class ProductTrader:
    def __init__(self, name: str, state: TradingState):
        self.name = name
        self.state = state
        self.position_limit = POS_LIMITS.get(name, 0)
        self.initial_position = state.position.get(name, 0)
        self.orders: list[Order] = []

        self.mkt_buy_orders, self.mkt_sell_orders = self._get_order_depth()
        self.bid_wall, self.wall_mid, self.ask_wall = self._get_walls()
        self.best_bid, self.best_ask = self._get_best_bid_ask()

        self.max_allowed_buy_volume = self.position_limit - self.initial_position
        self.max_allowed_sell_volume = self.position_limit + self.initial_position

    def _get_order_depth(self) -> tuple[dict[int, int], dict[int, int]]:
        depth: Optional[OrderDepth] = self.state.order_depths.get(self.name)
        if depth is None:
            return {}, {}
        buy_orders = {bp: abs(bv) for bp, bv in sorted(depth.buy_orders.items(), key=lambda x: -x[0])}
        sell_orders = {sp: abs(sv) for sp, sv in sorted(depth.sell_orders.items(), key=lambda x: x[0])}
        return buy_orders, sell_orders

    def _get_best_bid_ask(self) -> tuple[Optional[int], Optional[int]]:
        best_bid = max(self.mkt_buy_orders) if self.mkt_buy_orders else None
        best_ask = min(self.mkt_sell_orders) if self.mkt_sell_orders else None
        return best_bid, best_ask

    def _get_walls(self) -> tuple[Optional[int], Optional[float], Optional[int]]:
        bid_wall = min(self.mkt_buy_orders) if self.mkt_buy_orders else None
        ask_wall = max(self.mkt_sell_orders) if self.mkt_sell_orders else None
        wall_mid: Optional[float] = None
        if bid_wall is not None and ask_wall is not None:
            wall_mid = (bid_wall + ask_wall) / 2.0
        return bid_wall, wall_mid, ask_wall

    def bid(self, price: float, volume: int) -> None:
        abs_volume = min(abs(int(volume)), self.max_allowed_buy_volume)
        if abs_volume <= 0:
            return
        self.orders.append(Order(self.name, int(price), abs_volume))
        self.max_allowed_buy_volume -= abs_volume

    def ask(self, price: float, volume: int) -> None:
        abs_volume = min(abs(int(volume)), self.max_allowed_sell_volume)
        if abs_volume <= 0:
            return
        self.orders.append(Order(self.name, int(price), -abs_volume))
        self.max_allowed_sell_volume -= abs_volume


# ============================================================
# StaticTrader: wall-anchored MM with optional fair-value override
# ============================================================
class StaticTrader(ProductTrader):
    """Market-making strategy mirroring the FrankfurtHedgehogs RAINFOREST_RESIN
    logic. Operates around `fair` (defaults to wall_mid) which is a stable
    proxy for true value. Two-stage:

      1) TAKING - lift any ask <= fair - 1 (definite edge); also lift asks at
         fair when short (flatten); symmetric for bids.

      2) MAKING - post a passive bid one tick inside the bid wall (or one
         tick above the best substantive bid that's still below fair) and
         analogously for the ask.

    Every tick is independent in Prosperity, so this won't suffer typical
    MM adverse selection from stale quotes.
    """

    def __init__(self, name: str, state: TradingState, fair: Optional[float] = None):
        super().__init__(name, state)
        self.fair = fair if fair is not None else self.wall_mid

    def get_orders(self) -> list[Order]:
        if self.fair is None or self.bid_wall is None or self.ask_wall is None:
            return self.orders

        # ---- 1) TAKING ----
        for sp, sv in self.mkt_sell_orders.items():
            if sp <= self.fair - 1:
                self.bid(sp, sv)
            elif sp <= self.fair and self.initial_position < 0:
                self.bid(sp, min(sv, abs(self.initial_position)))

        for bp, bv in self.mkt_buy_orders.items():
            if bp >= self.fair + 1:
                self.ask(bp, bv)
            elif bp >= self.fair and self.initial_position > 0:
                self.ask(bp, min(bv, self.initial_position))

        # ---- 2) MAKING ----
        # Need at least a 3-tick wall spread to MM safely. With tighter walls
        # the "post one inside the wall" logic produces crossed quotes that
        # then get safety-fixed to a position that immediately lifts the offer.
        wall_spread = self.ask_wall - self.bid_wall
        if wall_spread < 3:
            return self.orders

        bid_price = int(self.bid_wall + 1)
        ask_price = int(self.ask_wall - 1)

        for bp, bv in self.mkt_buy_orders.items():
            overbid = bp + 1
            if bv > 1 and overbid < self.fair:
                bid_price = max(bid_price, overbid)
                break
            elif bp < self.fair:
                bid_price = max(bid_price, bp)
                break

        for sp, sv in self.mkt_sell_orders.items():
            underbid = sp - 1
            if sv > 1 and underbid > self.fair:
                ask_price = min(ask_price, underbid)
                break
            elif sp > self.fair:
                ask_price = min(ask_price, sp)
                break

        # Inventory-aware skew: when long, lean both quotes lower; when short,
        # lean them higher. Nudges flow toward flattening positions.
        # Tuned via grid search: 100 (1 tick per 100 contracts) outperforms
        # 30/50/200 on the historical round 3 days; aggressive skews shave the
        # edge captured per fill more than they reduce inventory risk.
        skew = self.initial_position // 100
        bid_price -= skew
        ask_price -= skew

        # Quotes must respect the wall and be uncrossed.
        bid_price = min(bid_price, int(self.ask_wall) - 1)
        ask_price = max(ask_price, int(self.bid_wall) + 1)
        if bid_price >= ask_price:
            return self.orders

        self.bid(bid_price, self.max_allowed_buy_volume)
        self.ask(ask_price, self.max_allowed_sell_volume)

        return self.orders


# ============================================================
# HYDROGEL-only overlay: long-horizon mean-reversion takes
# ============================================================
class HydrogelOverlayTrader(StaticTrader):
    """StaticTrader extended with a long-horizon mean-reversion overlay.

    BEFORE the standard wall-MM phases run, if the best mid is at least
    DISLOC_TICKS away from LONG_FAIR_HYDRO, we lift asks (or hit bids) up
    to MAX_OVERLAY_VOL contracts at any price still <= LONG_FAIR_HYDRO - 1
    (or >= LONG_FAIR_HYDRO + 1). The overlay uses the parent's bid()/ask()
    helpers, so it shares the position-budget bookkeeping with the MM
    layer. We mutate the local order-book dict to remove consumed lots,
    so the StaticTrader take phase doesn't double-target them.
    """

    def get_orders(self) -> list[Order]:
        if self.best_bid is None or self.best_ask is None:
            return super().get_orders()

        cur_mid = (self.best_bid + self.best_ask) / 2.0
        dislocation = cur_mid - LONG_FAIR_HYDRO

        if dislocation <= -DISLOC_TICKS:
            # Way below long-run fair: lift any cheap ask
            budget = MAX_OVERLAY_VOL
            for px in sorted(list(self.mkt_sell_orders.keys())):
                if px > LONG_FAIR_HYDRO - 1 or budget <= 0:
                    break
                avail = self.mkt_sell_orders[px]
                qty = min(avail, budget, self.max_allowed_buy_volume)
                if qty <= 0:
                    break
                self.bid(px, qty)
                self.mkt_sell_orders[px] -= qty
                if self.mkt_sell_orders[px] == 0:
                    del self.mkt_sell_orders[px]
                budget -= qty

        elif dislocation >= DISLOC_TICKS:
            # Way above long-run fair: hit any rich bid
            budget = MAX_OVERLAY_VOL
            for px in sorted(list(self.mkt_buy_orders.keys()), reverse=True):
                if px < LONG_FAIR_HYDRO + 1 or budget <= 0:
                    break
                avail = self.mkt_buy_orders[px]
                qty = min(avail, budget, self.max_allowed_sell_volume)
                if qty <= 0:
                    break
                self.ask(px, qty)
                self.mkt_buy_orders[px] -= qty
                if self.mkt_buy_orders[px] == 0:
                    del self.mkt_buy_orders[px]
                budget -= qty

        return super().get_orders()


# ============================================================
# Black-Scholes helpers + smile fair-value computation
# ============================================================
def _smile_iv(S: float, K: float, T: float) -> float:
    if T <= 0 or S <= 0:
        return SMILE_C
    m = math.log(K / S) / math.sqrt(T)
    return max(0.01, SMILE_A * m * m + SMILE_B * m + SMILE_C)


def _bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0 or S <= 0:
        return max(0.0, S - K)
    sqrt_t = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return S * _N.cdf(d1) - K * _N.cdf(d2)


def _bs_theo(S: float, K: float, T: float) -> float:
    return _bs_call(S, K, T, _smile_iv(S, K, T))


def _option_fair(
    state: TradingState,
    name: str,
    K: int,
    T: float,
    last_trader_data: dict,
    new_trader_data: dict,
) -> Optional[float]:
    """Compute a stable fair price for a voucher.

    Combines the smile-based BS theo with an EMA of the realised
    (wall_mid - theo) bias to absorb persistent per-strike mispricings (which
    diagnostics show exist, e.g. VEV_5400 sits ~0.3 IV below the smile).
    """
    extract_depth: Optional[OrderDepth] = state.order_depths.get(EXTRACT_SYMBOL)
    if extract_depth is None or not extract_depth.buy_orders or not extract_depth.sell_orders:
        return None

    S = (max(extract_depth.buy_orders) + min(extract_depth.sell_orders)) / 2.0

    depth: Optional[OrderDepth] = state.order_depths.get(name)
    if depth is None or not depth.buy_orders or not depth.sell_orders:
        return None
    wall_mid = (min(depth.buy_orders) + max(depth.sell_orders)) / 2.0

    theo = _bs_theo(S, K, T)
    diff = wall_mid - theo
    key = f"{name}_bias"
    prev = last_trader_data.get(key, diff)
    alpha = 2.0 / (THEO_BIAS_WINDOW + 1.0)
    new_bias = alpha * diff + (1.0 - alpha) * prev
    new_trader_data[key] = new_bias

    return theo + new_bias


# ============================================================
# Trader entry point
# ============================================================
class Trader:
    # MM all strikes that have a tradable two-sided book. Tight-spread
    # strikes (VEV_5400/5500, wall spread <= 2) are rejected by StaticTrader
    # itself since making would cross. Deep OTM (VEV_6000/6500) are effectively
    # zero-bid; we skip them to save loop time.
    ACTIVE_VOUCHERS = {"VEV_4000", "VEV_4500", "VEV_5000", "VEV_5100", "VEV_5200",
                       "VEV_5300", "VEV_5400", "VEV_5500"}

    def run(self, state: TradingState):
        if state.traderData:
            try:
                last_trader_data = json.loads(state.traderData)
            except Exception:
                last_trader_data = {}
        else:
            last_trader_data = {}
        new_trader_data: dict = {}

        result: dict[Symbol, list[Order]] = {}

        # ---- Delta-1 products ----
        # HYDROGEL_PACK gets the long-horizon mean-reversion overlay on top
        # of wall-MM; EXTRACT stays as pure wall-MM (its mean is less stable
        # day-over-day, so a hardcoded fair would be over-fit there).
        for sym in (HYDROGEL_SYMBOL, EXTRACT_SYMBOL):
            if sym not in state.order_depths:
                continue
            try:
                if sym == HYDROGEL_SYMBOL:
                    t = HydrogelOverlayTrader(sym, state)
                else:
                    t = StaticTrader(sym, state)
                orders = t.get_orders()
                if orders:
                    result[sym] = orders
            except Exception:
                pass

        # ---- Vouchers: BS-anchored static MM around (theo + EMA bias) ----
        tte_days = _tte_days_at_day_start() - state.timestamp / 1_000_000
        T = tte_days / DAYS_PER_YEAR
        for sym in self.ACTIVE_VOUCHERS:
            if sym not in state.order_depths or T <= 0:
                continue
            try:
                K = VOUCHER_STRIKES[sym]
                fair = _option_fair(state, sym, K, T, last_trader_data, new_trader_data)
                if fair is None:
                    continue
                t = StaticTrader(sym, state, fair=fair)
                orders = t.get_orders()
                if orders:
                    result[sym] = orders
            except Exception:
                pass

        # ---- Persist any unused trader-data keys so EMAs survive ticks ----
        for k, v in last_trader_data.items():
            new_trader_data.setdefault(k, v)

        conversions = 0
        try:
            trader_data_str = json.dumps(new_trader_data)
        except Exception:
            trader_data_str = ""

        logger.flush(state, result, conversions, trader_data_str)
        return result, conversions, trader_data_str
