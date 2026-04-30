"""Round 5 algorithmic-trading strategy.

Design overview
---------------
The 50 round-5 products break into two buckets:

  1. **Tradable** (≈40 products) — run a passive market maker (`MMTrader`)
     that:
       * uses `wall_mid` (deepest bid + ask, /2) as fair value — far
         less reactive to single-tick inner-book moves than book_mid;
       * quotes 1 tick inside the inner book on whichever side has
         volume > 1 (skipping 1-lot probe orders that are
         informational / near-cancel);
       * takes any obvious mispricing vs fair (asks ≤ fair-1 or bids
         ≥ fair+1).

  2. **Disabled** (`DISABLED_PRODUCTS`) — the handful of products that
     consistently bleed to MM on the 3-day backtest (most likely
     because their tick volatility dominates the spread we capture).

The framework below also supports `SignalManager`s that can override
`fair_value`, `target_position`, etc. per product, but the cross-leg
pair signals we tried (SNACKPACK CHOC/VAN sum, PISTACHIO/STRAWBERRY/
RASPBERRY trio, PEBBLES XL/basket, OXYGEN_SHAKE pair) didn't beat the
plain MM in this iteration — they're left in place as a starting
point for the next iteration.

Logging is kept compatible with the IMC visualizer (Logger class).
"""

import json
import math
from json import JSONEncoder
from typing import Any, Dict, List, Optional


# ============================================================
# Inlined datamodel (single-file submission compatibility).
# The Prosperity submission environment provides a top-level
# `datamodel` module — these definitions match its public API
# so we can run the same file locally and remotely.
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
        return self.__str__()


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


# ---------------------------------------------------------------------------
# Logger (visualizer-compatible) — unchanged from existing trader
# ---------------------------------------------------------------------------

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]],
              conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions, "", "",
        ]))
        max_item_length = (self.max_log_length - base_length) // 3
        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))
        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp, trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        return [[l.symbol, l.product, l.denomination] for l in listings.values()]

    def compress_order_depths(self, ods: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        return {s: [od.buy_orders, od.sell_orders] for s, od in ods.items()}

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        out = []
        for arr in trades.values():
            for t in arr:
                out.append([t.symbol, t.price, t.quantity, t.buyer, t.seller, t.timestamp])
        return out

    def compress_observations(self, observations: Observation) -> list[Any]:
        conv = {}
        for p, o in observations.conversionObservations.items():
            conv[p] = [o.bidPrice, o.askPrice, o.transportFees, o.exportTariff,
                       o.importTariff, o.sugarPrice, o.sunlightIndex]
        return [observations.plainValueObservations, conv]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        out = []
        for arr in orders.values():
            for o in arr:
                out.append([o.symbol, o.price, o.quantity])
        return out

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""
        while lo <= hi:
            mid = (lo + hi) // 2
            cand = value[:mid]
            if len(cand) < len(value):
                cand += "..."
            if len(json.dumps(cand)) <= max_length:
                out = cand
                lo = mid + 1
            else:
                hi = mid - 1
        return out


logger = Logger()


# ---------------------------------------------------------------------------
# Universe & group definitions
# ---------------------------------------------------------------------------

POS_LIMIT = 10  # all 50 products share the same limit

ALL_PRODUCTS = [
    # GALAXY_SOUNDS
    "GALAXY_SOUNDS_DARK_MATTER", "GALAXY_SOUNDS_BLACK_HOLES",
    "GALAXY_SOUNDS_PLANETARY_RINGS", "GALAXY_SOUNDS_SOLAR_WINDS",
    "GALAXY_SOUNDS_SOLAR_FLAMES",
    # SLEEP_POD
    "SLEEP_POD_SUEDE", "SLEEP_POD_LAMB_WOOL", "SLEEP_POD_POLYESTER",
    "SLEEP_POD_NYLON", "SLEEP_POD_COTTON",
    # MICROCHIP
    "MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_SQUARE",
    "MICROCHIP_RECTANGLE", "MICROCHIP_TRIANGLE",
    # PEBBLES
    "PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL",
    # ROBOT
    "ROBOT_VACUUMING", "ROBOT_MOPPING", "ROBOT_DISHES",
    "ROBOT_LAUNDRY", "ROBOT_IRONING",
    # UV_VISOR
    "UV_VISOR_YELLOW", "UV_VISOR_AMBER", "UV_VISOR_ORANGE",
    "UV_VISOR_RED", "UV_VISOR_MAGENTA",
    # TRANSLATOR
    "TRANSLATOR_SPACE_GRAY", "TRANSLATOR_ASTRO_BLACK",
    "TRANSLATOR_ECLIPSE_CHARCOAL", "TRANSLATOR_GRAPHITE_MIST",
    "TRANSLATOR_VOID_BLUE",
    # PANEL
    "PANEL_1X2", "PANEL_2X2", "PANEL_1X4", "PANEL_2X4", "PANEL_4X4",
    # OXYGEN_SHAKE
    "OXYGEN_SHAKE_MORNING_BREATH", "OXYGEN_SHAKE_EVENING_BREATH",
    "OXYGEN_SHAKE_MINT", "OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_GARLIC",
    # SNACKPACK
    "SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA", "SNACKPACK_PISTACHIO",
    "SNACKPACK_STRAWBERRY", "SNACKPACK_RASPBERRY",
]


# ---------------------------------------------------------------------------
# Per-product market-making book-keeping
# ---------------------------------------------------------------------------

class MMTrader:
    """Stateless-per-tick passive MM with inventory skew + take logic.

    Owns the book-keeping for one product: best bid/ask, walls, capacity,
    and exposes `bid()` / `ask()` helpers that auto-clip to the position
    limit so we can never over-quote.
    """

    def __init__(self, name: str, state: TradingState):
        self.name = name
        self.state = state
        self.position_limit = POS_LIMIT
        self.position = state.position.get(name, 0)

        od = state.order_depths.get(name)
        # buy_orders: dict {price: positive volume}
        # sell_orders: dict {price: NEGATIVE volume in the data model}
        self.buy_orders = dict(od.buy_orders) if od else {}
        self.sell_orders = {p: -abs(v) for p, v in od.sell_orders.items()} if od else {}

        self.best_bid = max(self.buy_orders) if self.buy_orders else None
        self.best_ask = min(self.sell_orders) if self.sell_orders else None
        self.bid_wall = min(self.buy_orders) if self.buy_orders else None
        self.ask_wall = max(self.sell_orders) if self.sell_orders else None
        self.wall_mid = (
            (self.bid_wall + self.ask_wall) / 2
            if self.bid_wall is not None and self.ask_wall is not None
            else None
        )
        self.book_mid = (
            (self.best_bid + self.best_ask) / 2
            if self.best_bid is not None and self.best_ask is not None
            else self.wall_mid
        )

        self.max_buy = self.position_limit - self.position
        self.max_sell = self.position_limit + self.position
        self.orders: list[Order] = []

        # Default fair_value = wall_mid (deepest bid/ask average) — slow-moving
        # and avoids the adverse-selection hit you take when chasing the
        # inner book. Signal managers can override this per product.
        self.fair_value: Optional[float] = self.wall_mid if self.wall_mid is not None else self.book_mid
        self.fair_value_shift: float = 0.0  # signal-driven additive shift to fair
        self.target_position: int = 0
        # Per-strategy aggressiveness (signal managers may override)
        self.take_edge: float = 1.0       # take only if mispriced by this much vs fair
        # Tiny skew leans reserves off drift in prod (hurt raw backtest; helps balance).
        self.skew_per_unit: float = 0.1
        self.min_take_volume: int = 2     # don't lift small-volume bait orders
        # Frankfurt-style: improve only vs levels with vol >= this (skip 1-lot probes).
        self.min_overbid_volume: int = 2
        self.max_make_size: int = 10      # cap each new make-side quote (vs limit 10)
        self.soft_position_cap: int = 7   # cap inventory build-up vs hard limit 10
        self.enabled: bool = True         # set False to skip the product entirely

    # -- Order placement primitives -------------------------------------------------
    def bid(self, price: int, qty: int) -> None:
        qty = max(0, min(qty, self.max_buy))
        if qty <= 0 or price is None:
            return
        self.orders.append(Order(self.name, int(price), int(qty)))
        self.max_buy -= qty

    def ask(self, price: int, qty: int) -> None:
        qty = max(0, min(qty, self.max_sell))
        if qty <= 0 or price is None:
            return
        self.orders.append(Order(self.name, int(price), -int(qty)))
        self.max_sell -= qty

    # -- Take + make engine ---------------------------------------------------------
    def trade(self) -> list[Order]:
        if not self.enabled:
            return self.orders
        if self.fair_value is None or self.best_bid is None or self.best_ask is None:
            return self.orders

        # Apply any signal-driven shift on top of the wall-mid fair.
        fair = self.fair_value + self.fair_value_shift

        # Reservation-price shift: when long, lower BOTH quotes (less eager to
        # buy more, more eager to sell); when short, raise BOTH. Pulls
        # inventory back toward `target_position` over time.
        # See Avellaneda-Stoikov (2008) for the canonical derivation.
        inv_shift = -(self.position - self.target_position) * self.skew_per_unit

        # 1. TAKE: only on obvious mispricing vs fair value (no inventory
        # tilt — that path leads to sweeping the book on signal flips).
        # Require min_take_volume on the level so we don't get baited by
        # 1-lot probe orders posted to manipulate our quotes.
        buy_take_thr = fair - self.take_edge
        sell_take_thr = fair + self.take_edge

        # Same asymmetric band as MAKES (`bid_room` / `ask_room`), but enforced on
        # TAKE too — soft cap used to gate only passive quotes lets aggressive
        # lifts sprint to ±position_limit when misprices line up (bad in prod).
        soft_pc = self.soft_position_cap
        take_buy_remain = max(
            0, min(self.position_limit - self.position, soft_pc - self.position)
        )
        take_sell_remain = max(
            0,
            min(
                self.position_limit + self.position,
                self.position + soft_pc,
            ),
        )

        for ask_p in sorted(self.sell_orders):
            ask_v = abs(self.sell_orders[ask_p])
            if ask_p > buy_take_thr or take_buy_remain <= 0 or self.max_buy <= 0:
                break
            if ask_v < self.min_take_volume:
                continue
            qty = min(ask_v, self.max_buy, take_buy_remain)
            if qty <= 0:
                continue
            mb0 = self.max_buy
            self.bid(ask_p, qty)
            take_buy_remain -= mb0 - self.max_buy

        for bid_p in sorted(self.buy_orders, reverse=True):
            bid_v = self.buy_orders[bid_p]
            if bid_p < sell_take_thr or take_sell_remain <= 0 or self.max_sell <= 0:
                break
            if bid_v < self.min_take_volume:
                continue
            qty = min(bid_v, self.max_sell, take_sell_remain)
            if qty <= 0:
                continue
            ms0 = self.max_sell
            self.ask(bid_p, qty)
            take_sell_remain -= ms0 - self.max_sell

        # 2. MAKE: passive quotes 1 tick inside first level whose volume ≥
        # min_overbid_volume; otherwise match best bid / ask vs fair. Skip on
        # 1-tick-wide markets.
        spread = self.best_ask - self.best_bid
        if spread < 2:
            return self.orders

        # Soft position cap: stop quoting on the side that would push us
        # past `soft_position_cap`. Otherwise inventory drifts to the
        # exchange limit and we lose all flexibility.
        soft = self.soft_position_cap
        bid_room = max(0, soft - self.position)
        ask_room = max(0, self.position + soft)

        if self.max_buy > 0 and bid_room > 0:
            bid_price = self.bid_wall + 1  # default fallback
            for bp in sorted(self.buy_orders, reverse=True):
                bv = self.buy_orders[bp]
                if bv >= self.min_overbid_volume and bp + 1 < fair:
                    bid_price = max(bid_price, bp + 1)
                    break
                elif bp < fair:
                    bid_price = max(bid_price, bp)
                    break
            bid_price += int(round(inv_shift))           # reservation-price shift
            bid_price = min(self.best_ask - 1, bid_price)
            qty = min(self.max_buy, bid_room, self.max_make_size)
            if bid_price <= fair and qty > 0:
                self.bid(bid_price, qty)

        if self.max_sell > 0 and ask_room > 0:
            ask_price = self.ask_wall - 1  # default fallback
            for sp in sorted(self.sell_orders):
                sv = abs(self.sell_orders[sp])
                if sv >= self.min_overbid_volume and sp - 1 > fair:
                    ask_price = min(ask_price, sp - 1)
                    break
                elif sp > fair:
                    ask_price = min(ask_price, sp)
                    break
            ask_price += int(round(inv_shift))           # same shift on both sides
            ask_price = max(self.best_bid + 1, ask_price)
            qty = min(self.max_sell, ask_room, self.max_make_size)
            if ask_price >= fair and qty > 0:
                self.ask(ask_price, qty)

        return self.orders


# ---------------------------------------------------------------------------
# Helpers for signal state stored in traderData JSON across ticks
# ---------------------------------------------------------------------------

def update_ema_var(td: dict, key: str, value: float, alpha: float, init_std: float
                  ) -> tuple[float, float]:
    """Online EMA mean and EMA variance. Returns (mean, std)."""
    bucket = td.setdefault("ema", {}).setdefault(key, {})
    prev_mean = float(bucket.get("m", value))
    prev_var = float(bucket.get("v", init_std * init_std))
    mean = (1 - alpha) * prev_mean + alpha * value
    var = max(1.0, (1 - alpha) * prev_var + alpha * (value - mean) ** 2)
    bucket["m"] = mean
    bucket["v"] = var
    return mean, math.sqrt(var)


def hysteretic_state(td: dict, key: str, score: float, entry: float, exit_: float,
                     trend: bool) -> int:
    """Hysteretic regime state in {-1, 0, +1}.

    `trend=True` means we go *with* the score (return signal), `trend=False`
    means we mean-revert against it.
    """
    state = int(td.setdefault("regime", {}).get(key, 0))
    if score >= entry:
        state = 1 if trend else -1
    elif score <= -entry:
        state = -1 if trend else 1
    elif abs(score) <= exit_:
        state = 0
    td["regime"][key] = state
    return state


# ---------------------------------------------------------------------------
# Group-level signal managers
# ---------------------------------------------------------------------------

class SignalManager:
    """Computes (fair_value, target_position, take_edge, make_edge) for a
    set of leg products by inspecting the current state and updating the
    cross-tick statistics in `td`.
    """

    def assign(self, traders: dict[str, MMTrader], td: dict) -> None:
        raise NotImplementedError


class CrossLegFairShift(SignalManager):
    """Shift each leg's fair value toward what the basket regression implies.

    Generic cross-leg signal: if `sum_i w_i * mid_i` is stationary (true when
    the legs have near-±1 return correlations), the deviation from its EMA
    mean is the *residual* of the cointegrating relationship. We attribute
    that residual proportionally back to each leg and shift the fair value
    by a fraction of it.

    Crucially we shift FAIR VALUE only — never `target_position`. The latter
    blew up v1 because the take logic would sweep the book to chase the
    target. A bounded fair shift only re-prices our quotes; takes still
    need a real mispricing vs the SHIFTED fair, so this can never become a
    runaway directional bet.

    Bounded at MAX_SHIFT ticks per leg.
    """

    LEGS_W: tuple[tuple[str, float], ...] = ()  # (leg, weight)
    KEY: str = ""
    INIT_STD: float = 100.0
    MAX_SHIFT: float = 2.0
    SHIFT_K: float = 0.5  # fraction of residual to attribute (per leg)

    def assign(self, traders, td):
        legs = [l for l, _ in self.LEGS_W]
        if any(s not in traders for s in legs):
            return
        mids = {s: traders[s].wall_mid for s in legs}
        if any(v is None for v in mids.values()):
            return
        # Weighted basket value
        basket = sum(w * mids[s] for s, w in self.LEGS_W)
        mean, _std = update_ema_var(td, self.KEY, basket,
                                    alpha=0.02, init_std=self.INIT_STD)
        resid = basket - mean
        # Per-leg shift: -resid * SHIFT_K * sign(weight) / |weight|.
        # If weight is +1 and resid is +5 (basket is rich), shift = -2.5
        # → drop fair value, less eager to buy this leg.
        # If weight is -1 (e.g., negatively-correlated leg), the same
        # positive residual implies that leg is *cheap* on the spread,
        # so we'd shift its fair value UP instead.
        for leg, w in self.LEGS_W:
            shift = -resid * self.SHIFT_K * (1.0 if w > 0 else -1.0) / max(abs(w), 1.0)
            shift = max(-self.MAX_SHIFT, min(self.MAX_SHIFT, shift))
            traders[leg].fair_value_shift = float(shift)


class SnackpackChocVanShift(CrossLegFairShift):
    """Δcorr(CHOC, VAN) ≈ -0.92 → CHOC + VAN is stationary."""
    LEGS_W = (("SNACKPACK_CHOCOLATE", 1.0), ("SNACKPACK_VANILLA", 1.0))
    KEY = "xleg_chocvan"
    INIT_STD = 80.0
    MAX_SHIFT = 2.0
    SHIFT_K = 0.5


class SnackpackTrioShift(CrossLegFairShift):
    """Δcorr(PIST, STRAW) ≈ +0.91; both ≈ -0.87 vs RASP. Hence
    PIST + STRAW - 2*RASP is the (approximately) stationary combo.
    """
    LEGS_W = (
        ("SNACKPACK_PISTACHIO", 1.0),
        ("SNACKPACK_STRAWBERRY", 1.0),
        ("SNACKPACK_RASPBERRY", -2.0),
    )
    KEY = "xleg_trio"
    INIT_STD = 150.0
    MAX_SHIFT = 2.0
    SHIFT_K = 0.4


class PebblesClusterShift(CrossLegFairShift):
    """Each of XS/S/M/L has Δcorr ≈ -0.5 with XL; XS/S/M/L are uncorrelated
    with each other. Equal variances among XS-L (~15 tick std), XL ~30. The
    sum XS+S+M+L+XL is approximately stationary (var → 0 in the ideal limit).
    """
    LEGS_W = (
        ("PEBBLES_XS", 1.0),
        ("PEBBLES_S", 1.0),
        ("PEBBLES_M", 1.0),
        ("PEBBLES_L", 1.0),
        ("PEBBLES_XL", 1.0),
    )
    KEY = "xleg_pebbles"
    INIT_STD = 100.0
    MAX_SHIFT = 2.0
    SHIFT_K = 0.05  # 5 legs, residual gets attributed proportionally


# ---------------------------------------------------------------------------
# Top-level Trader
# ---------------------------------------------------------------------------

class Trader:
    # Cross-leg fair-value shifts only — bounded at ±2 ticks per leg, and
    # they touch fair value only (never `target_position`). Earlier
    # implementations that set inventory targets blew up because the take
    # logic would sweep the book to chase the target. Here, takes still
    # require real mispricing vs the *shifted* fair, so the worst case is
    # quoting marginally on the wrong side for a few ticks.
    # Pebbles cluster disabled temporarily — backtest-stable but noisy regime;
    # re-enable once we have gated EMA warm-up / spread checks.
    SIGNAL_MANAGERS: list[SignalManager] = [
        SnackpackChocVanShift(),
        SnackpackTrioShift(),
    ]

    # Empirically loses material PnL to plain MM on the 3-day backtest
    # (each was net negative on at least 2 of the 3 days). Disabled to
    # free risk budget; re-enable later if we get a signal that monetizes
    # them. Marginal-loss products are NOT disabled to avoid over-fitting.
    DISABLED_PRODUCTS: set[str] = {
        "SLEEP_POD_LAMB_WOOL",
        "PEBBLES_M",
        "PEBBLES_XS",
        "MICROCHIP_TRIANGLE",
        "MICROCHIP_OVAL",
        "PANEL_1X2",
        "ROBOT_MOPPING",
        "ROBOT_VACUUMING",
        "TRANSLATOR_SPACE_GRAY",
    }

    # Noise tracking: EMA of tick-to-tick |Δwall_mid| per product. Currently
    # we don't auto-tune anything from it — it's stored on traderData for
    # diagnostics and future use. Earlier experiment scaling take_edge by
    # noise hurt PnL by ~$33K (taking less aggressively on noisy names).
    NOISE_ALPHA = 0.05

    def run(self, state: TradingState):
        td = self._load_td(state.traderData)
        traders = {p: MMTrader(p, state) for p in ALL_PRODUCTS if p in state.order_depths}

        for p in self.DISABLED_PRODUCTS:
            if p in traders:
                traders[p].enabled = False

        # Track per-product tick noise (stored on traderData; used by
        # cross-leg signals, not by base MM after noise-edge experiment
        # showed it hurts PnL).
        prev_mid = td.setdefault("prev_mid", {})
        noise = td.setdefault("noise", {})
        for p, t in traders.items():
            if t.wall_mid is None:
                continue
            cur = float(t.wall_mid)
            pm = prev_mid.get(p)
            if pm is not None:
                d = abs(cur - pm)
                prev_n = float(noise.get(p, d))
                noise[p] = (1 - self.NOISE_ALPHA) * prev_n + self.NOISE_ALPHA * d
            prev_mid[p] = cur

        for mgr in self.SIGNAL_MANAGERS:
            try:
                mgr.assign(traders, td)
            except Exception as e:  # noqa: BLE001 - never crash submission
                logger.print(f"signal-mgr {type(mgr).__name__} failed: {e}")

        result: dict[Symbol, list[Order]] = {}
        for product, t in traders.items():
            try:
                orders = t.trade()
            except Exception as e:  # noqa: BLE001
                logger.print(f"trade {product} failed: {e}")
                orders = []
            if orders:
                result[product] = orders

        td_str = json.dumps(td, separators=(",", ":"))
        logger.flush(state, result, 0, td_str)
        return result, 0, td_str

    @staticmethod
    def _load_td(raw: str) -> dict:
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            return {}
