from enum import Enum
from typing import List, Optional, Tuple
from dataclasses import dataclass


class CandlePatternType(Enum):
    DOJI = "doji"
    HAMMER = "hammer"
    INVERTED_HAMMER = "inverted_hammer"
    SHOOTING_STAR = "shooting_star"
    SPINNING_TOP = "spinning_top"
    MARUBOZU_BULL = "marubozu_bull"
    MARUBOZU_BEAR = "marubozu_bear"
    ENGULFING_BULL = "engulfing_bull"
    ENGULFING_BEAR = "engulfing_bear"
    MORNING_STAR = "morning_star"
    EVENING_STAR = "evening_star"
    THREE_WHITE_SOLDIERS = "three_white_soldiers"
    THREE_BLACK_CROWS = "three_black_crows"
    HARAMI_BULL = "harami_bull"
    HARAMI_BEAR = "harami_bear"
    PIERCING_LINE = "piercing_line"
    DARK_CLOUD = "dark_cloud"
    TWEEZER_TOP = "tweezer_top"
    TWEEZER_BOTTOM = "tweezer_bottom"


@dataclass
class CandlePatternResult:
    pattern: CandlePatternType
    symbol: str
    timeframe: str
    confidence: float
    direction: str
    candle_index: int


class CandlePatternDetector:

    @staticmethod
    def _body(o, c):
        return abs(c - o)

    @staticmethod
    def _upper_shadow(o, h, c):
        return h - max(o, c)

    @staticmethod
    def _lower_shadow(o, l, c):
        return min(o, c) - l

    @staticmethod
    def _is_bullish(o, c):
        return c > o

    @staticmethod
    def _is_bearish(o, c):
        return c < o

    @staticmethod
    def _range(h, l):
        return h - l

    def detect_all(self, candles) -> List[CandlePatternType]:
        if not candles or len(candles) < 3:
            return []

        found = []

        last = candles[-1]
        prev = candles[-2]
        prev2 = candles[-3]

        o, h, l, c = last.open, last.high, last.low, last.close
        po, ph, pl, pc = prev.open, prev.high, prev.low, prev.close
        p2o, p2h, p2l, p2c = prev2.open, prev2.high, prev2.low, prev2.close

        rng = self._range(h, l)
        body = self._body(o, c)
        upper = self._upper_shadow(o, h, c)
        lower = self._lower_shadow(o, l, c)

        p_rng = self._range(ph, pl)
        p_body = self._body(po, pc)

        if rng == 0:
            return found

        body_ratio = body / rng

        if self._detect_doji(body_ratio, rng):
            found.append(CandlePatternType.DOJI)

        if self._detect_hammer(body_ratio, lower, upper, body, rng, o, c):
            found.append(CandlePatternType.HAMMER)

        if self._detect_inverted_hammer(body_ratio, lower, upper, body, rng, o, c):
            found.append(CandlePatternType.INVERTED_HAMMER)

        if self._detect_shooting_star(body_ratio, lower, upper, body, rng, o, c):
            found.append(CandlePatternType.SHOOTING_STAR)

        if self._detect_spinning_top(body_ratio, upper, lower, rng):
            found.append(CandlePatternType.SPINNING_TOP)

        if self._detect_marubozu_bull(body_ratio, o, c, upper, lower, rng):
            found.append(CandlePatternType.MARUBOZU_BULL)

        if self._detect_marubozu_bear(body_ratio, o, c, upper, lower, rng):
            found.append(CandlePatternType.MARUBOZU_BEAR)

        if p_rng > 0:
            if self._detect_engulfing_bull(o, c, po, pc, p_body, body):
                found.append(CandlePatternType.ENGULFING_BULL)

            if self._detect_engulfing_bear(o, c, po, pc, p_body, body):
                found.append(CandlePatternType.ENGULFING_BEAR)

            if self._detect_harami_bull(o, c, po, pc, p_body, body):
                found.append(CandlePatternType.HARAMI_BULL)

            if self._detect_harami_bear(o, c, po, pc, p_body, body):
                found.append(CandlePatternType.HARAMI_BEAR)

            if self._detect_piercing_line(o, c, po, pc, ph, pl):
                found.append(CandlePatternType.PIERCING_LINE)

            if self._detect_dark_cloud(o, c, po, pc, ph, pl):
                found.append(CandlePatternType.DARK_CLOUD)

            if self._detect_tweezer_top(h, ph, rng, p_rng, o, c, po, pc):
                found.append(CandlePatternType.TWEEZER_TOP)

            if self._detect_tweezer_bottom(l, pl, rng, p_rng, o, c, po, pc):
                found.append(CandlePatternType.TWEEZER_BOTTOM)

        if len(candles) >= 3 and p_rng > 0:
            p2_rng = self._range(p2h, p2l)
            if p2_rng > 0:
                if self._detect_morning_star(p2o, p2c, po, pc, o, c, p2h, p2l, ph, pl):
                    found.append(CandlePatternType.MORNING_STAR)

                if self._detect_evening_star(p2o, p2c, po, pc, o, c, p2h, p2l, ph, pl):
                    found.append(CandlePatternType.EVENING_STAR)

        if len(candles) >= 5:
            if self._detect_three_white_soldiers(candles):
                found.append(CandlePatternType.THREE_WHITE_SOLDIERS)

            if self._detect_three_black_crows(candles):
                found.append(CandlePatternType.THREE_BLACK_CROWS)

        return found

    def _detect_doji(self, body_ratio, rng):
        return body_ratio < 0.1 and rng > 0

    def _detect_hammer(self, body_ratio, lower, upper, body, rng, o, c):
        if body_ratio < 0.1:
            return False
        return (lower >= body * 2 and
                upper <= body * 0.5 and
                body_ratio < 0.4)

    def _detect_inverted_hammer(self, body_ratio, lower, upper, body, rng, o, c):
        if body_ratio < 0.1:
            return False
        return (upper >= body * 2 and
                lower <= body * 0.5 and
                body_ratio < 0.4)

    def _detect_shooting_star(self, body_ratio, lower, upper, body, rng, o, c):
        if body_ratio < 0.1:
            return False
        return (upper >= body * 2 and
                lower <= body * 0.5 and
                body_ratio < 0.4 and
                self._is_bearish(o, c))

    def _detect_spinning_top(self, body_ratio, upper, lower, rng):
        if rng == 0:
            return False
        return (0.1 <= body_ratio <= 0.35 and
                upper / rng > 0.2 and
                lower / rng > 0.2)

    def _detect_marubozu_bull(self, body_ratio, o, c, upper, lower, rng):
        return (self._is_bullish(o, c) and
                body_ratio > 0.85 and
                upper / rng < 0.05 and
                lower / rng < 0.05)

    def _detect_marubozu_bear(self, body_ratio, o, c, upper, lower, rng):
        return (self._is_bearish(o, c) and
                body_ratio > 0.85 and
                upper / rng < 0.05 and
                lower / rng < 0.05)

    def _detect_engulfing_bull(self, o, c, po, pc, p_body, body):
        return (self._is_bearish(po, pc) and
                self._is_bullish(o, c) and
                o <= pc and c >= po and
                body > p_body * 1.05)

    def _detect_engulfing_bear(self, o, c, po, pc, p_body, body):
        return (self._is_bullish(po, pc) and
                self._is_bearish(o, c) and
                o >= pc and c <= po and
                body > p_body * 1.05)

    def _detect_harami_bull(self, o, c, po, pc, p_body, body):
        return (self._is_bearish(po, pc) and
                self._is_bullish(o, c) and
                o >= pc and c <= po and
                body < p_body * 0.6 and
                p_body > 0)

    def _detect_harami_bear(self, o, c, po, pc, p_body, body):
        return (self._is_bullish(po, pc) and
                self._is_bearish(o, c) and
                o <= pc and c >= po and
                body < p_body * 0.6 and
                p_body > 0)

    def _detect_piercing_line(self, o, c, po, pc, ph, pl):
        p_mid = (po + pc) / 2
        return (self._is_bearish(po, pc) and
                self._is_bullish(o, c) and
                o < pc and
                c > p_mid and
                c < po)

    def _detect_dark_cloud(self, o, c, po, pc, ph, pl):
        p_mid = (po + pc) / 2
        return (self._is_bullish(po, pc) and
                self._is_bearish(o, c) and
                o > pc and
                c < p_mid and
                c > po)

    def _detect_morning_star(self, p2o, p2c, po, pc, o, c, p2h, p2l, ph, pl):
        p_body = self._body(po, pc)
        p_rng = self._range(ph, pl)
        if p_rng == 0:
            return False
        small_body = p_body / p_rng < 0.3
        return (self._is_bearish(p2o, p2c) and
                small_body and
                self._is_bullish(o, c) and
                c > (p2o + p2c) / 2 and
                max(po, pc) < min(p2o, p2c))

    def _detect_evening_star(self, p2o, p2c, po, pc, o, c, p2h, p2l, ph, pl):
        p_body = self._body(po, pc)
        p_rng = self._range(ph, pl)
        if p_rng == 0:
            return False
        small_body = p_body / p_rng < 0.3
        return (self._is_bullish(p2o, p2c) and
                small_body and
                self._is_bearish(o, c) and
                c < (p2o + p2c) / 2 and
                min(po, pc) > max(p2o, p2c))

    def _detect_three_white_soldiers(self, candles):
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]
        if not (self._is_bullish(c1.open, c1.close) and
                self._is_bullish(c2.open, c2.close) and
                self._is_bullish(c3.open, c3.close)):
            return False
        if not (c2.close > c1.close and c3.close > c2.close):
            return False
        if not (c2.open > c1.open and c3.open > c2.open):
            return False
        for cx in [c1, c2, c3]:
            rng = self._range(cx.high, cx.low)
            if rng == 0:
                return False
            body = self._body(cx.open, cx.close)
            if body / rng < 0.5:
                return False
        return True

    def _detect_three_black_crows(self, candles):
        c1, c2, c3 = candles[-3], candles[-2], candles[-1]
        if not (self._is_bearish(c1.open, c1.close) and
                self._is_bearish(c2.open, c2.close) and
                self._is_bearish(c3.open, c3.close)):
            return False
        if not (c2.close < c1.close and c3.close < c2.close):
            return False
        if not (c2.open < c1.open and c3.open < c2.open):
            return False
        for cx in [c1, c2, c3]:
            rng = self._range(cx.high, cx.low)
            if rng == 0:
                return False
            body = self._body(cx.open, cx.close)
            if body / rng < 0.5:
                return False
        return True

    def _detect_tweezer_top(self, h, ph, rng, p_rng, o, c, po, pc):
        avg_rng = (rng + p_rng) / 2
        if avg_rng == 0:
            return False
        tolerance = avg_rng * 0.05
        return (abs(h - ph) <= tolerance and
                self._is_bullish(po, pc) and
                self._is_bearish(o, c))

    def _detect_tweezer_bottom(self, l, pl, rng, p_rng, o, c, po, pc):
        avg_rng = (rng + p_rng) / 2
        if avg_rng == 0:
            return False
        tolerance = avg_rng * 0.05
        return (abs(l - pl) <= tolerance and
                self._is_bearish(po, pc) and
                self._is_bullish(o, c))
