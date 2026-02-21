from butterflow.typesystem import Operator, Atomic, Generic, DictType, Either
import numpy as np
import scipy
from bisect import bisect_left, bisect_right, insort
from collections import deque

# TODO: maybe we should use @singledispatch?
# TODO: vector neutralization for 2SLS, bucket neutralization for CI
# TODO: correl, corrline, irir

# --- Define the Type Environment (The Rules) ---
# Function signatures
# Mapping function names to their Operator signatures based on your input
STD_LIB = {
    "data": Operator(
        DictType({
            "id": Atomic("String")
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "const": Operator(
        DictType({
            "value": Atomic("Float")
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "abs": Operator(
        DictType({
            "signal": Either([Generic("Signal", Atomic("Float")), Atomic("Float")])
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "ts_delay": Operator(
        DictType({
            "signal": Generic("Signal", Atomic("Float")),
            "period": Either([Generic("Signal", Atomic("Float")), Atomic("Int")])
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "ts_diff": Operator(
        DictType({
            "signal": Generic("Signal", Atomic("Float")),
            "period": Either([Generic("Signal", Atomic("Float")), Atomic("Int")])
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "ts_mean": Operator(
        DictType({
            "signal": Generic("Signal", Atomic("Float")),
            "period": Either([Generic("Signal", Atomic("Float")), Atomic("Int")])
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "ts_sum": Operator(
        DictType({
            "signal": Generic("Signal", Atomic("Float")),
            "period": Either([Generic("Signal", Atomic("Float")), Atomic("Int")])
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "ts_decay_linear": Operator(
        DictType({
            "signal": Generic("Signal", Atomic("Float")),
            "period": Either([Generic("Signal", Atomic("Float")), Atomic("Int")])
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "ts_decay_exp": Operator(
        DictType({
            "signal": Generic("Signal", Atomic("Float")),
            "period": Either([Generic("Signal", Atomic("Float")), Atomic("Int")])
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "ts_std": Operator(
        DictType({
            "signal": Generic("Signal", Atomic("Float")),
            "period": Atomic("Int")
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "ts_mae": Operator(
        DictType({
            "signal": Generic("Signal", Atomic("Float")),
            "period": Atomic("Int")
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "ts_zscore": Operator(
        DictType({
            "signal": Generic("Signal", Atomic("Float")),
            "period": Atomic("Int")
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "ts_rank": Operator(
        DictType({
            "signal": Generic("Signal", Atomic("Float")),
            "period": Atomic("Int")
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "ts_min": Operator(
        DictType({
            "signal": Generic("Signal", Atomic("Float")),
            "period": Atomic("Int")
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "ts_max": Operator(
        DictType({
            "signal": Generic("Signal", Atomic("Float")),
            "period": Atomic("Int")
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "ts_argmin": Operator(
        DictType({
            "signal": Generic("Signal", Atomic("Float")),
            "period": Atomic("Int")
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "ts_argmax": Operator(
        DictType({
            "signal": Generic("Signal", Atomic("Float")),
            "period": Atomic("Int")
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "ts_argminmax": Operator(
        DictType({
            "signal": Generic("Signal", Atomic("Float")),
            "period": Atomic("Int")
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "ts_ffill": Operator(
        DictType({
            "signal": Generic("Signal", Atomic("Float")),
            "period": Atomic("Int")
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "tradewhen": Operator(
        DictType({
            "signal": Generic("Signal", Atomic("Float")),
            "enter": Either([Generic("Signal", Atomic("Float")), Atomic("Float")]),
            "exit": Either([Generic("Signal", Atomic("Float")), Atomic("Float")]),
            "period": Atomic("Int")
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "where": Operator(
        DictType({
            "condition": Generic("Signal", Atomic("Float")),
            "val_true": Either([Generic("Signal", Atomic("Float")), Atomic("Float")]),
            "val_false": Either([Generic("Signal", Atomic("Float")), Atomic("Float")]),
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "cs_rank": Operator(
        DictType({
            "signal": Generic("Signal", Atomic("Float")),
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "cs_zscore": Operator(
        DictType({
            "signal": Generic("Signal", Atomic("Float")),
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "add": Operator(
        DictType({
            "x": Either([Generic("Signal", Atomic("Float")), Atomic("Float")]),
            "y": Either([Generic("Signal", Atomic("Float")), Atomic("Float")]),
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "mid": Operator(
        DictType({
            "x": Either([Generic("Signal", Atomic("Float")), Atomic("Float")]),
            "y": Either([Generic("Signal", Atomic("Float")), Atomic("Float")]),
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "subtract": Operator(
        DictType({
            "x": Either([Generic("Signal", Atomic("Float")), Atomic("Float")]),
            "y": Either([Generic("Signal", Atomic("Float")), Atomic("Float")]),
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "divide": Operator(
        DictType({
            "dividend": Either([Generic("Signal", Atomic("Float")), Atomic("Float")]),
            "divisor": Either([Generic("Signal", Atomic("Float")), Atomic("Float")]),
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "multiply": Operator(
        DictType({
            "x": Either([Generic("Signal", Atomic("Float")), Atomic("Float")]),
            "y": Either([Generic("Signal", Atomic("Float")), Atomic("Float")])
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "equals": Operator(
        DictType({
            "x": Either([Generic("Signal", Atomic("Float")), Atomic("Float")]),
            "y": Either([Generic("Signal", Atomic("Float")), Atomic("Float")]),
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "greater": Operator(
        DictType({
            "signal": Either([Generic("Signal", Atomic("Float")), Atomic("Float")]),
            "thres": Either([Generic("Signal", Atomic("Float")), Atomic("Float")]),
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "less": Operator(
        DictType({
            "signal": Either([Generic("Signal", Atomic("Float")), Atomic("Float")]),
            "thres": Either([Generic("Signal", Atomic("Float")), Atomic("Float")]),
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "min": Operator(
        DictType({
            "x": Either([Generic("Signal", Atomic("Float")), Atomic("Float")]),
            "y": Either([Generic("Signal", Atomic("Float")), Atomic("Float")]),
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "max": Operator(
        DictType({
            "x": Either([Generic("Signal", Atomic("Float")), Atomic("Float")]),
            "y": Either([Generic("Signal", Atomic("Float")), Atomic("Float")]),
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "clip": Operator(
        DictType({
            "signal": Either([Generic("Signal", Atomic("Float")), Atomic("Float")]),
            "lower": Atomic("Float"),
            "upper": Atomic("Float"),
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "tail_to_nan": Operator(
        DictType({
            "signal": Either([Generic("Signal", Atomic("Float")), Atomic("Float")]),
            "lower": Atomic("Float"),
            "upper": Atomic("Float"),
        }),
        Generic("Signal", Atomic("Float"))
    ),

    "covariance": Operator(
        DictType({
            "returns": Generic("Signal", Atomic("Float")),
            "lookback": Atomic("Int")
        }),
        Generic("Matrix", Atomic("Float"))
    ),

}


def _check_ndarray(x, y, result, cache):
    # Casts result into Signal<Float> when the both inputs are all Atomic
    if type(x) != np.ndarray and type(y) != np.ndarray:
        return STD_LIB_IMPL.const(value=float(result)).compute(cache=cache)
    elif type(result) != np.ndarray:
        raise Exception("Type casting exception (Atomic result to Signal)")
    else:
        return result


def _convolve(signal, period, kernel_factory, average=True):
    # signal: ndarray
    # period: ndarray or integer
    # if average=False, sum is calculated
    n, p = signal.shape
    valid = np.isfinite(signal)
    input = np.zeros_like(signal, dtype=float)
    input[valid] = signal[valid]
    result = np.full_like(signal, np.nan, dtype=float)

    if type(period) == int:
        if 0 < period <= n:
            # kernel = np.ones((period, 1), dtype=float)
            kernel = kernel_factory(period)
            sum = scipy.signal.convolve2d(input, kernel, mode='valid')
            if average:
                count = scipy.signal.convolve2d(
                    valid, kernel, mode='valid')
                count[count == 0] = np.nan
                result[period - 1:] = sum / count
            else:
                result[period - 1:] = sum
        return result
    elif type(period) == np.ndarray:
        assert signal.shape == period.shape
        lookback = np.round(period).clip(min=0, max=n+1)
        lookback = np.nan_to_num(
            lookback, nan=0, posinf=0, neginf=0).astype(int)
        lbs = [int(x) for x in np.unique(lookback)]
        for lb in lbs:
            mask = lookback == lb
            result[mask] = _convolve(
                signal, lb, kernel_factory, average=average)[mask]
        return result
    else:
        raise Exception


class Node:
    def __repr__(self):
        # NOTE: function parameter does not start with underscore. 
        # TODO: try something safer
        args = ", ".join(
            f"{v}" for k, v in self.__dict__.items() if (not k.startswith('_')))
        return f"{self.__class__.__name__}({args})"

    def get_kwargs(self):
        op_name = type(self).__name__
        kws = STD_LIB[op_name].args.fields.keys()
        kwargs = {kw: getattr(self, kw) for kw in kws}
        return kwargs

    def compute(self, cache, flags=None):
        # Check cache
        if not flags:
            flags = dict()
        expr_str = repr(self)
        if cache and (expr_str in cache):
            return cache[expr_str]

        # Compute
        kwargs = dict()
        for kw, arg in self.get_kwargs().items():
            if issubclass(type(arg), Node):
                kwargs[kw] = arg.compute(cache)
            else:
                kwargs[kw] = arg
        self._cache = cache # NOTE: non-parameter member variable have to be marked with underscore (see repr())
        output = self._compute(**(kwargs | flags))
        if cache:
            cache[expr_str] = output

        return output


class STD_LIB_IMPL:
    @staticmethod
    def get(name):
        # Get subclass from string
        return getattr(STD_LIB_IMPL, name)

    # Graph Nodes
    class data(Node):
        def __init__(self, id): self.id = id
        def __repr__(self): return f'data("{self.id}")'

        def _compute(self, id):
            raise Exception(
                f"Data operator should be fetched from cache, and cannot be directly calculated.\nMissing item: {repr(self)}")
            # return np.ones(shape=(10, 10))

    class const(Node):
        def __init__(self, value): self.value = value
        def __repr__(self): return f'const("{self.value}")'

        def _compute(self, value):
            close = STD_LIB_IMPL.data(id="close").compute(cache=self._cache)
            return np.full_like(close, value, dtype=float)

    class abs(Node):
        def __init__(self, signal): self.signal = signal

        def _compute(self, signal):
            result = np.abs(signal)
            return _check_ndarray(signal, None, result, self._cache)

    class ts_delay(Node):
        def __init__(
                self, signal, period):
            self.signal, self.period = signal, period

        def _compute(self, signal, period):
            n, p = signal.shape
            result = np.full_like(signal, np.nan, dtype=float)
            if type(period) == int:
                if 0 <= period <= n:
                    result[period:, :] = signal[0:n-period, :]
                return result
            elif type(period) == np.ndarray:
                assert signal.shape == period.shape
                lookback = np.round(period).clip(min=0, max=n+1)
                lookback = np.nan_to_num(
                    lookback, nan=n+1, posinf=n+1, neginf=0).astype(int)
                lbs = [int(x) for x in np.unique(lookback)]
                for lb in lbs:
                    mask = lookback == lb
                    result[mask] = self._compute(signal, lb)[mask]
                return result
            else:
                raise Exception

    class ts_diff(Node):
        def __init__(
                self, signal, period):
            self.signal, self.period = signal, period

        def _compute(self, signal, period):
            n, p = signal.shape
            result = np.full_like(signal, np.nan, dtype=float)
            if type(period) == int:
                if 0 < period <= n:
                    result[period:, :] = signal[period:, :] - \
                        signal[0:n-period, :]
                return result
            elif type(period) == np.ndarray:
                assert signal.shape == period.shape
                lookback = np.round(period).clip(min=0, max=n+1)
                lookback = np.nan_to_num(
                    lookback, nan=0, posinf=0, neginf=0).astype(int)
                lbs = [int(x) for x in np.unique(lookback)]
                for lb in lbs:
                    mask = lookback == lb
                    result[mask] = self._compute(signal, lb)[mask]
                return result
            else:
                raise Exception

    class ts_mean(Node):
        def __init__(
                self, signal, period):
            self.signal, self.period = signal, period

        def _compute(self, signal, period):
            def kernel_factory(period): return np.ones(
                (period, 1), dtype=float)
            return _convolve(signal, period, kernel_factory, average=True)

    class ts_sum(Node):
        def __init__(
                self, signal, period):
            self.signal, self.period = signal, period

        def _compute(self, signal, period):
            def kernel_factory(period): return np.ones(
                (period, 1), dtype=float)
            return _convolve(signal, period, kernel_factory, average=False)

    class ts_decay_linear(Node):
        def __init__(
                self, signal, period):
            self.signal, self.period = signal, period

        def _compute(self, signal, period):
            def kernel_factory(period): return np.linspace(
                [1./period], [1.], num=period, dtype=float, axis=0)
            return _convolve(signal, period, kernel_factory, average=True)

    class ts_decay_exp(Node):
        def __init__(
                self, signal, period):
            self.signal, self.period = signal, period

        def _compute(self, signal, period):
            def kernel_factory(period): return np.exp(
                np.linspace([-3.], [0.], num=period, dtype=float, axis=0))
            return _convolve(signal, period, kernel_factory, average=True)

    class ts_std(Node):
        def __init__(
                self, signal, period):
            self.signal, self.period = signal, period

        def _compute(self, signal, period):
            n, p = signal.shape
            result = np.full_like(signal, np.nan, dtype=float)
            if type(period) == int:
                if 0 < period <= n:
                    for p in range(period, n+1):
                        result[p-1] = np.nanstd(signal[p-period:p], axis=0)
                return result
            else:
                raise Exception

    class ts_mae(Node):
        def __init__(
                self, signal, period):
            self.signal, self.period = signal, period

        def _compute(self, signal, period):
            n, p = signal.shape
            result = np.full_like(signal, np.nan, dtype=float)
            if type(period) == int:
                if 0 < period <= n:
                    for p in range(period, n+1):
                        result[p-1] = np.nanmean(
                            np.abs(signal[p-period:p] - np.nanmean(signal[p-period:p], axis=0)), axis=0)
                return result
            else:
                raise Exception

    class ts_zscore(Node):
        def __init__(
                self, signal, period):
            self.signal, self.period = signal, period

        def _compute(self, signal, period):
            if type(period) == int:
                mu = STD_LIB_IMPL.ts_mean(
                    signal=signal, period=period).compute(self._cache)
                sigm = STD_LIB_IMPL.ts_std(
                    signal=signal, period=period).compute(self._cache)
                z = STD_LIB_IMPL.divide(
                    dividend=signal-mu, divisor=sigm).compute(self._cache)
                return z
            else:
                raise Exception

    class ts_rank(Node):
        def __init__(
                self, signal, period):
            self.signal, self.period = signal, period

        def _compute(self, signal: np.ndarray, period: int) -> np.ndarray:
            """
            Parameters
            ----------
            signal : ndarray, shape (T, N)
                axis 0: rolling dimension (time)
                axis 1: independent series
            period : int, positive

            Returns
            -------
            out : ndarray, shape (T, N)
                Rolling rank of x[t, j] within x[t-period+1:t+1, j]
            """
            x = np.asarray(signal, dtype=float)
            T, N = x.shape
            out = np.full((T, N), np.nan)

            if period > 0:
                # one sorted buffer per column
                buffers = [[] for _ in range(N)]

                for t in range(T):
                    for j in range(N):
                        v = x[t, j]
                        buf = buffers[j]

                        # insert new value
                        if not np.isnan(v):
                            insort(buf, v)

                        # remove leaving value
                        if t >= period:
                            old = x[t - period, j]
                            if not np.isnan(old):
                                idx = bisect_left(buf, old)
                                del buf[idx]

                        # compute rank
                        if t >= period - 1 and not np.isnan(v):
                            left = bisect_left(buf, v)
                            right = bisect_right(buf, v)
                            out[t, j] = left + (right - left + 1) / 2

            return out

    class ts_min(Node):
        def __init__(self, signal, period):
            self.signal, self.period = signal, period

        def _compute(self, signal: np.ndarray, period: int, flip=False, argsort=False) -> np.ndarray:
            """
            Rolling minimum using monotonic deque (O(T * N))

            NaN handling:
            - NaNs are ignored
            - Output is NaN iff all values in the window are NaN
            """
            x = np.asarray(signal, dtype=float)
            T, N = x.shape
            out = np.full((T, N), np.nan)

            if period <= 0:
                return out

            # one deque + valid-count per column
            deques = [deque() for _ in range(N)]  # stores (index, value)
            valid_counts = np.zeros(N, dtype=int)

            for t in range(T):
                for j in range(N):
                    v = x[t, j]
                    dq = deques[j]

                    # ---- remove expired index ----
                    if dq and dq[0][0] <= t - period:
                        dq.popleft()

                    if t >= period and not np.isnan(x[t - period, j]):
                        valid_counts[j] -= 1

                    # ---- insert new value ----
                    if not np.isnan(v):
                        valid_counts[j] += 1
                        if not flip:
                            while dq and dq[-1][1] > v:
                                dq.pop()
                        else:
                            while dq and dq[-1][1] < v:
                                dq.pop()
                        dq.append((t, v))

                    # ---- read output ----
                    if t >= period - 1 and valid_counts[j] > 0:
                        if not argsort:
                            out[t, j] = dq[0][1]
                        else:
                            out[t, j] = (dq[0][0] - (t - period + 1))/period

            return out

    class ts_max(Node):
        def __init__(self, signal, period):
            self.signal, self.period = signal, period

        def _compute(self, signal: np.ndarray, period: int) -> np.ndarray:
            # Rolling maximum using monotonic deque
            return STD_LIB_IMPL.ts_min(signal=signal, period=period).compute(cache=self._cache, flags={"flip": True})

    class ts_argmin(Node):
        def __init__(
                self, signal, period):
            self.signal, self.period = signal, period

        def _compute(self, signal, period):
            return STD_LIB_IMPL.ts_min(signal=signal, period=period).compute(cache=self._cache, flags={"flip": False, "argsort": True})

    class ts_argmax(Node):
        def __init__(
                self, signal, period):
            self.signal, self.period = signal, period

        def _compute(self, signal, period):
            return STD_LIB_IMPL.ts_min(signal=signal, period=period).compute(cache=self._cache, flags={"flip": True, "argsort": True})

    class ts_argminmax(Node):
        def __init__(
                self, signal, period):
            self.signal, self.period = signal, period

        def _compute(self, signal, period):
            return (
                STD_LIB_IMPL.ts_argmin(
                    signal=signal, period=period).compute(cache=self._cache)
                - STD_LIB_IMPL.ts_argmax(signal=signal,
                                         period=period).compute(cache=self._cache)
            )

    class ts_ffill(Node):
        def __init__(
                self, signal, period):
            self.signal, self.period = signal, period

        def _compute(self, signal: np.ndarray, period: int) -> np.ndarray:
            """
            NumPy implementation of pandas.DataFrame.ffill(limit=limit)
            inf is processed as a valid number. (see tradewhen)

            Parameters
            ----------
            signal : np.ndarray (2D)
                Input array with NaNs.
            period : int
                Maximum number of consecutive NaNs to fill.

            Returns
            -------
            out : np.ndarray
                Forward-filled array.
            """
            arr = np.asarray(signal)
            if arr.ndim != 2:
                raise ValueError("Input must be 2D")

            n, m = arr.shape
            out = arr.copy()

            for j in range(m):
                col = out[:, j]

                valid = ~np.isnan(col)
                idx = np.where(valid, np.arange(n), -1)

                # last valid index up to each row
                last = np.maximum.accumulate(idx)

                # distance from last valid
                dist = np.arange(n) - last

                # fill condition
                fill_mask = (~valid) & (last >= 0) & (dist <= period)

                col[fill_mask] = col[last[fill_mask]]

            return out

    class tradewhen(Node):
        def __init__(
                self, signal, enter, exit, period):
            self.signal, self.enter, self.exit, self.period = signal, enter, exit, period

        def _compute(self, signal: np.ndarray, enter, exit, period: int) -> np.ndarray:
            """
            Updates output to signal, when enter is positive. 
            Keeps the value until the exit is positive.
            When exit is positive, the value is discarded to nan
            """
            if type(enter) == np.ndarray:
                enter_cond = enter > 0.
            else:
                enter_cond = np.full_like(signal, enter > 0., dtype=bool)
            if type(exit) == np.ndarray:
                exit_cond = exit > 0.
            else:
                exit_cond = np.full_like(signal, exit > 0., dtype=bool)
            exit_cond &= exit_cond & (~enter_cond)

            input = np.nan_to_num(
                signal, copy=True, nan=np.nan, posinf=np.nan, neginf=np.nan)
            input = np.where(enter_cond, input, np.nan)
            # np.inf is used as exit marker
            input = np.where(exit_cond, np.inf, input)

            output = STD_LIB_IMPL.ts_ffill(
                signal=input, period=period).compute(cache=self._cache)
            output = np.nan_to_num(
                output, copy=False, nan=np.nan, posinf=np.nan, neginf=np.nan)

            return output

    class where(Node):
        def __init__(
                self, condition, val_true, val_false):
            self.condition, self.val_true, self.val_false = condition, val_true, val_false

        def _compute(self, condition: np.ndarray, val_true, val_false) -> np.ndarray:
            """
            Depending on the sign of the condition, use val_true or val_false.
            When the condition is zero, the result is nan.
            """
            output = np.full_like(condition, np.nan, dtype=float)
            mask_true = condition > 0
            mask_false = condition < 0
            if type(val_true) == np.ndarray:
                output[mask_true] = val_true[mask_true]
            else:
                output[mask_true] = val_true
            if type(val_false) == np.ndarray:
                output[mask_false] = val_false[mask_false]
            else:
                output[mask_false] = val_false
            return output

    class cs_rank(Node):
        def __init__(self, signal): self.signal = signal

        def _compute(self, signal: np.ndarray) -> np.ndarray:
            """
            NumPy equivalent of pandas.DataFrame.rank(axis=1)
            - method='average'
            - ascending=True
            - na_option='keep'
            """
            x = np.asarray(signal, dtype=float)
            out = np.full_like(x, np.nan, dtype=float)

            for i in range(x.shape[0]):
                row = x[i]
                mask = ~np.isnan(row)
                vals = row[mask]

                if vals.size == 0:
                    continue

                order = np.argsort(vals, kind="mergesort")
                sorted_vals = vals[order]

                ranks = np.empty_like(sorted_vals, dtype=float)

                # assign average ranks for ties
                j = 0
                while j < len(sorted_vals):
                    k = j
                    while k < len(sorted_vals) and sorted_vals[k] == sorted_vals[j]:
                        k += 1
                    # pandas ranks start from 1
                    avg_rank = 0.5 * (j + 1 + k)
                    ranks[j:k] = avg_rank
                    j = k

                # invert permutation
                inv = np.empty_like(order)
                inv[order] = np.arange(len(order))

                out[i, mask] = ranks[inv]

            return out

    class cs_zscore(Node):
        def __init__(self, signal): self.signal = signal

        def _compute(self, signal: np.ndarray) -> np.ndarray:
            x = np.asarray(signal, dtype=float)
            out = np.full_like(x, np.nan, dtype=float)

            for i in range(x.shape[0]):
                row = x[i]
                mask = ~np.isnan(row)
                vals = row[mask]

                if vals.size == 0:
                    continue

                mu = np.nanmean(vals)
                sigma = np.nanstd(vals)
                if sigma <= 0.:
                    sigma = np.nan

                out[i, mask] = (vals - mu) / sigma

            return out

    class add(Node):
        def __init__(self, x, y): self.x, self.y = x, y

        def _compute(self, x, y):
            result = x + y
            return _check_ndarray(x, y, result, self._cache)

    class mid(Node):
        def __init__(self, x, y): self.x, self.y = x, y

        def _compute(self, x, y):
            result = (x + y) * 0.5
            return _check_ndarray(x, y, result, self._cache)

    class subtract(Node):
        def __init__(self, x, y): self.x, self.y = x, y

        def _compute(self, x, y):
            result = x - y
            return _check_ndarray(x, y, result, self._cache)

    class divide(Node):
        def __init__(self, dividend,
                     divisor): self.dividend, self.divisor = dividend, divisor
        def _compute(self, dividend,
                     divisor):
            result = dividend / divisor
            return _check_ndarray(dividend, divisor, result, self._cache)

    class multiply(Node):
        def __init__(self, x, y):
            self.x, self.y = x, y

        def _compute(self, x, y):
            result = x * y
            return _check_ndarray(x, y, result, self._cache)

    class equals(Node):
        def __init__(self, x, y):
            self.x, self.y = x, y

        def _compute(self, x, y):
            if type(x) == np.ndarray or type(y) == np.ndarray:
                result = (x == y).astype(float)
            else:
                result = 1. if x == y else 0.
            return _check_ndarray(x, y, result, self._cache)

    class greater(Node):
        def __init__(self, signal, thres):
            self.signal, self.thres = signal, thres

        def _compute(self, signal, thres):
            if type(signal) == np.ndarray or type(thres) == np.ndarray:
                result = (signal > thres).astype(float)
            else:
                result = 1. if signal > thres else 0.
            return _check_ndarray(signal, thres, result, self._cache)

    class less(Node):
        def __init__(self, signal, thres):
            self.signal, self.thres = signal, thres

        def _compute(self, signal, thres):
            if type(signal) == np.ndarray or type(thres) == np.ndarray:
                result = (signal < thres).astype(float)
            else:
                result = 1. if signal < thres else 0.
            return _check_ndarray(signal, thres, result, self._cache)

    class min(Node):
        def __init__(self, x, y):
            self.x, self.y = x, y

        def _compute(self, x, y):
            result = np.minimum(x, y)
            return _check_ndarray(x, y, result, self._cache)

    class max(Node):
        def __init__(self, x, y):
            self.x, self.y = x, y

        def _compute(self, x, y):
            result = np.maximum(x, y)
            return _check_ndarray(x, y, result, self._cache)

    class clip(Node):
        def __init__(self, signal, lower, upper):
            self.signal, self.lower, self.upper = signal, lower, upper

        def _compute(self, signal, lower, upper):
            result = np.clip(a=signal, a_min=lower, a_max=upper)
            return _check_ndarray(signal, None, result, self._cache)

    class tail_to_nan(Node):
        def __init__(self, signal, lower, upper):
            self.signal, self.lower, self.upper = signal, lower, upper

        def _compute(self, signal, lower, upper):
            if type(signal) == float:
                result = signal if lower < signal < upper else float('nan')
            elif type(signal) == np.ndarray:
                result = np.full_like(signal, np.nan, dtype=float)
                mask = (lower <= signal) & (signal <= upper)
                result[mask] = signal[mask]
            else:
                raise Exception
            return _check_ndarray(signal, None, result, self._cache)

    class covariance(Node):
        # TODO: shrinkage or regularization, neutralization, sector information etc.
        def __init__(self, returns, lookback):
            self.returns, self.lookback = returns, lookback

        def _compute(self, returns, lookback):
            n, p = returns.shape
            result = np.full((n, p, p), np.nan, dtype=float)
            for t in range(lookback, n):
                result[t] = np.cov(returns[t-lookback: t+1], rowvar=False)
            return result
