# todo : rank, cache

from collections import OrderedDict
import numpy as np 
import pandas as pd 

# GLOBAL_CACHE = {}

class Operator:
    def __init__(self, arguments=None):
        self.arguments = OrderedDict(arguments) if arguments else {}

    def __repr__(self):
        args_str = ", " .join([f"{k} = {repr(v)}" for k,v in self.arguments.items()])
        return f"{type(self).__name__} ({args_str})"

    def check_two_input(self,x,y):
        if x.keys() != y.keys():
            x0 = list(x.keys())[0]
            x1 = list(x.keys())[-1]
            y0 = list(y.keys())[0]
            y1 = list(y.keys())[-1]
            raise ValueError(f"[{type(self).__name__}], x=[{x0}..{x1}]({len(x.keys())}), y=[{y0}..{y1}]({len(y.keys())})")
        return 

    # def cached(fn):
    #     def cache(self, **kwargs) :
    #         if key in GLOBAL_CACHE:
    #             # need to implement .copy()
    #             return GLOBAL_CACHE[key].copy()
    #         key = repr(self)
    #         result = fn(self, **kwargs)
    #         GLOBAL_CACHE[key] = result.copy()
    #         return result
    #     return cache
    
    def calculate(self, dmgr):
        raise NotImplementedError("Subclasses should implement this method")

########################
### Binary operators ###
########################

class OpDivide(Operator):
    def __init__(self, x=None, y=None, eps=None, clip=None):
        super().__init__([("x", x), ("y", y), ("eps", eps), ("clip", clip)])

    def __repr__(self):
        return f"({self.arguments['x']} / {self.arguments['y']})"
    
    def calculate(self, dmgr):
        x = self.arguments["x"].calculate(dmgr)
        y = self.arguments["y"].calculate(dmgr)
        eps = float(self.arguments["eps"])
        clip = float(self.arguments["clip"])
        self.check_two_input(x,y)
        return OrderedDict([(k, np.clip( v1/np.where(np.abs(v2)<eps, np.nan, v2), -clip, clip)) for k,v1,v2 in zip(x.keys(), x.values(), y.values())])

class OpMultiply(Operator):
    def __init__(self, x=None, y=None):
        super().__init__([("x", x), ("y", y)])
    
    def __repr__(self):
        return f"({self.arguments['x']} * {self.arguments['y']})"

    def calculate(self, dmgr):
        x = self.arguments["x"].calculate(dmgr)
        y = self.arguments["y"].calculate(dmgr)
        self.check_two_input(x,y)
        return OrderedDict([(k, v1*v2) for k,v1,v2 in zip(x.keys(), x.values(), y.values())])

class OpAdd(Operator):
    def __init__(self, x=None, y=None):
        super().__init__([("x", x), ("y", y)])
    
    def __repr__(self):
        return f"({self.arguments['x']} + {self.arguments['y']})"

    def calculate(self, dmgr):
        x = self.arguments["x"].calculate(dmgr)
        y = self.arguments["y"].calculate(dmgr)
        self.check_two_input(x,y)
        return OrderedDict([(k, v1+v2) for k,v1,v2 in zip(x.keys(), x.values(), y.values())])

class OpSubtract(Operator):
    def __init__(self, x=None, y=None, eps=1e-6):
        super().__init__([("x", x), ("y", y), ("eps", eps)])

    def __repr__(self):
        return f"({self.arguments['x']} - {self.arguments['y']})"

    def calculate(self, dmgr):
        x = self.arguments["x"].calculate(dmgr)
        y = self.arguments["y"].calculate(dmgr)
        eps = float(self.arguments["eps"])
        self.check_two_input(x,y)
        return OrderedDict([(k, np.where(np.abs(v1-v2)<eps, 0, v1-v2)) for k,v1,v2 in zip(x.keys(), x.values(), y.values())])


class OpCorr(Operator):
    def __init__(self, x=None, y=None, a=None):
        super().__init__([("x", x), ("y", y), ("a", a)])
    
    def __repr__(self):
        return f"corr({repr(self.arguments['x'])},{repr(self.arguments['y'])},{self.arguments['a']})"

    @staticmethod
    def lookback_length(x, a):
        return x+a

    @staticmethod
    def calculate_one(df1, df2, a):
        return df1.rolling(a).corr(df2)

    def calculate(self, dmgr):
        # convert lookback from intervals to days, and fetch
        x = self.arguments["x"].calculate(dmgr)
        y = self.arguments["y"].calculate(dmgr)
        self.check_two_input(x,y)
        a = int(self.arguments["a"])
        keys = list(x.keys())
        values1 = list(x.values())
        values2 = list(y.values())
        blocks = a // len(values1[0])
        results = []
        for i, k in enumerate(keys):
            lookback_length = self.lookback_length(len(values1[i]), a)
            assert lookback_length == self.lookback_length(len(values2[i]), a)
            lookback1 = np.concatenate([[np.nan] * lookback_length] + [values1[ii] for ii in range(i-blocks-1, i+1) if ii >= 0])[-lookback_length:]
            lookback2 = np.concatenate([[np.nan] * lookback_length] + [values2[ii] for ii in range(i-blocks-1, i+1) if ii >= 0])[-lookback_length:]
            df = self.calculate_one(pd.Series(lookback1),pd.Series(lookback2),a)
            result = df.values[-len(values1[0]):]
            assert result.shape == values1[i].shape
            results.append(result)
        return OrderedDict(zip(keys,results))
    

class OpNeutralize(OpCorr):
    def __init__(self, x=None, y=None, a=None):
        super().__init__([("x", x), ("y", y), ("a", a)])
    
    def __repr__(self):
        return f"neut({repr(self.arguments['x'])},{repr(self.arguments['y'])},{self.arguments['a']})"

    def calculate(self, dmgr):
        corr = super().calculate(self, dmgr)
        x = self.arguments["x"].calculate(dmgr)
        y = self.arguments["y"].calculate(dmgr)
        self.check_two_input(corr,x)
        self.check_two_input(x,y)
        return OrderedDict([(key, v1-v2*corr) for key,v1,v2,c in zip(x.keys(), x.values(), y.values(), corr.values())])
    

########################
### Unary operators  ###
########################

class OpDelay(Operator):
    def __init__(self, x=None, day=None):
        super().__init__([("x", x), ("day", day)])
    
    def __repr__(self):
        return f"delay({repr(self.arguments['x'])},{self.arguments['day']})"

    def calculate(self, dmgr):
        x = self.arguments["x"].calculate(dmgr)
        day = int(self.arguments["day"])
        keys = list(x.keys())
        nans = np.full(x[keys[0]].shape, np.nan)
        values = [nans.copy() for _ in range(day)] + [x[k] for k in keys[:-day]]
        if len(keys) != len(values):
            raise ValueError(f"[OpDelay] keys and values do not match.\ndata length = {len(keys)}, delay = {day}")
        return OrderedDict(zip(keys, values))

class OpPower(Operator):
    def __init__(self, x=None, a=None):
        super().__init__([("x", x), ("a", a)])
    
    def __repr__(self):
        return f"power({repr(self.arguments['x'])},{self.arguments['a']})"

    def calculate(self, dmgr):
        x = self.arguments["x"].calculate(dmgr)
        a = float(self.arguments["a"])
        return OrderedDict((k,np.sign(v) * np.abs(np.power(v,a))) for k,v in x.items())

class OpAbs(Operator):
    def __init__(self, x=None):
        super().__init__([("x", x)])
    
    def __repr__(self):
        return f"abs({repr(self.arguments['x'])})"

    def calculate(self, dmgr):
        x = self.arguments["x"].calculate(dmgr)
        return OrderedDict((k,np.abs(v)) for k,v in x.items())

class OpClip(Operator):
    def __init__(self, x=None, lb=None, ub=None):
        super().__init__([("x", x), ("lb", lb), ("ub", ub)])
    
    def __repr__(self):
        return f"clip({repr(self.arguments['x'])},{self.arguments['lb']},{self.arguments['ub']})"

    def calculate(self, dmgr):
        x = self.arguments["x"].calculate(dmgr)
        lb = float(self.arguments["lb"])
        ub = float(self.arguments["ub"])
        return OrderedDict((k,np.clip(v,lb,ub)) for k,v in x.items())

class OpDiff(Operator):
    def __init__(self, x=None, day=None):
        super().__init__([("x", x), ("day", day)])
    
    def __repr__(self):
        return f"diff({repr(self.arguments['x'])},{self.arguments['day']})"

    def calculate(self, dmgr):
        x = self.arguments["x"].calculate(dmgr)
        day = int(self.arguments["day"])
        keys = list(x.keys())
        nans = np.full(x[keys[0]].shape, np.nan)
        values = [nans.copy() for _ in range(day)] + [x[k2]-x[k1] for k1, k2 in zip(keys[:-day], keys[day:])]
        if len(keys) != len(values):
            raise ValueError(f"[OpDiff] keys and values do not match.\ndata length = {len(keys)}, delay = {day}")
        return OrderedDict(zip(keys, values))


#####################
# Const, Decay, ... #
#####################

class OpConst(Operator):
    def __init__(self, value=None, like=None,):
        super().__init__([("value", value), ("like", like)])
    
    def __repr__(self):
        return f"({repr(self.arguments['value'])})"

    def calculate(self, dmgr):
        x = self.arguments["like"].calculate(dmgr)
        value = float(self.arguments['value'])
        return OrderedDict((k, np.full(v.shape, value)) for k,v in x.items())

class OpFlip(Operator):
    def __init__(self, x=None,):
        super().__init__([("x", x)])
    
    def calculate(self, dmgr):
        x = self.arguments["x"].calculate(dmgr)
        return OrderedDict((k, -v) for k,v in x.items())

class OpNop(Operator):
    def __init__(self, x=None,):
        super().__init__([("x", x)])
    
    def __repr__(self):
        return f"nop({repr(self.arguments['x'])})"

    def calculate(self, dmgr):
        return self.arguments["x"].calculate(dmgr)

class OpDiscretize(Operator):
    def __init__(self, x=None, step=None, limit=None):
        super().__init__([("x", x), ("step", step), ("limit", limit)])

    def __repr__(self):
        return f"descretize({repr(self.arguments['x'])},{self.arguments['step']},{self.arguments['limit']})"

    def calculate(self, dmgr):
        x = self.arguments["x"].calculate(dmgr)
        step = float(self.arguments["step"])
        limit = float(self.arguments["limit"])
        return OrderedDict([(k, np.sign(v) * np.clip(np.floor(np.abs(v/step))*step, 0, limit)) for k,v in x.items()])



########################
### Rolling operators ###
########################

class RollingOperator(Operator):
    def __init__(self, x=None, a=None):
        super().__init__([("x", x), ("a", a)])
    
    def __repr__(self):
        return f"Rolling({repr(self.arguments['x'])},{self.arguments['a']})"

    @staticmethod
    def lookback_length(x, a):
        # specify data length for calculate_one
        raise NotImplementedError("Subclasses should implement this method")

    @staticmethod
    def calculate_one(df, a):
        raise NotImplementedError("Subclasses should implement this method")

    def calculate(self, dmgr):
        # convert lookback from intervals to days, and fetch
        x = self.arguments["x"].calculate(dmgr)
        a = int(self.arguments["a"])
        keys = list(x.keys())
        values = list(x.values())
        blocks = a // len(values[0])
        results = []
        for i, k in enumerate(keys):
            lookback_length = self.lookback_length(len(values[i]), a)
            lookback = np.concatenate([[np.nan] * lookback_length] + [values[ii] for ii in range(i-blocks-1, i+1) if ii >= 0])[-lookback_length:]
            df = self.calculate_one(pd.Series(lookback),a)
            result = df.values[-len(values[0]):]
            assert result.shape == values[i].shape
            results.append(result)
        return OrderedDict(zip(keys,results))
    

class OpZscore(RollingOperator):
    def __repr__(self):
        return f"zscore({repr(self.arguments['x'])},{self.arguments['a']})"

    @staticmethod
    def lookback_length(x, a):
        return x+a+1

    @staticmethod
    def calculate_one(df, a):
        return (df - (df.shift(1).rolling(window=a).mean())) / (df.shift(1).rolling(window=a).std().apply(lambda x: np.nan if (abs(x)<1e-6) else x))

class OpMean(RollingOperator):
    def __repr__(self):
        return f"mean({repr(self.arguments['x'])},{self.arguments['a']})"

    @staticmethod
    def lookback_length(x, a):
        return x + a

    @staticmethod
    def calculate_one(df, a):
        return df.rolling(window=a).mean()
    
class OpRank(RollingOperator):
    def __repr__(self):
        return f"rank({repr(self.arguments['x'])},{self.arguments['a']})"

    @staticmethod
    def lookback_length(x, a):
        return x + a

    @staticmethod
    def calculate_one(df, a):
        return (df.rolling(window=a).rank(ascending=True) - 1) / a
    

class OpStd(RollingOperator):
    def __repr__(self):
        return f"std({repr(self.arguments['x'])},{self.arguments['a']})"

    @staticmethod
    def lookback_length(x, a):
        return x + a

    @staticmethod
    def calculate_one(df, a):
        return df.rolling(window=a).std()

class OpSem(RollingOperator):
    # normalized standard deviation (by sqrt length)
    def __repr__(self):
        return f"sem({repr(self.arguments['x'])},{self.arguments['a']})"

    @staticmethod
    def lookback_length(x, a):
        return x + a

    @staticmethod
    def calculate_one(df, a):
        return df.rolling(window=a).sem()

class OpQuantile02(RollingOperator):
    def __repr__(self):
        return f"quantile02({repr(self.arguments['x'])},{self.arguments['a']})"

    @staticmethod
    def lookback_length(x, a):
        return x + a

    @staticmethod
    def calculate_one(df, a):
        return df.rolling(window=a).quantile(.2, interpolation='midpoint')

class OpQuantile05(RollingOperator):
    def __repr__(self):
        return f"quantile05({repr(self.arguments['x'])},{self.arguments['a']})"

    @staticmethod
    def lookback_length(x, a):
        return x + a

    @staticmethod
    def calculate_one(df, a):
        return df.rolling(window=a).quantile(.5, interpolation='midpoint')
    
class OpQuantile08(RollingOperator):
    def __repr__(self):
        return f"quantile08({repr(self.arguments['x'])},{self.arguments['a']})"

    @staticmethod
    def lookback_length(x, a):
        return x + a

    @staticmethod
    def calculate_one(df, a):
        return df.rolling(window=a).quantile(.8, interpolation='midpoint')
    
    
class OpMax(RollingOperator):
    def __repr__(self):
        return f"max({repr(self.arguments['x'])},{self.arguments['a']})"

    @staticmethod
    def lookback_length(x, a):
        return x + a

    @staticmethod
    def calculate_one(df, a):
        return df.rolling(window=a).max()
    
class OpMin(RollingOperator):
    def __repr__(self):
        return f"min({repr(self.arguments['x'])},{self.arguments['a']})"

    @staticmethod
    def lookback_length(x, a):
        return x + a

    @staticmethod
    def calculate_one(df, a):
        return df.rolling(window=a).min()
    
class OpSkew(RollingOperator):
    def __repr__(self):
        return f"skew({repr(self.arguments['x'])},{self.arguments['a']})"

    @staticmethod
    def lookback_length(x, a):
        return x + a

    @staticmethod
    def calculate_one(df, a):
        return df.rolling(window=a).skew()
    
class OpKurt(RollingOperator):
    def __repr__(self):
        return f"kurt({repr(self.arguments['x'])},{self.arguments['a']})"

    @staticmethod
    def lookback_length(x, a):
        return x + a

    @staticmethod
    def calculate_one(df, a):
        return df.rolling(window=a).kurt()
    
class OpExpdecay(RollingOperator):
    # Should it be rolling operator? Maybe we could simply aggregate all and do single ewm
    def __repr__(self):
        return f"ewm({repr(self.arguments['x'])},{self.arguments['a']})"

    @staticmethod
    def lookback_length(x, a):
        return x + a * 3

    @staticmethod
    def calculate_one(df, a):
        return df.ewm(halflife=3*a).mean()
    
class OpExpneut(RollingOperator):
    # Should it be rolling operator? Maybe we could simply aggregate all and do single ewm
    def __repr__(self):
        return f"expneut({repr(self.arguments['x'])},{self.arguments['a']})"

    @staticmethod
    def lookback_length(x, a):
        return x + a * 3

    @staticmethod
    def calculate_one(df, a):
        return df - df.ewm(halflife=3*a).mean()
    
#####################
# Special operators #
#####################

class OpData(Operator):
    def __init__(self, expr):
        super().__init__({"expr": expr})
    
    def __repr__(self):
        return f"'{self.arguments['expr']}'"
    
    def __int__(self):
        return int(self.arguments['expr'])
    
    def __float__(self):
        return float(self.arguments['expr'])

    def calculate(self, dmgr):
        return dmgr.get(self.arguments["expr"])

Operators = {k[2:].lower(): v for k, v in globals().items() if k.startswith("Op")}



# class OpArgmax(Operator):
# ffill
# clip