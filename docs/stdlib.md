# Standard Library (Clean-like)

## Core Behaviors (Type Classes)

### Comparator
Defines binary comparison logic.

```clean
class Comparator a b c :: a b -> c
```

**Standard Instances**:
*   `Ratio`: `a / b` (Requires `SameRepresentation a b` and `NonZero b`)
*   `Spread`: `a - b` (Requires `SameRepresentation a b` and `SameUnit a b`)

### Normalizer
Defines unary normalization logic.

```clean
class Normalizer a b :: a -> b
```

## Meta-Properties (Multi-Param Classes)

These classes enforce relationships between types.

```clean
// Skeleton: Checks if 'a' and 'b' are physically compatible
class SameRepresentation a b

// Skeleton: Checks if 'a' and 'b' share the same semantic unit (e.g. USD)
class SameUnit a b
```

## Core Properties (Type Classes)

```clean
class NonZero a
class Stationary a
class Ranged a
class Count a
class Monetary a
```

## Built-in Operators

Binary operators with broadcasting support.

```clean
class Div a b c :: a b -> c

// Scalar Division
instance Div (Constant Real) (Constant Real) (Constant Real)
    where div(a, b) = a / b

instance Div (Constant Int) (Constant Int) (Constant Int)
    where div(a, b) = a / b

// Series Division (Broadcasting)
instance Div (Series a) (Constant a) (Series a)
    where div(a, b) = a / b

instance Div (Series a) (Series a) (Series a)
    where div(a, b) = a / b

// DataFrame Division
instance Div (DataFrame a) (DataFrame a) (DataFrame a)
    where div(a, b) = a / b

instance Div (DataFrame a) (Series a) (DataFrame a) // Axis alignment implied
    where div(a, b) = a / b
```
