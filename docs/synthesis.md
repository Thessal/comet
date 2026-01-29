# Synthesis & Type Class Instance Resolution

The Comet synthesizer transforms a functional flow definition into an execution graph by resolving **Type Class Instances**. Unlike standard compilers that expect a single unique instance for a function call, Comet **iterates** through all valid instances to generate multiple strategy branches.

## 1. Type Class Constraints (Properties)

Properties (e.g., `NonZero`, `Stationary`) are now defined as **Type Classes**.

### 1.1 Sources
Instances ("Facts") are derived from the Type definitions and explicitly declared in the module.

```clean
// "Type Volume derives { NonZero }" becomes:
instance NonZero Volume
```

### 1.2 Constraint Checking (Pruning)
Functions (Behaviors) impose constraints on their type variables. Strategies are pruned if no instance satisfies the constraints.

*Example*: `compare :: a b -> c | NonZero a`

If `a` is a `Price` (which might not be `NonZero` in some contexts), and no `instance NonZero Price` exists, this path is rejected.

### 1.3 Newtype Wrapping (Ensures)
Functions that "ensure" a property now return a **Newtype Wrapper**.

```clean
// "ensures { Stationary }" becomes:
:: Stationary a = Stationary a
instance Stationary (Stationary a)

diff :: a -> Stationary a
```

## 2. Synthesis via Instance Resolution

The core synthesis loop happens at function application sites (nodes in the flow graph).

### The Algorithm
When encountering a function application `f a b`:
1.  Identify the Type Class associated with `f` (e.g., `Comparator`).
2.  Find **ALL** matching instances for the concrete types of `a` and `b`.
    *   *Note*: In standard Clean/Haskell, overlapping instances are an error or require specific selection. In Comet, **all** valid overlaps are distinct branches.
3.  Filter instances where constraints (e.g., `| NonZero a`) are not satisfied.
4.  For each remaining instance, fork the synthesis graph.

### Example

```clean
// User Code
ratio = compare vol1 vol2

// Available Instances
instance Comparator vol1 vol2 ... | NonZero vol2  // (Ratio Logic)
instance Comparator vol1 vol2 ... | SameUnit vol1 vol2 // (Spread Logic)
```

If `vol2` is `NonZero` AND they share units, **BOTH** branches are generated.
The `ratio` variable effectively becomes a superposition of `[RatioNode, SpreadNode]`.
