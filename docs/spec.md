# Comet Language Specification

Comet is a domain-specific language for defining **Semantic Logical Flows** in quantitative finance. It focuses on **Meaning** (Properties, Relationships) rather than **Representation** (Structs, Pointers).

## Design Philosophy

-   **Logical Typing**: Types represent *what* data is (e.g., `Volume`), not just how it's stored.
-   **Semantic Properties**: Data carries properties (`NonZero`, `Stationary`, `USD`) that determine valid operations.
-   **Context-Driven Synthesis**: Operations are generated only when the semantic context (properties of inputs) allows it.

## Syntax Overview

### 1. Terminology: Quant vs. Code

To align with the "Semantic" design philosophy, we use specific terms:

| Concept | Replaces (CS Term) | Description |
| :--- | :--- | :--- |
| **Behavior** | `Trait` | Defines an abstract logical capability (e.g., `Comparator`). It is about *what* makes sense to do with data, not just what methods exist. |
| **Implementation** | `Impl` | A specific, grounded logic for a Behavior (e.g., `Ratio`, `Spread`). It represents a *valid hypothesis* or *model* for that behavior. |
| **Property** | `Constraint` / `Marker` | A semantic label (e.g., `NonZero`, `USD`). It describes the *meaning* of the data, enforcing categorical consistency. |

### 2. File Structure

Comet supports modularity via imports.

```comet
import "stdlib.cm"
import "data/universe.cm"
```

### 3. Semantic Attributes (Properties)

We define logic properties that can be attached to data.

```comet
// Logical Markers
Property NonZero
Property Stationary
Property Count
Property Monetary
```

### 2. Domain Types

Types are defined by their structure AND their semantic properties.

```comet
// Basic Representation (The "Container")
Struct Series {}

// Domain Concepts (The "Meaning")
Type Volume : Series derives { NonZero, Count }
Type Price  : Series derives { NonZero, Monetary }
Type Return : Series derives { Stationary }
```

### 3. Behaviors (Traits) & Logic

Behaviors are defined on **abstract concepts**, not just specific structs.

```comet
// Abstract Behavior
Behavior Comparator(A, B) -> Indicator

// Logic: Ratio
// "A Ratio comparison is valid if B is physically capable of being a denominator (NonZero)"
Implementation Ratio implements Comparator(A, B) 
where B is NonZero 
{
    return A / B
}

// Logic: Spread
// "A Spread comparison is valid if A and B share the same Unit"
Implementation Spread implements Comparator(A, B)
where A.Unit == B.Unit
{
    return A - B
}
```

## 5. Type-Driven Function Synthesis

The core of Comet is the **Flow**, where users describe a high-level intent, and the compiler synthesizes the valid mathematical operations based on the semantic properties.

### Complex Example: Event Driven Strategy

```comet
Flow EventDrivenStrategy {
    // 1. Inputs (Semantic Definition)
    // "Get me US Earnings and Volume data"
    ebit <- Universe(Earnings) where Unit is USD
    vol  <- Universe(Volume)   // inherently NonZero, Count
    
    // 2. Transformations
    days <- [21, 63]
    avg_vol <- MovingAverage(vol, days) // Inherits 'Count' and 'NonZero' from vol? Yes.

    // 3. Synthesis (The "Quant" Logic)
    // We request a Comparison. The compiler checks available Implementations.
    
    // Check: Can we use "Ratio"?
    // Logic: Is 'avg_vol' NonZero? 
    // Yes (derived from Volume). -> Valid.
    
    // Check: Can we use "Spread"?
    // Logic: Do 'vol' and 'avg_vol' share units?
    // Yes (both are Count). -> Valid.
    
    // Result: Both Ratio and Spread are generated.
    spike <- Comparator(vol, avg_vol)

    // 4. Trigger
    signal <- Trigger(ebit, spike, Condition.GreaterThan, 4.0)
    
    return signal
}
```

### Guarding Against Nonsense

If we tried to compare `Volume` and `Price` using `Spread`:
-   `Volume` is `Count`
-   `Price` is `Monetary`
-   Condition `A.Unit == B.Unit` fails.
-> **Spread** is not generated.

If we tried to compare `Volume` and `0` using `Ratio`:
-   `0` is not `NonZero`.
-   Condition `B is NonZero` fails.
-> **Ratio** is not generated.

This ensures that the generated strategies are **Categorically Consistent**.
