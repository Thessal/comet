# Comet Language Specification

Comet is a domain-specific language for synthesize **set** of functions in quantitative finance. 

Comet is a **declarative** language, which means that the user only needs to specify the **intent** of the function, and the compiler will generate the actual implementation.

We exploit type matching similar to functional programming, and automatically synthesize all possible code that can be correctly compiled.

Comet is **strongly typed**.

There are three callable objects in comet : **Behaviors**, **Flows**, **Functions**, and three layer of concepts : **Types**, **Categories**, **Constraints**.

Calling a Behavior or Function requires **named** arguments set.

## Design Philosophy

- **Types** 
    - Types are used to specify how the data is stored in the memory.
    - All data in comet are stored as a **Series**, which is updated for each time step in rolling manner.
    - Type keywords specify how the data is stored in the memory. Comet supports the following types.
        - `Series` : single f64 is stored for each time step.
        - `DataFrame` : fixed size array of f64 is assigned for each time step. (with flattened indexing)
        - `Matrix` : fixed 2d array of f64 is assigned for each time step.
        - `Vector` : Iliffe vector. Used for variable length data such as string.
        - Literal primitives: `String`, `Int`, `Float`, `Bool`. Supported for scalar arguments.
    - Depending on the operator / behavior, there are compatible types. For example,
        - `Series` data can be "added" with another `DataFrame` data.

- **Categories**
    - Categories specify the semantic category of the data, not like computer science, but like mathematics.
    - Categories specify which behaviors can be 'glued' together, and which cannot.
    - Categories are defined in the fly.
    - For example,
        - `Nonezero` data can be a divisor of another data.
        - `Monetary` data can be "compared" with another `Monetary` data.
    - Categories can be represented in combinatorial way. List representation is a parsed form of combinatorial representation.
        - Combinatorial representation : `( Nonzero | Monetary (Indicator) )`
            - Category with no union can be written without parenthesis : For example `Monetary Indicator` is identical to `( Monetary Indicator )`
        - List representation (parsed) : `[Nonzero, {Monetary, Indicator}]`
    - Rules of combinatoric expansion
        - Addition : (LHS) (RHS) 
            - When addition is expanded, RHS is appended to each type that matches LHS.
            - `Monetary Indicator` can be expanded to `[{Monetary, Indicator}]`
            - `( Nonzero | Monetary Indicator | Monetary)` can be expanded to `[Nonzero, {Monetary, Indicator}, Monetary]`
            - `( Nonzero | Monetary ) ( Indicator | Finite )` can be expanded to `[{Nonzero, Indicator}, {Nonzero, Finite}, {Monetary, Indicator}, {Monetary, Finite}]`
            - Same type added is removed when expanded : `( A A ) == A`
        - Commutative Addition : 
            - `{Monetary, Indicator}` is identical to `{Indicator, Monetary}`
        - Union : (LHS) | (RHS)
            - When union is expanded, All type that matches RHS is appended to each type that matches LHS. Duplicates are removed.
            - `( A | B ) | ( C | D )` can be expanded to `[A, B, C, D]`
            - `A C | A C | A D` can be expanded to `[A C, A D]`
        - Subtract : (LHS) - (RHS) 
            - When subtraction is expanded, patterns that matches RHS is removed from LHS.
            - `( Monetary | Indicator ) - Indicator` is expanded to `[Monetary]`
            - `( Monetary | Indicator ) - (Monetary Nonzero)` can be expanded to `[Monetary, Indicator]` because `Monetary` nor `Nonzero` are not matched by `{Monetary, Nonzero}`.
            - `( Monetary | Indicator ) Nonzero - ( Monetary Nonzero )` can be expanded to `{Indicator Nonzero}`.

- **Constraints**
    - Constraints specify the input and output parameters of behaviors, by combining types and categories.
    - Constraints can be **matched** to determine valid types, like a pattern matching in functional programming.
    - Constraints(constraint set) are set of constraint.
    - Each constraint is a composed of a single type, followed by zero or more categories.
        - `<type name> ( <combinatorial representations of categories> )`
        - `<type name> <category representation with set size is one>`
    - Matching : 
        - Single type can be matched to a constraint, when expansion of the constraint includes the type.
            - output `Series Nonzero` can be used for `Series` or `Series Nonzero` arguments.  
        - Constraints can be matched to a constraint, when expansion of the constraint includes the constraint.
            - `Series Nonzero` or `Series Monetary` output can be used as `Series (Nonzero | Monetary)` arguments.
            - `Series Nonzero` output cannot be used as `DataFrame (Nonzero | Monetary Nonzero)` argument because "DataFrame" is not matched by `Series`.
            - `Series Nonzero` output cannot be used as `Series Monetary Nonzero` argument because "Nonzero" is not matched by "Monetary Nonzero".
            - `Series` output cannot be used as `Series Monetary` argument because no category is not "Monetary" category.
            - `Series Monetary` output can be used as `Series` argument because Monetary series is a series.
    - Assignment : 
        - Constraint variable '(symbol) can capture a constraints e.g. 'a , 'b  etc. 
        - Constraint can be stored to the constraint variable and recovered from the variable.

- **Behaviors**
    - Behaviors are similar to functions, but multiple possible functions can be matched for each behavior.
    - Behaviors define the "Interface" or "Trait" of a flow.
    - Behavior is a mapping from input type constraints to output type constraints.
    - To prevent infinite loop, behaviors are not allowed to be recursive, and only one function can be matched for each behavior.
    - Behavior is defined by the following syntax : 
        - `Behavior <behavior name> (<parameter name> : <input constraints>, ...) -> <output constraints>`
    - Example : 
        - `Behavior compare (signal: (DataFrame | Series) 'a, reference: (DataFrame | Series) Finite Positive) -> ('a Finite Indicator)`
            - Constraint variable 'a or 'b etc can be used to capture a constraint set.
            - It means that Compare is a mapping from A, B into (a Finite). If A is DataFrame, the result is also DataFrame.
        - Behavior can be fully expanded into the following list : 
            - `compare(signal=DataFrame, reference=DataFrame Finite Positive) -> DataFrame Finite Indicator`
            - `compare(signal=Series, reference=Series Finite Positive) -> Series Finite Indicator`
        - These example functions with following types are valid for Compare:
            - `Fn divide(signal:DataFrame, reference:DataFrame Finite Positive) -> DataFrame Finite Indicator`
            - `Fn diff(signal:DataFrame Finite Positive SomeOtherType, reference:DataFrame Finite Positive SomeOtherType) -> DataFrame Finite Indicator`
            - `Fn divide_1d(signal:Series, reference:Series Finite Positive) -> Series Finite Indicator`
    - "depth" parameter is reserved, and cannot be used in behavior definition.
    - "depth" parameter is 1 by default, and can be specified in the behavior call. It is used to limit the depth of function calls in the behavior when the behavior is expanded.

- **Flows**
    - Flow is a list of behaviors, that forms a path of transformations that generates data with type constraints.
    - Flow is defined by the following syntax : 
        - `Flow <flow name> { <statements> } -> <output constraints>`
        - Statements are separated by newlines.
        - Each statement in the flow are either assignment or return, of behavior or function calls.
    - Last statement is returned.

    - Statement in a Flow can be defined by chained function.
        - 
        ```
            Flow volume_spike { 
                Compare(signal=data("volume"), reference=HistoricalVolume(signal=data("volume"), lookback=days()), depth=1) 
            }
        ```
        - Given that days is a behavior `Behavior days() -> Days ("21" | "63")`, `data` is a stdlib function `Fn data(symbol: String) -> DataFrame`, `HistoricalVolume` is a behavior `Behavior HistoricalVolume(signal: DataFrame, lookback: Days) -> DataFrame`, `Compare` is a behavior `Behavior Compare(signal: DataFrame, reference: DataFrame Finite Positive) -> DataFrame Finite Indicator`
        - Compare, HistoricalVolume, days are behaviors or functions, so parenthesis is added.
        - The `volume_spike` flow defined above, can be matched to a chain of functions, which can be translated into LLVM code.
            - `rank_diff(a=data("volume"), b=ts_mean(a=data("volume"), period=21))`
            - `rank_diff(a=data("volume"), b=ts_mean(a=data("volume"), period=63))`
            - `divide(a=data("volume"), b=ts_mean(a=data("volume"), period=21))`
            - `divide(a=data("volume"), b=ts_mean(a=data("volume"), period=63))`
        - Each functions match each elements of chains, and keeps the composition structure of the chained behavior. 
    - For readability, symbols can be used to chain functions, similar to assignment or definitions.
        - 
        ```
            Flow volume_spike { 
                volume = data("volume")
                variousdays = days()
                Compare(signal=volume, reference=HistoricalVolume(signal=volume, lookback=variousdays), depth=1) 
            }
        ```
        - This code is translated into the first example internally.
        - `volume` and `variousdays` are symbols that capture the result of `data("volume")` and `days()` respectively.
        - However, they have only local scope in the flow, and does not mean storing data in memory.
        - The effect of assignment is to replace the symbol with the result of the function call. So, `days()` is called twice in the above example. It is the user's responsibility to avoid function call explosion.
    - Last flow in the code should return DataFrame, which are the final result of the flow.
    
- **Functions**
    - Functions map that receives a list of concepts and returns a concept.
    - Functions are defined in stdlib, and comet do not support user-defined functions.
    - To map stdlib functions into comet symbols, we use `Fn` keyword, which is similar concept with C header.
        - `Fn <function name> ( <parameter name> : <input constraints>, ... ) -> <output constraints>`
        - `Fn Ratio ( signal: DataFrame, reference: DataFrame Positive ) -> (DataFrame Finite)`
    - It exists only to help type checking and code generation.
    - Function can be matched to a behavior all of the following conditions are met: 
        - Input keywords are valid for the behavior.
        - Input types are valid for the behavior.
        - Output type is valid for the behavior.
    - Example: 
        - `Fn Ratio ( signal: DataFrame, reference: DataFrame Positive ) -> (DataFrame Finite)` can be matched to a behavior `behavior Compare (signal: (DataFrame | Series) 'a, reference: (DataFrame | Series) 'b Finite Positive) -> ('a Finite)`

## File Structure

Comet supports modularity via imports.

```comet
import "stdlib.cm"
import "data/price.cm"
```