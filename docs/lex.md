# Lexical Specification (`lex.md`)

This document defines the lexical structure of the Comet language.

## 1. Keywords

Keywords are reserved identifiers.

### Structural Keywords
-   `Type`
-   `Struct`
-   `Enum`
-   `Flow`
-   `Behavior` (replaces Trait)
-   `Implementation` (replaces Impl)
-   `Property`
-   `fn`

### Semantic Keywords
-   `derives` (used in Type definitions)
-   `implements` (used in Behavior definitions)
-   `where` (constraints)
-   `is` (property check: `where B is NonZero`)

### Logic/Control Flow
-   `return`
-   `let` (standard binding, though discouraged in Flow for combinatorial vars)

## 2. Operators & Punctuation

-   `<-` : Generator assignment (in Flow)
-   `::` : Path separator (e.g., `Comparator::compare`)
-   `->` : Return type arrow
-   `=>` : Match arrow (if used in expressions)
-   `.` : Member access
-   `:` : Type annotation
-   `=` : Assignment (Expression)
-   `==`, `!=`, `<`, `>`, `<=`, `>=` : Comparison
-   `+`, `-`, `*`, `/` : Arithmetic
-   `{`, `}` : Block delimiters
-   `(`, `)` : Parentheses
-   `[`, `]` : List literals
-   `,` : Separator

## 3. Literals

-   **Identifiers**: `[a-zA-Z_][a-zA-Z0-9_]*`
-   **Strings**: Double-quoted `"string"`
-   **Integers**: `123`
-   **Floats**: `123.45`
-   **Booleans**: `true`, `false`

## 4. Comments

-   Line comments: `// ...`
-   Block comments: `/* ... */`
