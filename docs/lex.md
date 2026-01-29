# Lexical Specification (`lex.md`)

This document defines the lexical structure of the Comet language.

## 1. Keywords

Keywords are reserved identifiers.

### Structural Keywords
-   `type`
-   `behavior`
-   `flow`
-   `fn`

### Logic/Control Flow
-   `return`

## 2. Operators & Punctuation

-   `::` : Path separator (e.g., `Comparator::compare`)
-   `->` : Return type arrow
-   `=>` : Match arrow (if used in expressions)
-   `.` : Member access
-   `:` : Type annotation
-   `=` : Assignment (Expression)
-   `{`, `}` : Block delimiters
-   `(`, `)` : Parentheses
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
