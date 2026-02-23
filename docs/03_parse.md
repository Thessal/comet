# Grammar Specification (`parse.md`)

This document defines the EBNF grammar for Comet.

## 1. Top Level

```ebnf
Program         ::= (Declaration)*
Declaration     ::= BehaviorDecl | FlowDecl | FuncDecl
```

## 2. Declarations

```ebnf
BehaviorDecl    ::= "Behavior" Identifier "(" ParamList ")" "->" TypeRef
FuncDecl        ::= "Fn" Identifier "(" ParamList ")" "->" TypeRef
FlowDecl        ::= "Flow" Identifier "{" Block "}" "->" TypeRef
Block           ::= (Statement)*
Statement       ::= Assignment | Return
Assignment      ::= Identifier "=" Expr
Return          ::= Expr

TypeRef         ::= Constraint
ParamList       ::= (Param ("," Param)*)?
Param           ::= Identifier ":" Constraint

// A constraint combines a single type, followed by optional categories
Types           ::= "Series" | "DataFrame" | "Matrix" | "Vector"
Constraint      ::= Types (CategoryExpr)?
// Categories can be matched or expanded into list representation during parsing
CategoryExpr    ::= CategoryTerm (("-" | "|") CategoryTerm)*
CategoryTerm    ::= Identifier | "'" Identifier | "(" CategoryExpr ")" | CategoryTerm Identifier```


## 4. Expressions

```ebnf
Expr            ::= Atom | CallExpr
Atom            ::= Literal | Identifier | ParenExpr

CallExpr        ::= Identifier "(" ArgValues ")"
ArgValues       ::= (Arg ("," Arg)*)?
Arg             ::= Identifier "=" Expr

ParenExpr       ::= "(" Expr ")"
```

## 5. Primitives
```ebnf
Identifier      ::= "'"? [a-zA-Z_][a-zA-Z0-9_]*
Literal         ::= Integer | Float | String
```
