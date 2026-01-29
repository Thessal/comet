# Grammar Specification (`parse.md`)

This document defines the EBNF grammar for Comet.

## 1. Top Level

```ebnf
Program         ::= (Declaration)*
Declaration     ::= ImportDecl | TypeDecl | BehaviorDecl | FlowDecl | FuncDecl
```

## 2. Declarations

```ebnf
TypeDecl        ::= "type" Identifier (":" TypeRef*)?
BehaviorDecl    ::= "behavior" Identifier "(" ParamList ")" "->" TypeRef
FuncDecl        ::= "fn" Identifier "(" ParamList ")" "->" TypeRef "{" Block "}"
FlowDecl        ::= "flow" Identifier "=" Expr

TypeRef         ::= Identifier
ParamList       ::= (Param ("," Param)*)?
Param           ::= Identifier ":" TypeList
TypeList        ::= Identifier (Identifier)*
```


## 4. Expressions

```ebnf
Expr            ::= Atom | CallExpr
Atom            ::= Literal | Identifier | ParenExpr

CallExpr        ::= Identifier "(" ArgValues ")"
ArgValues       ::= (Arg ("," Arg)*)?
Arg             ::= (Identifier "=")? Expr

ParenExpr       ::= "(" Expr ")"
```

## 5. Primitives
```ebnf
Identifier      ::= [a-zA-Z_][a-zA-Z0-9_]*
Literal         ::= Integer | Float | String | Boolean
PropertyList    ::= Identifier ("," Identifier)*
FieldList       ::= (Identifier ":" TypeName)*
WhereClause     ::= "where" Expr
```
