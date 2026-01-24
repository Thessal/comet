# Grammar Specification (`parse.md`)

This document defines the EBNF grammar for Comet.

## 1. Top Level

```ebnf
Program         ::= (Declaration)*
Declaration     ::= ImportDecl | TypeDecl | StructDecl | EnumDecl | BehaviorDecl | ImplDecl | FlowDecl | FuncDecl | PropertyDecl
```

## 2. Declarations

```ebnf
PropertyDecl    ::= "Property" Identifier

TypeDecl        ::= "Type" Identifier ":" Identifier "derives" "{" PropertyList "}"
StructDecl      ::= "Struct" Identifier "{" FieldList "}"
EnumDecl        ::= "Enum" Identifier "{" EnumVariantList "}"

BehaviorDecl    ::= "Behavior" Identifier "(" ArgList ")" "->" Identifier
ImplDecl        ::= "Implementation" Identifier "implements" Identifier "(" ArgList ")" (WhereClause)? "{" Block "}"

FuncDecl        ::= "fn" Identifier "(" ParamList ")" "->" Identifier (ConstraintList)? "{" Block "}"

FlowDecl        ::= "Flow" Identifier "{" FlowStmtList "}"
```

## 3. Flow Statements

```ebnf
FlowStmtList    ::= (FlowStmt)*
FlowStmt        ::= GeneratorStmt | AssignmentStmt | ExprStmt | ReturnStmt

GeneratorStmt   ::= Identifier "<-" Expr (WhereClause)?
AssignmentStmt  ::= Identifier "=" Expr (WhereClause)?
ReturnStmt      ::= "return" Expr
```

## 4. Expressions

```ebnf
Expr            ::= OrExpr
OrExpr          ::= AndExpr ("||" AndExpr)*
AndExpr         ::= EqExpr ("&&" EqExpr)*
EqExpr          ::= RelExpr (("==" | "!=") RelExpr)*
RelExpr         ::= AddExpr (("<" | ">" | "<=" | ">=") AddExpr)*
AddExpr         ::= MulExpr (("+" | "-") MulExpr)*
MulExpr         ::= UnaryExpr (("*" | "/") UnaryExpr)*
UnaryExpr       ::= ("-" | "!")? Atom
Atom            ::= Literal | Identifier | CallExpr | MemberExpr | ParenExpr | ListLiteral

CallExpr        ::= Path "(" ArgValues ")"
Path            ::= Identifier ("::" Identifier)*
MemberExpr      ::= Atom "." Identifier
ParenExpr       ::= "(" Expr ")"
ListLiteral     ::= "[" (Expr ("," Expr)*)? "]"
```

## 5. Primitives
```ebnf
Identifier      ::= [a-zA-Z_][a-zA-Z0-9_]*
Literal         ::= Integer | Float | String | Boolean
PropertyList    ::= Identifier ("," Identifier)*
FieldList       ::= (Identifier ":" TypeName)*
WhereClause     ::= "where" Expr
```
