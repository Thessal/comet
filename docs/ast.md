# Abstract Syntax Tree (`ast.md`)

This document defines the internal representation (Rust types) for the Comet AST.

## 1. Top Level

```rust
pub struct Program {
    pub declarations: Vec<Declaration>,
}

// INCONSISTENT: Missing enum definition and Implementation variant
// pub enum Declaration {
//     Type(TypeDecl),
//     Behavior(BehaviorDecl),
//     Function(FuncDecl),
//     Flow(FlowDecl),
// }
```

## 2. Type Definitions

```rust
pub struct TypeDecl {
    pub name: Ident,
    pub constraints: Vec<Ident>,
}
```

## 3. Logic Definitions (Behaviors & Impls)

pub struct BehaviorDecl {
    pub name: Ident,
    pub args: Vec<Param>, 
    pub return_type: Ident,
}

pub struct Param {
    pub name: Ident,
    pub ty: Vec<Ident>, // Type constraints
}

pub struct FuncDecl {
    pub name: Ident,
    pub args: Vec<Param>,
    pub return_type: Ident,
    pub body: Block,
}

## 4. Flow Logic

// INCONSISTENT SYNTAX: Flow should be a block of statements based on examples
// pub struct FlowDecl {
//     pub name: Ident,
//     pub expr: Expr,
// }

## 5. Expressions

pub enum Expr {
    Literal(Literal),
    Identifier(Ident),
    Call { path: Path, args: Vec<Expr> },
    Paren(Box<Expr>),
}

pub struct Path {
    pub segments: Vec<Ident>,
}
```
