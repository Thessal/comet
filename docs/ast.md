# Abstract Syntax Tree (`ast.md`)

This document defines the internal representation (Rust types) for the Comet AST.

## 1. Top Level

```rust
pub struct Program {
    pub declarations: Vec<Declaration>,
}

pub enum Declaration {
    Type(TypeDecl),
    Struct(StructDecl),
    Enum(EnumDecl),
    Behavior(BehaviorDecl),
    Impl(ImplDecl),
    Function(FuncDecl),
    Flow(FlowDecl),
    Property(PropertyDecl),
}
```

## 2. Type Definitions

```rust
pub struct TypeDecl {
    pub name: Ident,
    pub parent: Ident,
    pub properties: Vec<Ident>,
}

pub struct StructDecl {
    pub name: Ident,
    pub fields: Vec<Field>,
}

pub struct Field {
    pub name: Ident,
    pub ty: TypeRef,
}
```

## 3. Logic Definitions (Behaviors & Impls)

```rust
pub struct BehaviorDecl {
    pub name: Ident,
    pub args: Vec<Ident>,
    pub return_type: Ident,
}

pub struct ImplDecl {
    pub name: Ident, // Unique name (e.g., "Ratio")
    pub behavior: Ident,
    pub args: Vec<Ident>, // e.g., ["A", "B"]
    pub constraints: Option<Expr>, // "where B is NonZero"
    pub body: Block,
}
```

## 4. Flow Logic

```rust
pub struct FlowDecl {
    pub name: Ident,
    pub body: Vec<FlowStmt>,
}

pub enum FlowStmt {
    Generator {
        target: Ident,
        source: Expr, // e.g. "Universe(Earnings)" or "Comparator(x, y)"
        constraints: Option<Expr>,
    },
    Assignment {
        target: Ident,
        expr: Expr,
    },
    Return(Expr),
}
```

## 5. Expressions

```rust
pub enum Expr {
    Literal(Literal),
    Identifier(Ident),
    BinaryOp { left: Box<Expr>, op: Op, right: Box<Expr> },
    Call { path: Path, args: Vec<Expr> },
    MemberAccess { target: Box<Expr>, field: Ident },
    PropertyCheck { target: Box<Expr>, property: Ident }, // "is NonZero"
}

pub enum Op {
    Add, Sub, Mul, Div, Eq, Neq, Lt, Gt, And, Or
}

pub struct Path {
    pub segments: Vec<Ident>, // e.g., ["Comparator", "compare"]
}
```
