## 1. Top Level

```rust
pub struct Program {
    pub declarations: Vec<Declaration>,
}
```

```rust
pub enum Declaration {
    Behavior(BehaviorDecl),
    Function(FuncDecl),
    Flow(FlowDecl),
}
```

import statement is simply copy-paste of the imported module into the current module, which is handled before parsing.

## 2. Constraints Definitions, that can be expanded or matched

```rust
pub enum TypeDecl {
    Series,
    DataFrame,
    Matrix,
    Vector,
    String,
    Int,
    Float,
    Bool,
}
```

```rust
// A parsed Combinatorial representation of categories
pub struct CategorySetDecl {
    pub name: Option<Ident>, // e.g. 'a, 'b
    pub categories: Vec<Ident>,
}

// A constraint combines a Type and a CategorySet
pub struct ConstraintDecl {
    pub base_type: TypeDecl,
    pub category_set: CategorySetDecl,
}
```

## 3. Logic Definitions (Behaviors & Impls)

```rust
pub struct Param {
    pub name: Ident,
    pub constraint: ConstraintDecl
}
```

```rust
pub struct BehaviorDecl {
    pub name: Ident,
    pub args: Vec<Param>, 
    pub return_constraint: ConstraintDecl,
    pub depth: u32,
}

pub struct FuncDecl { // Used only to map stdlib functions
    pub name: Ident,
    pub args: Vec<Param>,
    pub return_constraint: ConstraintDecl,
}

pub struct FlowDecl {
    pub name: Ident,
    pub return_constraint: ConstraintDecl,
    pub body: Block,
}
```

## 4. Logic Calls (TODO: How can we avoid infinite loop...)

```rust
pub enum Expr {
    Symbol(Ident),
    Call(Call),
    Literal(Literal),
}

pub enum Call{
    Behavior(BehaviorCall),
    Function(FuncCall),
}

pub struct BehaviorCall{
    pub identifier: Ident,
    pub arguments: Vec<(Ident, Expr)>,
}

pub struct FuncCall{
    pub identifier: Ident,
    pub arguments: Vec<(Ident, Expr)>,
}

pub enum Literal{
    Integer(i64),
    Float(f64),
    String(String),
}

```

## 5. Flow body

```rust
pub enum FlowStmt {
    Expr(Expr),
    Assignment {
        target: String,
        expr: Expr,
    },
}
```

Assignments within a Flow are resolved and recursively substituted into the final target expressions using `comet::synthesis::substitute_expr` before synthesis.
