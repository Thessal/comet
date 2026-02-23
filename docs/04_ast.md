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

## 2. Constraints Definitions, that can be expanded or matched

```rust
pub enum TypeDecl {
    Series,
    DataFrame,
    Matrix,
    Vector,
}
```

```rust
pub struct ConstraintSetDecl {
    pub name: Ident,
    pub constraints: Vec<Ident>,
}
```

## 3. Logic Definitions (Behaviors & Impls)

```rust
pub struct Param {
    pub name: Ident,
    pub constraints: ConstraintSetDecl
}
```

```rust
pub struct BehaviorDecl {
    pub name: Ident,
    pub args: Vec<Param>, 
    pub return_constraint: ConstraintSetDecl,
}

pub struct FuncDecl { // Used only to map stdlib functions
    pub name: Ident,
    pub args: Vec<Param>,
    pub return_constraint: ConstraintSetDecl,
}

pub struct FlowDecl {
    pub name: Ident,
    pub return_constraint: ConstraintSetDecl,
    pub body: Block,
}
```

## 4. Logic Calls (TODO: How can we avoid infinite loop...)

```rust
pub enum Expr {
    Symbol(Ident),
    Call(Call),
}

pub enum Call{
    Behavior(BehaviorCall),
    Function(FuncCall),
}

pub enum BehaviorCall{
    Identifier(Ident),
    Arguments(Vec<Expr>),
}

pub enum FuncCall{
    Identifier(Ident),
    Arguments(Vec<Expr>),
}

```

## 5. Flow body

```rust
pub struct Block {
    pub statements: Vec<Statement>,
}

pub enum Statement {
    Assignment(Assignment),
    Return(Return),
}

pub struct Assignment {
    pub name: Ident,
    pub expr: Expr,
}

pub struct Return {
    pub expr: Expr,
}
```
