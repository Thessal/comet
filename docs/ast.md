# Abstract Syntax Tree (`ast.md`)

This document defines the internal representation (Rust types) for the Comet AST (Clean-like).

## 1. Top Level

```rust
pub struct Program {
    pub module_name: Ident,
    pub imports: Vec<Import>,
    pub declarations: Vec<Declaration>,
}

pub enum Declaration {
    Adt(AdtDecl),           // Algebraic Data Type (:: Type = ...)
    TypeSynonym(TypeSynDecl), // Type Synonym (:: Type :== ...)
    Class(ClassDecl),       // Type Class (class Name a ...)
    Instance(InstanceDecl), // Instance (instance Name Type ...)
    Function(FuncDecl),     // Function (name :: Type -> Type)
}
```

## 2. Type Definitions

```rust
pub struct AdtDecl {
    pub name: Ident,
    pub type_vars: Vec<Ident>,
    pub constructors: Vec<Constructor>,
}

pub struct Constructor {
    pub name: Ident,
    pub index: Option<u32>, // For numbered fields
    pub args: Vec<TypeRef>,
}
```

## 3. Logic Definitions (Classes & Instances)

```rust
pub struct ClassDecl {
    pub name: Ident,
    pub type_vars: Vec<Ident>, // e.g. ["a", "b"]
    pub signature: TypeRef,    // :: a b -> c
}

pub struct InstanceDecl {
    pub class_name: Ident,
    pub types: Vec<TypeRef>,   // e.g. [Volume, Volume, Series]
    pub constraints: Vec<Constraint>, // | SameUnit a b
    pub members: Vec<FuncDecl>, // where compare a b = ...
}

pub struct Constraint {
    pub class_name: Ident,
    pub type_args: Vec<Ident>, // e.g. ["a", "b"] for SameUnit a b
}
```

## 4. Function Logic

```rust
pub struct FuncDecl {
    pub name: Ident,
    pub body: Expr,
    pub where_block: Option<Vec<FuncDecl>>, // Local definitions
}
```

## 5. Expressions

```rust
pub enum Expr {
    Literal(Literal),
    Identifier(Ident),
    Application { func: Box<Expr>, args: Vec<Expr> }, // Function application (f x y)
    Let { bindings: Vec<Binding>, body: Box<Expr> },
    Case { target: Box<Expr>, arms: Vec<CaseArm> },
    Lambda { args: Vec<Ident>, body: Box<Expr> },
}
```

## 6. Types

```rust
pub enum TypeRef {
    Concrete(Ident),
    Variable(Ident),
    Application(Box<TypeRef>, Vec<TypeRef>), // List a, Tree (Int, a)
    Function(Vec<TypeRef>, Box<TypeRef>),    // a -> b -> c
}
```
