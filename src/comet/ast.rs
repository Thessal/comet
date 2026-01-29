
// Abstract Syntax Tree Definitions
// Based on docs/ast.md

pub type Ident = String;

#[derive(Debug, Clone, PartialEq)]
pub struct Program {
    pub module_name: Ident,
    pub imports: Vec<Import>,
    pub declarations: Vec<Declaration>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Import {
    pub path: String, // e.g. "Data.Universe"
}

#[derive(Debug, Clone, PartialEq)]
pub enum Declaration {
    Adt(AdtDecl),           // Algebraic Data Type (:: Type = ...)
    TypeSynonym(TypeSynDecl), // Type Synonym (:: Type :== ...)
    Class(ClassDecl),       // Type Class (class Name a ...)
    Instance(InstanceDecl), // Instance (instance Name Type ...)
    Function(FuncDecl),     // Function (name :: Type -> Type)
}

// 2. Type Definitions

#[derive(Debug, Clone, PartialEq)]
pub struct AdtDecl {
    pub name: Ident,
    pub type_vars: Vec<Ident>,
    pub constructors: Vec<Constructor>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct TypeSynDecl {
    pub name: Ident,
    pub type_vars: Vec<Ident>,
    pub target: TypeRef,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Constructor {
    pub name: Ident,
    pub index: Option<u32>, // For numbered fields if needed
    pub args: Vec<TypeRef>,
}

// 3. Logic Definitions (Classes & Instances)

#[derive(Debug, Clone, PartialEq)]
pub struct ClassDecl {
    pub name: Ident,
    pub type_vars: Vec<Ident>, // e.g. ["a", "b"]
    pub signature: Option<TypeRef>,    // :: a b -> c (The abstract function signature)
    // In Clean, classes can have members. For now, treating the class itself as the single function signature provider
    // or as a grouping. `docs/spec.md` says: `class Comparator a b c :: a b -> c`.
    // So the class *defines* a function.
}

#[derive(Debug, Clone, PartialEq)]
pub struct InstanceDecl {
    pub class_name: Ident,
    pub types: Vec<TypeRef>,   // e.g. [Volume, Volume, Series]
    pub constraints: Vec<Constraint>, // | SameUnit a b
    pub members: Vec<FuncDecl>, // where compare a b = ... (implementation)
}

#[derive(Debug, Clone, PartialEq)]
pub struct Constraint {
    pub class_name: Ident,
    pub type_args: Vec<Ident>, // e.g. ["a", "b"] for SameUnit a b
}

// 4. Function Logic

#[derive(Debug, Clone, PartialEq)]
pub struct FuncDecl {
    pub name: Ident,
    pub signature: Option<TypeRef>, 
    pub constraints: Vec<Constraint>, // New: | Normalized a
    pub args: Vec<Ident>, 
    pub body: Expr,
    pub where_block: Option<Vec<FuncDecl>>, 
}

// 5. Expressions

#[derive(Debug, Clone, PartialEq)]
pub enum Expr {
    Literal(Literal),
    Identifier(Ident),
    Application { func: Box<Expr>, args: Vec<Expr> }, // Function application (f x y)
    Let { bindings: Vec<Binding>, body: Box<Expr> },
    Case { target: Box<Expr>, arms: Vec<CaseArm> },
    Lambda { args: Vec<Ident>, body: Box<Expr> },
    BinaryOp { left: Box<Expr>, op: Op, right: Box<Expr> }, // Helper for common ops even if they are function calls
}

#[derive(Debug, Clone, PartialEq)]
pub struct Binding {
    pub name: Ident,
    pub expr: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub struct CaseArm {
    pub pattern: Pattern,
    pub expr: Expr,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    Literal(Literal),
    Constructor { name: Ident, args: Vec<Ident> }, // Simple destructuring
    Wildcard,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Op {
    Add, Sub, Mul, Div, Eq, Neq, Lt, Gt, And, Or
}

// 6. Types

#[derive(Debug, Clone, PartialEq)]
pub enum TypeRef {
    Concrete(Ident),
    Variable(Ident),
    Application(Box<TypeRef>, Vec<TypeRef>), // List a, Tree (Int, a)
    Function(Vec<TypeRef>, Box<TypeRef>),    // a -> b -> c
}

#[derive(Debug, Clone, PartialEq)]
pub enum Literal {
    Integer(i64),
    Float(f64),
    String(String),
    Boolean(bool),
}
