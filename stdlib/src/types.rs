#[derive(Debug, Clone, PartialEq)]
#[repr(usize)]
pub enum Signal {
    // Used to evaluate parameters in runtime
    Void,
    Float(Option<f64>),
    Int(Option<i64>),
    String(Option<String>),
    Vector(Option<Vec<f64>>),
    DataFrame(Option<Vec<Vec<f64>>>),
}

impl Into<usize> for Signal {
    fn into(self) -> usize {
        match self {
            Signal::Void => 0,
            Signal::Float(_) => 1,
            Signal::Int(_) => 2,
            Signal::String(_) => 3,
            Signal::Vector(_) => 4,
            Signal::DataFrame(_) => 5,
        }
    }
}

impl Signal {
    pub fn is_none(&self) -> bool {
        match self {
            Signal::Void => false,
            Signal::Float(None) => true,
            Signal::Int(None) => true,
            Signal::String(None) => true,
            Signal::Vector(None) => true,
            Signal::DataFrame(None) => true,
            _ => false,
        }
    }
}

// Test
// assert_eq!(TypeDecl[TypeDecl::DataFrame as usize], TypeDecl::DataFrame);
// std::mem::discriminant(d) == std::mem::discriminant(i)
