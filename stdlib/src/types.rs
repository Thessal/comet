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

// Test
// assert_eq!(TypeDecl[TypeDecl::DataFrame as usize], TypeDecl::DataFrame);
