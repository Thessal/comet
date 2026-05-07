use std::fmt::{self, Write};

#[derive(Clone, PartialEq)]
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

impl fmt::Debug for Signal {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Signal::Void => "void".to_string(),
                Signal::Float(Some(x)) => format!("f[{:.3}]", x),
                Signal::Int(Some(i)) => format!("i[{:}]", i),
                Signal::String(Some(s)) => format!("s[{:}]", s),
                Signal::Vector(Some(_v)) => "vec[Some]".to_string(),
                Signal::DataFrame(Some(_df)) => "df[Some]".to_string(),
                Signal::Vector(None) => "vec[None]".to_string(),
                Signal::DataFrame(None) => "df[None]".to_string(),
                Signal::Float(None) => "f[None]".to_string(),
                Signal::Int(None) => "i[None]".to_string(),
                Signal::String(None) => "s[None]".to_string(),
            }
        )
    }
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
    // cast data into dataframe, for embedding
    pub fn to_dataframe(&self, size: (usize, usize)) -> Vec<Vec<f64>> {
        match self {
            Signal::Void => vec![vec![0.0; size.1]; size.0],
            Signal::Float(Some(x)) => vec![vec![x.clone(); size.1]; size.0],
            Signal::Int(Some(i)) => vec![vec![i.clone() as f64; size.1]; size.0],
            Signal::String(Some(_s)) => vec![vec![0.0; size.1]; size.0],
            Signal::Vector(Some(_v)) => panic!(),
            Signal::DataFrame(Some(df)) => {
                // assert_eq!(df.len(), size.0);
                // assert!(df.iter().map(|x| x.len()).all(|x| x == size.1));
                df.clone() // is it deep copy?
            }
            _ => panic!(),
        }
    }
}

// Test
// assert_eq!(TypeDecl[TypeDecl::DataFrame as usize], TypeDecl::DataFrame);
// std::mem::discriminant(d) == std::mem::discriminant(i)
