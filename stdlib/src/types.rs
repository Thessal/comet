use std::fmt::{self};
use tch::Tensor;

#[repr(usize)]
pub enum Signal {
    // Used to evaluate parameters in runtime
    Void,
    Float(Option<f64>),
    Int(Option<i64>),
    String(Option<String>),
    DataFrame(Option<Tensor>),
}

// tch::Tensor is conservatively !Send and !Sync in tch-rs.
// We implement them here because libtorch manages thread safety internally,
// and we need to store OperatorSpec templates (with Signal::DataFrame(None)) in a static OnceLock.
unsafe impl Send for Signal {}
unsafe impl Sync for Signal {}

impl Clone for Signal {
    fn clone(&self) -> Self {
        match self {
            Signal::Void => Signal::Void,
            Signal::Float(f) => Signal::Float(*f),
            Signal::Int(i) => Signal::Int(*i),
            Signal::String(s) => Signal::String(s.clone()),
            Signal::DataFrame(Some(t)) => Signal::DataFrame(Some(t.shallow_clone())),
            Signal::DataFrame(None) => Signal::DataFrame(None),
        }
    }
}

// Custom PartialEq to allow type comparisons in AST (ignoring tensor data if present)
impl PartialEq for Signal {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Signal::Void, Signal::Void) => true,
            (Signal::Float(a), Signal::Float(b)) => a == b,
            (Signal::Int(a), Signal::Int(b)) => a == b,
            (Signal::String(a), Signal::String(b)) => a == b,
            (Signal::DataFrame(a), Signal::DataFrame(b)) => {
                // For compiler type matching, we only care if both are None or Some.
                // We don't do deep tensor equality here.
                a.is_some() == b.is_some()
            }
            _ => false,
        }
    }
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
                Signal::DataFrame(Some(_df)) => "df[Some]".to_string(),
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
            Signal::DataFrame(_) => 4,
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
            Signal::DataFrame(None) => true,
            _ => false,
        }
    }
    // cast data into dataframe, for embedding
    pub fn to_dataframe(&self, size: (usize, usize), device: tch::Device) -> Tensor {
        let shape = [size.0 as i64, size.1 as i64];
        match self {
            Signal::Void => Tensor::zeros(&shape, (tch::Kind::Float, device)),
            Signal::Float(Some(x)) => Tensor::full(&shape, *x, (tch::Kind::Float, device)),
            Signal::Int(Some(i)) => Tensor::full(&shape, *i as f64, (tch::Kind::Float, device)),
            Signal::String(Some(_s)) => Tensor::zeros(&shape, (tch::Kind::Float, device)),
            Signal::DataFrame(Some(df)) => df.shallow_clone(),
            _ => panic!("Cannot cast empty signal to dataframe"),
        }
    }
}
