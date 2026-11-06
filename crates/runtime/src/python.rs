use crate::runtime::StackEnv;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyclass(name = "StackEnv", unsendable)]
pub struct PyStackEnv {
    pub inner: StackEnv,
}

#[pymethods]
impl PyStackEnv {
    #[new]
    #[pyo3(signature = (filename, use_cuda=false, cache_capacity=300, normalize_book=true, max_weight=Some(0.05)))]
    pub fn new(
        filename: &str,
        use_cuda: bool,
        cache_capacity: usize,
        normalize_book: bool,
        max_weight: Option<f64>,
    ) -> Self {
        Self {
            inner: StackEnv::from_file(
                filename,
                use_cuda,
                cache_capacity,
                normalize_book,
                max_weight,
            ),
        }
    }

    #[getter]
    pub fn vocab_size(&self) -> usize {
        self.inner.action_space.size()
    }

    #[getter]
    pub fn rows(&self) -> i64 {
        self.inner.rows
    }

    #[getter]
    pub fn cols(&self) -> i64 {
        self.inner.cols
    }

    #[getter]
    pub fn target_field(&self) -> String {
        self.inner.target_field.clone()
    }

    #[getter]
    pub fn metric(&self) -> String {
        self.inner.metric.clone()
    }

    #[getter]
    pub fn epochs(&self) -> usize {
        self.inner.epochs
    }

    #[getter]
    pub fn weights_path(&self) -> Option<String> {
        self.inner.weights_path.clone()
    }

    pub fn reset(&mut self) {
        self.inner.reset();
    }

    pub fn step(&mut self, action_idx: usize) -> bool {
        self.inner.step(action_idx)
    }

    pub fn valid_action_mask(&self) -> Vec<bool> {
        self.inner.valid_action_mask()
    }

    pub fn alpha_bytes<'py>(&mut self, py: Python<'py>) -> Bound<'py, PyBytes> {
        let vec = self.inner.alpha_matrix();
        let bytes = unsafe {
            std::slice::from_raw_parts(
                vec.as_ptr() as *const u8,
                vec.len() * std::mem::size_of::<f32>(),
            )
        };
        PyBytes::new(py, bytes)
    }

    pub fn evaluate_stack(&mut self) -> Vec<Vec<f32>> {
        self.inner.evaluate_stack()
    }

    pub fn evaluate_top(&mut self) -> Option<Vec<f32>> {
        self.inner.evaluate_top()
    }

    pub fn expr(&self) -> String {
        self.inner.expr()
    }

    pub fn action_name(&self, idx: usize) -> String {
        self.inner.action_name(idx)
    }

    pub fn class_matrix(&self) -> Vec<f32> {
        self.inner.class_matrix()
    }

}

#[pymodule]
fn comet_rs(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyStackEnv>()?;
    Ok(())
}
