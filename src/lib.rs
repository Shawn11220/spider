use pyo3::prelude::*;

pub mod bio;
pub mod db;
pub mod search;
pub mod storage;

use db::SpiderDB;

/// A Python module implemented in Rust.
#[pymodule]
fn spider(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<SpiderDB>()?;
    Ok(())
}
