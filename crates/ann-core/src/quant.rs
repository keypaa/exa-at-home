// crates/ann-core/src/quant.rs
/// M1: exact float dot product. Later milestones add binary paths beside this.
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn dot_unit_vectors() {
        assert!((dot(&[1.0, 0.0], &[1.0, 0.0]) - 1.0).abs() < 1e-6);
        assert!(dot(&[1.0, 0.0], &[0.0, 1.0]).abs() < 1e-6);
    }
}
