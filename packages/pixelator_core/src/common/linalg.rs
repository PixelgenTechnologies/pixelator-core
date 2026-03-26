use ndarray::Array1;
use ndarray::Array2;

use sprs::CsMat;

pub fn outer(a: &Array1<f64>, b: &Array1<f64>) -> Array2<f64> {
    let dima = a.shape()[0];
    let dimb = b.shape()[0];
    let mut out = Array2::zeros((dima, dimb));

    for i in 0..dima {
        for j in 0..dimb {
            out[[i, j]] = a[i] * b[j];
        }
    }

    out
}

pub fn trace(matrix: CsMat<f64>) -> f64 {
    matrix.diag().iter().map(|(_, x)| *x).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_outer_basic() {
        let a = Array1::from(vec![1.0, 2.0, 3.0]);
        let b = Array1::from(vec![4.0, 5.0]);
        let result = outer(&a, &b);
        let expected =
            Array2::from_shape_vec((3, 2), vec![4.0, 5.0, 8.0, 10.0, 12.0, 15.0]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_outer_with_zeros() {
        let a = Array1::from(vec![0.0, 2.0]);
        let b = Array1::from(vec![0.0, 5.0]);
        let result = outer(&a, &b);
        let expected = Array2::from_shape_vec((2, 2), vec![0.0, 0.0, 0.0, 10.0]).unwrap();
        assert_eq!(result, expected);
    }

    #[test]
    fn test_outer_single_element() {
        let a = Array1::from(vec![7.0]);
        let b = Array1::from(vec![3.0]);
        let result = outer(&a, &b);
        let expected = Array2::from_shape_vec((1, 1), vec![21.0]).unwrap();
        assert_eq!(result, expected);
    }
}
