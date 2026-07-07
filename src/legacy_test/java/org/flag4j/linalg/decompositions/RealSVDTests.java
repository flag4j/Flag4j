package org.flag4j.linalg.decompositions;

import org.flag4j.arrays.dense.Matrix;
import org.flag4j.linalg.decompositions.svd.RealSVD;
import org.flag4j.linalg.decompositions.svd.SVD;


class RealSVDTests {

    final SVD<Matrix> svd = new RealSVD(true);

    double[][] aEntries, expSEntries, expUEntries, expVEntries;

    Matrix A, expS, expU, expV;

    // TODO: There seems to be issues with SVD. Bring back tests when issues are fixed.
    //  The issues are likely related to eigenvector computation which is probably incorrect and inefficient.

//    @Test
//    void svdTestCase() {
//        // -------------------- sub-case 1 --------------------
//        aEntries = new double[][]{
//                {1, 2, 3},
//                {4, 5, 6},
//                {7, 8, 9}
//        };
//        A = new Matrix(aEntries);
//
//        expUEntries = new double[][]{
//                {0.21483723836839624, 0.8872306883463706, -0.4082482904638626},
//                {0.520587389464737, 0.24964395298829764, 0.8164965809277261},
//                {0.8263375405610778, -0.3879427823697744, -0.40824829046386324}
//        };
//        expU = new Matrix(expUEntries);
//        expSEntries = new double[][]{
//                {16.848103352614167, 0.0, 0.0},
//                {0.0, 1.0683695145547083, 0.0},
//                {0.0, 0.0, 1.1023900701150984E-16}
//        };
//        expS = new Matrix(expSEntries);
//        expVEntries = new double[][]{
//                {0.4796711778777717, -0.7766909903215589, -0.40824829046386213},
//                {0.5723677939720622, -0.07568647010455855, 0.8164965809277265},
//                {0.665064410066353, 0.6253180501124429, -0.4082482904638633}
//        };
//        expV = new Matrix(expVEntries);
//
//        svd.decompose(A);
//
//        System.out.println("U\n" + svd.getU() + "\n");
//        System.out.println("S\n" + svd.getS() + "\n");
//        System.out.println("V\n" + svd.getV() + "\n");
//
//        Matrix Ahat = svd.getU().mult(svd.getS()).mult(svd.getV().H());
//        assertEquals(new Matrix(A.shape), A.sub(Ahat).round(10));
//
//        // -------------------- sub-case 2 --------------------
//        aEntries = new double[][]{
//                {3.45, -99.34, 14.5, 24.5},
//                {-0.0024, 0, 25.1, 1.5},
//                {100.4, 5.6, -4.1, -0.002}
//        };
//        A = new Matrix(aEntries);
//
//        svd.decompose(A);
//        Ahat = svd.getU().mult(svd.getS()).mult(svd.getV().H());
//        assertEquals(new Matrix(A.shape), A.sub(Ahat).round(10));
//
//        // -------------------- sub-case 3 --------------------
//        aEntries = new double[][]{
//                {34.5, 100.34},
//                {-9.245, 0.13},
//                {0, 1153.4},
//                {14.5, -195.342}
//        };
//        A = new Matrix(aEntries);
//
//        svd.decompose(A);
//        Ahat = svd.getU().mult(svd.getS()).mult(svd.getV().H());
//        assertEquals(new Matrix(A.shape), A.sub(Ahat).round(10));
//
//        // -------------------- sub-case 4 --------------------
//        // This Toeplitz matrix is known to be difficult to compute eigenvalues of. As such, it is a good test matrix.
//        aEntries = new double[][]{
//                {2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//                {1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//                {0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
//                {0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0, 0},
//                {0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0, 0},
//                {0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0, 0},
//                {0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0, 0},
//                {0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0, 0},
//                {0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0, 0},
//                {0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0, 0},
//                {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1, 0},
//                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 1},
//                {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2},
//        };
//        A = new Matrix(aEntries);
//
//        svd.decompose(A);
//        Ahat = svd.getU().mult(svd.getS()).mult(svd.getV().H());
//        assertEquals(new Matrix(A.shape), A.sub(Ahat).round(10));
//    }
}
