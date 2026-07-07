package org.flag4j;

import org.flag4j.arrays.backend.semiring_arrays.AbstractDenseSemiringMatrix;
import org.flag4j.arrays.dense.Matrix;

public final class TestUtils {

    private TestUtils() {}


    /**
     * @param a Matrix of interest.
     * @return {@code true} if the matrix only has zeros below the principle diagonal; {@code false} otherwise.
     */
    public static boolean isUpperTriLike(Matrix a) {
        final int bound = Math.min(a.numRows, a.numCols);

        // Ensure lower half is zeros.
        for(int i=1; i<bound; i++) {
            int rowOffset = i*a.numCols;

            for(int j=0; j<i; j++)
                if(a.data[rowOffset + j] != 0) return false; // No need to continue.
        }

        return true;
    }


    /**
     * @param a Matrix of interest.
     * @return {@code true} if the matrix only has zeros below the principle diagonal; {@code false} otherwise.
     */
    public static boolean isUpperTriLike(AbstractDenseSemiringMatrix<?, ?, ?> a) {
        final int bound = Math.min(a.numRows, a.numCols);

        // Ensure lower half is zeros.
        for(int i=1; i<bound; i++) {
            int rowOffset = i*a.numCols;

            for(int j=0; j<i; j++)
                if(!a.data[rowOffset + j].isZero()) return false; // No need to continue.
        }

        return true;
    }
}
