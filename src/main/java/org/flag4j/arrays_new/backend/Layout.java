package org.flag4j.arrays_new.backend;

import org.flag4j.arrays_new.NewShape;
import org.flag4j.util.ArrayUtils;

import java.util.Arrays;

// TODO: Finish docs.
/// An instance of the `Layout` class
public record Layout(int offset, int[] strides) {
    static Layout contiguous(NewShape shape) {
        // TODO NOW: We need to add protection against integer overflow here...
        shape.isIntSized();

        int rank = shape.getRank();
        int[] strides = new int[rank];

        ArrayUtils.cumSum(shape.getDims(), strides);

        int acc = 1;
        for (int a = rank - 1; a >= 0; a--) {   // stride[a] = product of dims to the right
            strides[a] = acc;
            acc *= shape.getSize(a);
        }

        return new Layout(0, strides);
    }

    /**
     * Converts an nD index to the 1D index of the data buffer of an nD array with this layout.
     * @param idxND The ND index to convert.
     * @return The 1D index of a nD array's data buffer corresponding to the nD index {idxND}.
     */
    public int to1DIndex(int... idxND) {
        int p = offset;
        for (int a = 0; a < idxND.length; a++) p += idxND[a] * strides[a];
        return p;
    }


    public int[] toNDIndex(int idx) {
        return null;
    }
}
