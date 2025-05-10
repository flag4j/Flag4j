package org.flag4j;

import org.flag4j.arrays.backend.AbstractNDArray;
import org.flag4j.arrays.backend.MatrixMixin;
import org.flag4j.arrays.backend.VectorMixin;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public final class TestHelpers {

    public static void printAsNumpyArray(Object... args) {
        for(Object arg : args) {
            if(arg instanceof MatrixMixin<?, ?, ?, ?>) {
                printAsNumpyArray((MatrixMixin<?, ?, ?, ?>) arg);
            } else {
                System.out.print(arg.toString());
            }
        }
    }


    public static void printAsJavaArray(Object... args) {
        for(Object arg : args) {
            if(arg instanceof MatrixMixin<?, ?, ?, ?>) {
                printAsJavaArray((MatrixMixin<?, ?, ?, ?>) arg);
            } else if(arg instanceof VectorMixin<?,?,?,?>) {
                printAsJavaArray((VectorMixin<?,?,?,?>) arg);
            } else {
                System.out.print(arg.toString()); // Type not found, fall back to toString method.
            }
        }
    }


    private static <T extends MatrixMixin<?, ?, ?, ?>> void printAsNumpyArray(T A) {
        System.out.println(" = np.array([");

        for(int i=0; i<A.numRows(); i++) {
            System.out.print("\t[");
            for(int j=0; j<A.numCols(); j++) {

                if(A instanceof CMatrix B) {
                    if(B.get(i, j).im > 0) {
                        System.out.print(B.get(i, j).re + "+" + B.get(i, j).im + "j");
                    } else {
                        System.out.print(B.get(i, j).re + B.get(i, j).im + "j");
                    }
                } else {
                    // Then must be real.
                    System.out.print(A.get(i, j));
                }

                if(j < A.numCols()-1) {
                    System.out.print(", ");
                }
            }
            System.out.print("]");

            if(i < A.numRows()-1) System.out.println(",");
        }

        System.out.println("\n])");
    }


    private static <T extends VectorMixin<?,?,?,?>> void printAsJavaArray(T A) {
        System.out.print("{");

        for(int i=0; i<A.length(); i++) {
            if(A instanceof CVector B) {
                System.out.print("new Complex128(" + B.get(i).re + ", " + B.get(i).im + ")");
            } else {
                // Then must be real.
                System.out.print(A.get(i));
            }

            if(i < A.length()-1) System.out.print(", ");
        }
        System.out.println("};");
    }


    private static <T extends MatrixMixin<?, ?, ?, ?>> void printAsJavaArray(T A) {
        System.out.println("{");

        for(int i=0; i<A.numRows(); i++) {
            System.out.print("\t{");
            for(int j=0; j<A.numCols(); j++) {
                if(A instanceof CMatrix B) {
                    System.out.print("new Complex128(" + B.get(i, j).re + ", " + B.get(i, j).im + ")");
                } else {
                    // Then must be real.
                    System.out.print(A.get(i, j));
                }

                if(j < A.numCols()-1) {
                    System.out.print(", ");
                }
            }
            System.out.print("}");

            if(i < A.numRows()-1) {
                System.out.println(",");
            }
        }

        System.out.println("\n};");
    }


    /**
     * Finds differences between two dense tensors and prints them out.
     * @param a First tensor to compare.
     * @param b Second tensor to compare.
     */
    public static <T extends AbstractNDArray<?,?,?>> List<int[]> findDiff(T a, T b) {
        if(!a.shape.equals(b.shape)) {
            System.out.printf("Not the same shape: %s and %s\n", a.shape, b.shape);
        }

        int stop = a.totalEntries().intValueExact();
        List<int[]> diffIndices = new ArrayList<>();

        for(int i=0; i<stop; i++) {
            int[] indices = a.shape.getNdIndices(i);

            if(!a.get(indices).equals(b.get(indices))) {
                System.out.printf("Difference at %s: %s, %s\n",
                        Arrays.toString(a.shape.getNdIndices(i)),
                        a.get(indices),
                        b.get(indices));
                diffIndices.add(indices);
            }
        }

        return diffIndices;
    }
}
