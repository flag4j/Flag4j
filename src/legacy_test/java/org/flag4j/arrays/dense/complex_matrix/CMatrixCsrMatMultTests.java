package org.flag4j.arrays.dense.complex_matrix;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.arrays.dense.CVector;
import org.flag4j.arrays.sparse.CsrCMatrix;
import org.flag4j.arrays.sparse.CsrMatrix;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.flag4j.numbers.Complex128.ZERO;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CMatrixCsrMatMultTests {
    CMatrix A;
    Complex128[][] aEntries;

    CsrMatrix Breal;
    CsrCMatrix B;
    Shape bShape;
    double[] bRealEntries;
    Complex128[] bEntries;
    int[] bRowPointers;
    int[] bColIndices;

    CMatrix exp;
    Complex128[][] expEntries;


    @Test
    void standardRealTests() {
        // ------------------------ sub-case 1 ------------------------
        aEntries = new Complex128[][]{
                {new Complex128("0.1013+0.5667i"), new Complex128("0.56204+0.08795i"), new Complex128("0.26919+0.74589i"), new Complex128("0.40605+0.37181i"), new Complex128("0.41665+0.56373i")},
                {new Complex128("0.88631+0.70806i"), new Complex128("0.94873+0.70971i"), new Complex128("0.73508+0.92932i"), new Complex128("0.32551+0.08181i"), new Complex128("0.80165+0.87963i")},
                {new Complex128("0.01923+0.20639i"), new Complex128("0.01025+0.53356i"), new Complex128("0.77862+0.04428i"), new Complex128("0.24381+0.01189i"), new Complex128("0.5903+0.51795i")},
                {new Complex128("0.65994+0.40064i"), new Complex128("0.21257+0.16288i"), new Complex128("0.85927+0.11806i"), new Complex128("0.8716+0.70231i"), new Complex128("0.83819+0.21429i")},
                {new Complex128("0.29866+0.71364i"), new Complex128("0.46553+0.89626i"), new Complex128("0.25626+0.07154i"), new Complex128("0.27779+0.59077i"), new Complex128("0.3676+0.45885i")}};
        A = new CMatrix(aEntries);
        bShape = new Shape(5, 5);
        bRealEntries = new double[]{0.35336, 0.80623, 0.7923, 0.96503, 0.33155, 0.26233, 0.93305, 0.97519};
        bRowPointers = new int[]{0, 2, 3, 7, 8, 8};
        bColIndices = new int[]{1, 4, 4, 0, 1, 2, 3, 4};
        Breal = new CsrMatrix(bShape, bRealEntries, bRowPointers, bColIndices);
        expEntries = new Complex128[][]{
                {new Complex128("0.2597764257+0.7198062267i"), new Complex128("0.1250453125+0.4475489415i"), new Complex128("0.0706166127+0.1956693237i"), new Complex128("0.2511677295+0.6959526645i"), new Complex128("0.9229512905+0.8891587199i")},
                {new Complex128("0.7093742524+0.8968216796i"), new Complex128("0.5569022755999999+0.5583161276i"), new Complex128("0.1928335364+0.2437885156i"), new Complex128("0.685866394+0.8671020260000001i"), new Complex128("1.7836825872+1.2129427407000002i")},
                {new Complex128("0.7513916586+0.0427315284i"), new Complex128("0.26494657380000003+0.0876110044i"), new Complex128("0.2042553846+0.0116159724i"), new Complex128("0.726491391+0.041315454i"), new Complex128("0.2613859518+0.6007324068000001i")},
                {new Complex128("0.8292213281+0.1139314418i"), new Complex128("0.5180873669+0.18071294340000002i"), new Complex128("0.2254122991+0.0309706798i"), new Complex128("0.8017418735+0.11015588300000001i"), new Complex128("1.5504582412+1.1369435001000001i")},
                {new Complex128("0.2472985878+0.06903824620000001i"), new Complex128("0.19049750059999998+0.27589091740000005i"), new Complex128("0.0672246858+0.018767088200000004i"), new Complex128("0.239103393+0.066750397i"), new Complex128("0.8805261008999998+1.8615777715i")}};
        exp = new CMatrix(expEntries);
        assertEquals(exp, A.mult(Breal));

        // ------------------------ sub-case 2 ------------------------
        aEntries = new Complex128[][]{
                {new Complex128("0.66805+0.1356i"), new Complex128("0.02304+0.45698i"), new Complex128("0.21488+0.42425i")},
                {new Complex128("0.41083+0.05417i"), new Complex128("0.73696+0.55199i"), new Complex128("0.14116+0.82652i")},
                {new Complex128("0.37502+0.66468i"), new Complex128("0.0909+0.08818i"), new Complex128("0.67084+0.0016i")},
                {new Complex128("0.57517+0.93909i"), new Complex128("0.42799+0.32726i"), new Complex128("0.92712+0.44066i")},
                {new Complex128("0.9064+0.65531i"), new Complex128("0.12811+0.44938i"), new Complex128("0.3343+0.45097i")},
                {new Complex128("0.49565+0.04573i"), new Complex128("0.84561+0.56477i"), new Complex128("0.5758+0.99876i")},
                {new Complex128("0.82457+0.04468i"), new Complex128("0.73855+0.10166i"), new Complex128("0.23595+0.92517i")},
                {new Complex128("0.93964+0.6983i"), new Complex128("0.80439+0.29492i"), new Complex128("0.15538+0.7558i")},
                {new Complex128("0.17902+0.59831i"), new Complex128("0.78638+0.43736i"), new Complex128("0.33244+0.23329i")},
                {new Complex128("0.6007+0.16697i"), new Complex128("0.54194+0.05845i"), new Complex128("0.64488+0.21455i")},
                {new Complex128("0.26206+0.22261i"), new Complex128("0.4165+0.64287i"), new Complex128("0.27519+0.17844i")}};
        A = new CMatrix(aEntries);

        bShape = new Shape(3, 10);
        bRealEntries = new double[]{0.58579, 0.44167, 0.79592};
        bRowPointers = new int[]{0, 1, 3, 3};
        bColIndices = new int[]{1, 4, 5};
        Breal = new CsrMatrix(bShape, bRealEntries, bRowPointers, bColIndices);

        expEntries = new Complex128[][]{
                {new Complex128("0.0"), new Complex128("0.39133700950000005+0.07943312400000001i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.010176076800000001+0.20183435660000001i"), new Complex128("0.0183379968+0.3637195216i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.2406601057+0.031732244300000004i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.32549312319999996+0.24379742329999998i"), new Complex128("0.5865612031999999+0.43933988079999997i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.21968296580000002+0.38936289720000006i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.040147802999999996+0.038946460599999996i"), new Complex128("0.072349128+0.0701842256i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.3369288343+0.5501095311i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.1890303433+0.1445409242i"), new Complex128("0.34064580079999995+0.2604727792i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.5309600560000001+0.3838740449i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0565823437+0.1984776646i"), new Complex128("0.10196531119999999+0.3576705296i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.2903468135+0.026788176700000003i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.3734805687+0.2494419659i"), new Complex128("0.6730379111999999+0.44951173839999997i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.4830248603+0.0261730972i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.3261953785+0.0449001722i"), new Complex128("0.587826716+0.08091322719999999i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.5504317156+0.40905715700000006i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.35527493130000004+0.13025731640000002i"), new Complex128("0.6402300888+0.2347327264i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.10486812580000002+0.35048401490000003i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.3473204546+0.19316879120000002i"), new Complex128("0.6258955695999999+0.3481035712i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.35188405300000003+0.09780935630000001i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.2393586398+0.025815611500000002i"), new Complex128("0.43134088479999994+0.046521524i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")},
                {new Complex128("0.0"), new Complex128("0.15351212740000003+0.1304027119i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.183955555+0.2839363929i"), new Complex128("0.33150068+0.5116730904i"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0"), new Complex128("0.0")}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.mult(Breal));

        // ------------------------ sub-case 3 ------------------------
        A = new CMatrix(24, 516);
        Breal = new CsrMatrix(15, 12);
        assertThrows(LinearAlgebraException.class, ()->A.mult(Breal));
    }


    @Test
    void standardTests() {
        // ------------------------ sub-case 1 ------------------------
        aEntries = new Complex128[][]{
                {new Complex128(0.2259, 0.42103), new Complex128(0.28846, 0.02369), new Complex128(0.01058, 0.56772), new Complex128(0.95379, 0.05841), new Complex128(0.74607, 0.44243)},
                {new Complex128(0.54801, 0.91401), new Complex128(0.14567, 0.17494), new Complex128(0.57924, 0.7371), new Complex128(0.12672, 0.1154), new Complex128(0.6019, 0.78311)},
                {new Complex128(0.12358, 0.62565), new Complex128(0.18628, 0.85896), new Complex128(0.50953, 0.95163), new Complex128(0.57341, 0.24745), new Complex128(0.95169, 0.32643)}};
        A = new CMatrix(aEntries);
        bShape = new Shape(5, 5);
        bEntries = new Complex128[]{new Complex128(0.46134, 0.36271), new Complex128(0.84641, 0.7844), new Complex128(0.37528, 0.27378),
                new Complex128(0.71721, 0.72083), new Complex128(0.83229, 0.48221), new Complex128(0.58432, 0.8158), new Complex128(0.5328, 0.79834), new Complex128(0.41335, 0.35008)};
        bRowPointers = new int[]{0, 3, 4, 4, 7, 8};
        bColIndices = new int[]{0, 1, 2, 4, 0, 3, 4, 1};
        B = new CsrCMatrix(bShape, bEntries, bRowPointers, bColIndices);
        expEntries = new Complex128[][]{
                {new Complex128(0.7171689077, 0.7847153040000001), new Complex128(0.01445022709999999, 0.9776225884), new Complex128(-0.030493841400000013, 0.2198510404), new Complex128(0.5096676948, 0.8122320131999999), new Complex128(0.6513582065000001, 1.0174908833)},
                {new Complex128(-0.02888087890000001, 0.7775899977), new Complex128(-0.2784640836999999, 1.7378979185999999), new Complex128(-0.044580465000000014, 0.4930438506), new Complex128(-0.020098289600000013, 0.17080870399999998), new Complex128(-0.04623803950000001, 0.3931227883)},
                {new Complex128(0.18800343009999992, 0.8159152694), new Complex128(-0.10705606509999999, 1.0945900442), new Complex128(-0.12491335460000003, 0.2686276644), new Complex128(0.13318522119999995, 0.6123778619999999), new Complex128(-0.377598643, 1.3399484134000001)}};
        exp = new CMatrix(expEntries);
        assertEquals(exp, A.mult(B));

        // ------------------------ sub-case 2 ------------------------
        aEntries = new Complex128[][]{
                {new Complex128(0.296, 0.48224), new Complex128(0.21107, 0.36182), new Complex128(0.54805, 0.07346)},
                {new Complex128(0.16943, 0.1518), new Complex128(0.45495, 0.5998), new Complex128(0.31595, 0.12229)},
                {new Complex128(0.75293, 0.12555), new Complex128(0.71752, 0.99826), new Complex128(0.20345, 0.82965)},
                {new Complex128(0.90122, 0.14502), new Complex128(0.57932, 0.31246), new Complex128(0.47947, 0.64057)},
                {new Complex128(0.93961, 0.56065), new Complex128(0.39266, 0.27952), new Complex128(0.79273, 0.12665)},
                {new Complex128(0.5556, 0.21159), new Complex128(0.06416, 0.39728), new Complex128(0.12435, 0.77808)},
                {new Complex128(0.06261, 0.11049), new Complex128(0.87951, 0.69049), new Complex128(0.85275, 0.32878)},
                {new Complex128(0.45795, 0.11123), new Complex128(0.82144, 0.3513), new Complex128(0.20372, 0.88586)},
                {new Complex128(0.22205, 0.53269), new Complex128(0.06012, 0.99425), new Complex128(0.03268, 0.60358)},
                {new Complex128(0.7371, 0.98015), new Complex128(0.55131, 0.96068), new Complex128(0.86053, 0.08451)},
                {new Complex128(0.39417, 0.60287), new Complex128(0.75213, 0.70276), new Complex128(0.47889, 0.34281)}};
        A = new CMatrix(aEntries);

        bShape = new Shape(3, 10);
        bEntries = new Complex128[]{new Complex128(0.40213, 0.62104), new Complex128(0.92285, 0.9137), new Complex128(0.45774, 0.49795)};
        bRowPointers = new int[]{0, 1, 1, 3};
        bColIndices = new int[]{4, 1, 9};
        B = new CsrCMatrix(bShape, bEntries, bRowPointers, bColIndices);

        expEntries = new Complex128[][]{
                {ZERO, new Complex128(0.4386475405, 0.568545846), ZERO, ZERO, new Complex128(-0.1804598496, 0.3777510112), ZERO, ZERO, ZERO, ZERO, new Complex128(0.214285, 0.30652707790000006)},
                {ZERO, new Complex128(0.1798380845, 0.40153884149999997), ZERO, ZERO, new Complex128(-0.0261409861, 0.1662661412), ZERO, ZERO, ZERO, ZERO, new Complex128(0.0837286475, 0.21330432710000002)},
                {ZERO, new Complex128(-0.5702973725, 0.9515347675), ZERO, ZERO, new Complex128(0.2248041689, 0.5180870687), ZERO, ZERO, ZERO, ZERO, new Complex128(-0.3199970145, 0.4810719185)},
                {ZERO, new Complex128(-0.14280991949999994, 1.0292417635), ZERO, ZERO, new Complex128(0.2723443778, 0.6180105614), ZERO, ZERO, ZERO, ZERO, new Complex128(-0.09949923369999997, 0.5319665982999999)},
                {ZERO, new Complex128(0.6158507755, 0.8411963535), ZERO, ZERO, new Complex128(0.029659293299999945, 0.8089895789), ZERO, ZERO, ZERO, ZERO, new Complex128(0.2997988627, 0.4527126745000001)},
                {ZERO, new Complex128(-0.5961752985, 0.831669723), ZERO, ZERO, new Complex128(0.09201757439999997, 0.43013651070000003), ZERO, ZERO, ZERO, ZERO, new Complex128(-0.330524967, 0.4180784217)},
                {ZERO, new Complex128(0.4865540514999999, 1.082572298), ZERO, ZERO, new Complex128(-0.04344135030000001, 0.0833146581), ZERO, ZERO, ZERO, ZERO, new Complex128(0.226621784, 0.5751226197)},
                {ZERO, new Complex128(-0.62140728, 1.003654865), ZERO, ZERO, new Complex128(0.1150771543, 0.32913418790000004), ZERO, ZERO, ZERO, ZERO, new Complex128(-0.3478631942, 0.5069359304)},
                {ZERO, new Complex128(-0.521332308, 0.586873519), ZERO, ZERO, new Complex128(-0.2415288311, 0.3521125617), ZERO, ZERO, ZERO, ZERO, new Complex128(-0.2855937178, 0.29255571519999996)},
                {ZERO, new Complex128(0.7169233235, 0.8642563145), ZERO, ZERO, new Complex128(-0.3123023330000001, 0.8519163035), ZERO, ZERO, ZERO, ZERO, new Complex128(0.3518172477, 0.4671845209)},
                {ZERO, new Complex128(0.1287181395, 0.7539240014999999), ZERO, ZERO, new Complex128(-0.21589880270000003, 0.48722744990000005), ZERO, ZERO, ZERO, ZERO, new Complex128(0.048504869099999987, 0.3953811249)}};
        exp = new CMatrix(expEntries);

        assertEquals(exp, A.mult(B));

        // ------------------------ sub-case 3 ------------------------
        A = new CMatrix(24, 516);
        B = new CsrCMatrix(15, 12);
        assertThrows(LinearAlgebraException.class, ()->A.mult(B));
    }


    @Test
    void standardVectorTests() {
        Shape aShape;
        Complex128[] aData;
        Complex128[] bData;
        Complex128[] expData;
        int[] aRowPointers;
        int[] aColIndices;
        CsrCMatrix A;
        CVector B;
        CVector exp;


        // ------------------------ sub-case 1 ------------------------
        aShape = new Shape(11, 3);
        aData = new Complex128[]{new Complex128(0.66236, 0.99511), new Complex128(0.64644, 0.38726), new Complex128(0.74006, 0.0656)};
        aRowPointers = new int[]{0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 3, 3};
        aColIndices = new int[]{0, 2, 0};
        A = new CsrCMatrix(aShape, aData, aRowPointers, aColIndices);

        Shape bShape = new Shape(3);
        bData = new Complex128[]{new Complex128(0.3658, 0.98422), new Complex128(0.41882, 0.92433), new Complex128(0.0877, 0.37298)};
        B = new CVector(bShape, bData);

        Shape expShape = new Shape(11);
        expData = new Complex128[]{ZERO, ZERO, ZERO,
                new Complex128(-0.7371158762000001, 1.0159191972), ZERO,
                ZERO, ZERO, new Complex128(-0.08774744679999999, 0.2750718932),
                ZERO, new Complex128(0.20614911600000002, 0.7523783332), ZERO};
        exp = new CVector(expShape, expData);

        assertEquals(exp, A.mult(B));

        // ------------------------ sub-case 2 ------------------------
        aShape = new Shape(25, 12);
        aData = new Complex128[]{new Complex128(0.93957, 0.74939), new Complex128(0.85329, 0.4632), new Complex128(0.74275, 0.36821)};
        aRowPointers = new int[]{0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3};
        aColIndices = new int[]{3, 11, 3};
        A = new CsrCMatrix(aShape, aData, aRowPointers, aColIndices);

        bShape = new Shape(12);
        bData = new Complex128[]{new Complex128(0.71476, 0.21926), new Complex128(0.70663, 0.32465), new Complex128(0.86338, 0.91874), new Complex128(0.858, 0.18053), new Complex128(0.91248, 0.40531), new Complex128(0.19292, 0.76875), new Complex128(0.09796, 0.25519), new Complex128(0.33066, 0.93037), new Complex128(0.19831, 0.98659), new Complex128(0.63703, 0.12761), new Complex128(0.43991, 0.98135), new Complex128(0.54505, 0.15659)};
        B = new CVector(bShape, bData);

        expShape = new Shape(25);
        expData = new Complex128[]{ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, new Complex128(0.6708636833, 0.8125971921), ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, new Complex128(0.3925532265, 0.38608384110000005), new Complex128(0.5708065487, 0.4500128375), ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, ZERO};
        exp = new CVector(expShape, expData);

        assertEquals(exp, A.mult(B));

        // ------------------------ sub-case 3 ------------------------
        aShape = new Shape(8, 12);
        aData = new Complex128[]{new Complex128(0.49723, 0.92929), new Complex128(0.97348, 0.99936), new Complex128(0.446, 0.17502), new Complex128(0.70681, 0.54578), new Complex128(0.87631, 0.11464), new Complex128(0.8615, 0.34637), new Complex128(0.31813, 0.74686), new Complex128(0.58007, 0.62119)};
        aRowPointers = new int[]{0, 1, 1, 3, 3, 5, 6, 7, 8};
        aColIndices = new int[]{2, 3, 8, 4, 9, 3, 8, 4};
        A = new CsrCMatrix(aShape, aData, aRowPointers, aColIndices);

        bShape = new Shape(12);
        bData = new Complex128[]{new Complex128(0.18108, 0.66938), new Complex128(0.50518, 0.39486), new Complex128(0.49211, 0.5359), new Complex128(0.82519, 0.59277), new Complex128(0.35919, 0.5141), new Complex128(0.06634, 0.15878), new Complex128(0.99391, 0.72529), new Complex128(0.25949, 0.64906), new Complex128(0.8205, 0.25117), new Complex128(0.94748, 0.6847), new Complex128(0.81247, 0.26422), new Complex128(0.30642, 0.20797)};
        B = new CVector(bShape, bData);

        expShape = new Shape(8);
        expData = new Complex128[]{new Complex128(-0.2533146557, 0.7237784589), ZERO, new Complex128(0.5328985606, 1.6573373480000002), ZERO, new Complex128(0.7250857767000001, 1.2680383034), new Complex128(0.5055834401, 0.7964924153), new Complex128(0.07343683880000001, 0.6927033421000001), new Complex128(-0.11099843570000001, 0.5213392231)};
        exp = new CVector(expShape, expData);

        assertEquals(exp, A.mult(B));

        // ------------------------ sub-case 4 ------------------------
        aShape = new Shape(12, 12);
        aData = new Complex128[]{};
        aRowPointers = new int[]{0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
        aColIndices = new int[]{};
        A = new CsrCMatrix(aShape, aData, aRowPointers, aColIndices);

        bShape = new Shape(12);
        bData = new Complex128[]{new Complex128(0.5048, 0.33308), new Complex128(0.89792, 0.43697), new Complex128(0.11148, 0.44758), new Complex128(0.48047, 0.79639), new Complex128(0.90456, 0.12293), new Complex128(0.2917, 0.64718), new Complex128(0.8388, 0.82837), new Complex128(0.79096, 0.5637), new Complex128(0.35008, 0.1461), new Complex128(0.24666, 0.35544), new Complex128(0.58987, 0.27922), new Complex128(0.68159, 0.04372)};
        B = new CVector(bShape, bData);

        expShape = new Shape(12);
        expData = new Complex128[]{ZERO, ZERO, ZERO, ZERO, ZERO, 
                ZERO, ZERO, ZERO, ZERO, ZERO, ZERO, 
                ZERO};
        exp = new CVector(expShape, expData);
        assertEquals(exp, A.mult(B));

        // ------------------------ sub-case 5 ------------------------
        aShape = new Shape(10, 10);
        A = new CsrCMatrix(aShape);
        bShape = new Shape(6);
        B = new CVector(bShape);

        CsrCMatrix finalA = A;
        CVector finalB = B;
        assertThrows(IllegalArgumentException.class, ()-> finalA.mult(finalB));

        aShape = new Shape(10, 10);
        A = new CsrCMatrix(aShape);
        bShape = new Shape(32);
        B = new CVector(bShape);

        CsrCMatrix finalA1 = A;
        CVector finalB1 = B;
        assertThrows(IllegalArgumentException.class, ()-> finalA1.mult(finalB1));
    }
}
