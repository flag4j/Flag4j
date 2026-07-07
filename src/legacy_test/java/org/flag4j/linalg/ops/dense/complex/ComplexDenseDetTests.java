package org.flag4j.linalg.ops.dense.complex;

import org.flag4j.arrays.dense.CMatrix;
import org.flag4j.linalg.ops.dense.field_ops.DenseFieldDeterminant;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class ComplexDenseDetTests {

    Complex128[][] entries;
    CMatrix A;  // Matrix to compute determinate of.
    Complex128 act;  // Stores actual computed determinate.
    Complex128 exp;  // Stores expected determinate.

    @Test
    void det1Test() {
        // -------------- sub-case 1 --------------
        entries = new Complex128[][]{{new Complex128(9.5, -14.5)}};
        A = new CMatrix(entries);
        exp = new Complex128(9.5, -14.5);
        act = ComplexDenseDeterminant.det(A);
        assertEquals(exp, act);

        // -------------- sub-case 2 --------------
        entries = new Complex128[][]{{new Complex128(1.4)}};
        A = new CMatrix(entries);
        exp = new Complex128(1.4);
        act = ComplexDenseDeterminant.det(A);
        assertEquals(exp, act);

        // -------------- sub-case 3 --------------
        entries = new Complex128[][]{{new Complex128(-9)}};
        A = new CMatrix(entries);
        exp = new Complex128(-9);
        act = ComplexDenseDeterminant.det(A);
        assertEquals(exp, act);

        // -------------- sub-case 4 --------------
        entries = new Complex128[][]{{new Complex128(0, 9765.134523400202002)}};
        A = new CMatrix(entries);
        exp = new Complex128(0, 9765.134523400202002);
        act = ComplexDenseDeterminant.det(A);
        assertEquals(exp, act);

        // -------------- sub-case 5 --------------
        entries = new Complex128[][]{{Complex128.ZERO, Complex128.ZERO}, {Complex128.ZERO, Complex128.ZERO}};
        A = new CMatrix(entries);
        assertThrows(LinearAlgebraException.class, ()-> DenseFieldDeterminant.det1(A));
    }


    @Test
    void det2Test() {
        // -------------- sub-case 1 --------------
        entries = new Complex128[][]{
                {new Complex128("-1.0"), new Complex128("5.0")},
                {new Complex128("133.5"), new Complex128("25.0")}};
        A = new CMatrix(entries);
        exp = new Complex128(-692.5);
        act = ComplexDenseDeterminant.det(A);
        assertEquals(exp, act);

        // -------------- sub-case 2 --------------
        entries = new Complex128[][]{
                {new Complex128("-1.0+0.345i"), new Complex128("5.0+2.0i")},
                {new Complex128("133.5+0.00014i"), new Complex128("-0.0-25.0i")}};
        A = new CMatrix(entries);
        exp = new Complex128("-658.87472 - 242.0007i");
        act = ComplexDenseDeterminant.det(A);
        assertEquals(exp, act);

        // -------------- sub-case 3 --------------
        entries = new Complex128[][]{
                {new Complex128("0.0+8.0i"), new Complex128("-3.0+2.0i")},
                {new Complex128("-3.0+2.0i"), new Complex128("-0.0-25.0i")}};
        A = new CMatrix(entries);
        exp = new Complex128("195+12i");
        act = ComplexDenseDeterminant.det(A);
        assertEquals(exp, act);

        // -------------- sub-case 4 --------------
        entries = new Complex128[][]{
                {new Complex128("1.0"), new Complex128("6.0-2.0i")},
                {new Complex128("6.0+2.0i"), new Complex128("5.0")}};
        A = new CMatrix(entries);
        exp = new Complex128("-35");
        act = ComplexDenseDeterminant.det(A);
        assertEquals(exp, act);

        // -------------- sub-case 5 --------------
        entries = new Complex128[][]{{Complex128.ZERO}};
        A = new CMatrix(entries);
        assertThrows(LinearAlgebraException.class, ()->DenseFieldDeterminant.det2(A));

        // -------------- sub-case 6 --------------
        entries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO}};
        A = new CMatrix(entries);
        assertThrows(LinearAlgebraException.class, ()->DenseFieldDeterminant.det2(A));
    }


    @Test
    void det3Test() {
        // -------------- sub-case 1 --------------
        entries = new Complex128[][]{
                {new Complex128("1"), new Complex128("2"), new Complex128("3")},
                {new Complex128("4"), new Complex128("5"), new Complex128("6")},
                {new Complex128("7"), new Complex128("8"), new Complex128("9")}};
        A = new CMatrix(entries);
        exp = Complex128.ZERO;
        act = ComplexDenseDeterminant.det(A);
        assertEquals(exp, act);

        // -------------- sub-case 2 --------------
        entries = new Complex128[][]{
                {new Complex128("19.45"), new Complex128("-0.0035"), new Complex128("125.6")},
                {new Complex128("1.000035"), new Complex128("5.6"), new Complex128("1556.78679")},
                {new Complex128("6.0"), new Complex128("-992.45"), new Complex128("15.7")}};
        A = new CMatrix(entries);
        exp = new Complex128("29923693.980974607");
        act = ComplexDenseDeterminant.det(A);
        assertEquals(exp, act);

        // -------------- sub-case 3 --------------
        entries = new Complex128[][]{
                {new Complex128("19.45+0.02026i"), new Complex128("-0.0035+8.0i"), new Complex128("125.6")},
                {new Complex128("1.000035"), new Complex128("5.6+30.49856i"), new Complex128("1556.78679-985.355i")},
                {new Complex128("6.0-2.0i"), new Complex128("-992.45-2.0i"), new Complex128("15.7+0.091456i")}};
        A = new CMatrix(entries);
        exp = new Complex128("30046261.835600328 - 18882193.78222524i");
        act = ComplexDenseDeterminant.det(A);
        assertEquals(exp, act);

        // -------------- sub-case 4 --------------
        entries = new Complex128[][]{{Complex128.ZERO, Complex128.ZERO}, {Complex128.ZERO, Complex128.ZERO}};
        A = new CMatrix(entries);
        assertThrows(LinearAlgebraException.class, ()->DenseFieldDeterminant.det3(A));

        // -------------- sub-case 5 --------------
        entries = new Complex128[][]{
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO},
                {Complex128.ZERO, Complex128.ZERO, Complex128.ZERO, Complex128.ZERO}};
        A = new CMatrix(entries);
        assertThrows(LinearAlgebraException.class, ()->DenseFieldDeterminant.det3(A));
    }

    @Test
    void detLUTest() {
        // -------------- sub-case 1 --------------
        entries = new Complex128[][]{
                {new Complex128("1.0"), new Complex128("-53.0"), new Complex128("611.5"), new Complex128("234.6")},
                {new Complex128("-0.00351"), new Complex128("0.0"), new Complex128("24.56"), new Complex128("14.6")},
                {new Complex128("998.3"), new Complex128("2.5"), new Complex128("1.0"), new Complex128("0.00000025")},
                {new Complex128("-92.5"), new Complex128("14.5"), new Complex128("7.1"), new Complex128("34.0")}};
        A = new CMatrix(entries);
        exp = new Complex128("7935026.786550307");
        act = ComplexDenseDeterminant.det(A);
        assertEquals(exp, act);

        // -------------- sub-case 2 --------------
        entries = new Complex128[][]{
                {new Complex128("0.41059+0.9284i"), new Complex128("0.13828+0.91072i"), new Complex128("0.70458+0.9232i"), new Complex128("0.6675+0.94988i")},
                {new Complex128("0.46771+0.6101i"), new Complex128("0.82633+0.28485i"), new Complex128("0.22087+0.69659i"), new Complex128("0.97234+0.36372i")},
                {new Complex128("0.97155+0.84824i"), new Complex128("0.13442+0.81678i"), new Complex128("0.65467+0.31053i"), new Complex128("0.62112+0.6425i")},
                {new Complex128("0.68325+0.50348i"), new Complex128("0.50074+0.96227i"), new Complex128("0.84584+0.45291i"), new Complex128("0.57167+0.06987i")}};
        A = new CMatrix(entries);
        exp = new Complex128("-0.33775706366650027+0.46301856289254556i");
        act = ComplexDenseDeterminant.det(A);
        assertEquals(exp, act);

        // -------------- sub-case 3 --------------
        entries = new Complex128[][]{
                {new Complex128("0.00399+0.99393i"), new Complex128("0.5602+0.87092i"), new Complex128("0.94279+0.72353i"), new Complex128("0.26672+0.89313i"), new Complex128("0.93043+0.36442i"), new Complex128("0.56652+0.75951i"), new Complex128("0.38806+0.72435i"), new Complex128("0.34656+0.05705i"), new Complex128("0.98495+0.98219i"), new Complex128("0.19058+0.99072i"), new Complex128("0.71766+0.87942i"), new Complex128("0.67899+0.64893i"), new Complex128("0.57014+0.89789i"), new Complex128("0.35228+0.72217i"), new Complex128("0.91898+0.55195i")},
                {new Complex128("0.83284+0.56167i"), new Complex128("0.72028+0.14505i"), new Complex128("0.84075+0.29224i"), new Complex128("0.93349+0.37263i"), new Complex128("0.97302+0.3842i"), new Complex128("0.881+0.79757i"), new Complex128("0.83075+0.93624i"), new Complex128("0.92814+0.47567i"), new Complex128("0.26395+0.93256i"), new Complex128("0.01236+0.38783i"), new Complex128("0.1645+0.91338i"), new Complex128("0.5939+0.55956i"), new Complex128("0.41956+0.05676i"), new Complex128("0.16001+0.19826i"), new Complex128("0.65655+0.63476i")},
                {new Complex128("0.64889+0.68114i"), new Complex128("0.20075+0.60461i"), new Complex128("0.03752+0.54424i"), new Complex128("0.16755+0.49303i"), new Complex128("0.08295+0.81619i"), new Complex128("0.54819+0.34945i"), new Complex128("0.61779+0.65259i"), new Complex128("0.32885+0.34203i"), new Complex128("0.88752+0.02295i"), new Complex128("0.74886+0.94301i"), new Complex128("0.46661+0.46275i"), new Complex128("0.0622+0.06144i"), new Complex128("0.12673+0.08713i"), new Complex128("0.36444+0.22611i"), new Complex128("0.8904+0.09787i")},
                {new Complex128("0.08212+0.21184i"), new Complex128("0.77892+0.8349i"), new Complex128("0.06279+0.86456i"), new Complex128("0.83241+0.71363i"), new Complex128("0.45711+0.5999i"), new Complex128("0.60289+0.7781i"), new Complex128("0.22582+0.91327i"), new Complex128("0.9733+0.61772i"), new Complex128("0.61748+0.438i"), new Complex128("0.02977+0.65873i"), new Complex128("0.64303+0.19697i"), new Complex128("0.45741+0.53814i"), new Complex128("0.50226+0.34232i"), new Complex128("0.9652+0.07049i"), new Complex128("0.84176+0.60646i")},
                {new Complex128("0.9111+0.32226i"), new Complex128("0.91951+0.93551i"), new Complex128("0.239+0.48126i"), new Complex128("0.16065+0.63575i"), new Complex128("0.71097+0.72539i"), new Complex128("0.10342+0.22849i"), new Complex128("0.96078+0.55933i"), new Complex128("0.98355+0.44822i"), new Complex128("0.96949+0.91869i"), new Complex128("0.58076+0.45637i"), new Complex128("0.90556+0.04729i"), new Complex128("0.26906+0.77125i"), new Complex128("0.168+0.82065i"), new Complex128("0.06128+0.08468i"), new Complex128("0.35874+0.97037i")},
                {new Complex128("0.77575+0.3898i"), new Complex128("0.7468+0.58737i"), new Complex128("0.75703+0.43206i"), new Complex128("0.00236+0.2227i"), new Complex128("0.69134+0.50974i"), new Complex128("0.23623+0.08836i"), new Complex128("0.7311+0.49747i"), new Complex128("0.44573+0.63388i"), new Complex128("0.21488+0.76934i"), new Complex128("0.10153+0.46149i"), new Complex128("0.68971+0.63242i"), new Complex128("0.22933+0.37297i"), new Complex128("0.70293+0.71534i"), new Complex128("0.01343+0.76882i"), new Complex128("0.84475+0.50923i")},
                {new Complex128("0.82659+0.68179i"), new Complex128("0.63321+0.89338i"), new Complex128("0.37323+0.10609i"), new Complex128("0.77274+0.88217i"), new Complex128("0.57556+0.79726i"), new Complex128("0.73872+0.10875i"), new Complex128("0.82994+0.44293i"), new Complex128("0.8565+0.50379i"), new Complex128("0.85324+0.31997i"), new Complex128("0.35543+0.44813i"), new Complex128("0.16575+0.45721i"), new Complex128("0.30071+0.71059i"), new Complex128("0.91801+0.05638i"), new Complex128("0.98455+0.75933i"), new Complex128("0.03145+0.84904i")},
                {new Complex128("0.80872+0.41368i"), new Complex128("0.43987+0.81422i"), new Complex128("0.25294+0.16763i"), new Complex128("0.2439+0.09514i"), new Complex128("0.25768+0.7513i"), new Complex128("0.31547+0.86362i"), new Complex128("0.60704+0.8838i"), new Complex128("0.36612+0.06238i"), new Complex128("0.0969+0.59707i"), new Complex128("0.96099+0.87713i"), new Complex128("0.28446+0.45232i"), new Complex128("0.95354+0.87055i"), new Complex128("0.6418+0.68782i"), new Complex128("0.73455+0.04973i"), new Complex128("0.71273+0.48545i")},
                {new Complex128("0.8469+0.80951i"), new Complex128("0.71347+0.76274i"), new Complex128("0.35338+0.97138i"), new Complex128("0.50707+0.28844i"), new Complex128("0.23745+0.73877i"), new Complex128("0.82702+0.8458i"), new Complex128("0.08864+0.39934i"), new Complex128("0.02697+0.55562i"), new Complex128("0.20154+0.59157i"), new Complex128("0.6372+0.50199i"), new Complex128("0.98408+0.68102i"), new Complex128("0.32651+0.76286i"), new Complex128("0.19108+0.25295i"), new Complex128("0.25825+0.35311i"), new Complex128("0.75368+0.72706i")},
                {new Complex128("0.34448+0.04438i"), new Complex128("0.30705+0.53208i"), new Complex128("0.36304+0.99933i"), new Complex128("0.68529+0.11284i"), new Complex128("0.38368+0.8677i"), new Complex128("0.19719+0.13337i"), new Complex128("0.43275+0.56616i"), new Complex128("0.68267+0.99983i"), new Complex128("0.89103+0.16879i"), new Complex128("0.61772+0.67517i"), new Complex128("0.1031+0.83389i"), new Complex128("0.2915+0.45911i"), new Complex128("0.02961+0.30488i"), new Complex128("0.48733+0.58809i"), new Complex128("0.49888+0.6453i")},
                {new Complex128("0.75839+0.61182i"), new Complex128("0.76859+0.50697i"), new Complex128("0.28901+0.4631i"), new Complex128("0.63993+0.58199i"), new Complex128("0.76412+0.1298i"), new Complex128("0.31395+0.22497i"), new Complex128("0.20133+0.35658i"), new Complex128("0.81514+0.55195i"), new Complex128("0.66929+0.33394i"), new Complex128("0.12604+0.23074i"), new Complex128("0.47876+0.13968i"), new Complex128("0.88261+0.02473i"), new Complex128("0.15945+0.15319i"), new Complex128("0.09627+0.83235i"), new Complex128("0.11175+0.68569i")},
                {new Complex128("0.28752+0.19156i"), new Complex128("0.38266+0.30584i"), new Complex128("0.48181+0.10724i"), new Complex128("0.3457+0.31926i"), new Complex128("0.53183+0.37206i"), new Complex128("0.27003+0.4516i"), new Complex128("0.94186+0.43975i"), new Complex128("0.60834+0.37273i"), new Complex128("0.48761+0.48447i"), new Complex128("0.12911+0.98924i"), new Complex128("0.10905+0.72429i"), new Complex128("0.8166+0.76886i"), new Complex128("0.92743+0.06145i"), new Complex128("0.45529+0.97434i"), new Complex128("0.79746+0.75906i")},
                {new Complex128("0.38163+0.21341i"), new Complex128("0.92648+0.56462i"), new Complex128("0.0785+0.47142i"), new Complex128("0.46961+0.6336i"), new Complex128("0.73251+0.90862i"), new Complex128("0.87166+0.25754i"), new Complex128("0.6156+0.18451i"), new Complex128("0.53142+0.52466i"), new Complex128("0.90182+0.27299i"), new Complex128("0.79387+0.35015i"), new Complex128("0.12427+0.29163i"), new Complex128("0.31682+0.14227i"), new Complex128("0.18861+0.52372i"), new Complex128("0.47167+0.75696i"), new Complex128("0.86124+0.8967i")},
                {new Complex128("0.63451+0.55208i"), new Complex128("0.30552+0.90471i"), new Complex128("0.18693+0.52159i"), new Complex128("0.95717+0.41237i"), new Complex128("0.93016+0.78026i"), new Complex128("0.74042+0.8602i"), new Complex128("0.42962+0.4916i"), new Complex128("0.67385+0.18248i"), new Complex128("0.53536+0.84218i"), new Complex128("0.45487+0.74388i"), new Complex128("0.87338+0.13472i"), new Complex128("0.5436+0.17422i"), new Complex128("0.19605+0.12928i"), new Complex128("0.28215+0.03012i"), new Complex128("0.88307+0.24797i")},
                {new Complex128("0.82246+0.03127i"), new Complex128("0.59935+0.32426i"), new Complex128("0.94404+0.28634i"), new Complex128("0.93376+0.44786i"), new Complex128("0.40843+0.65209i"), new Complex128("0.08543+0.05725i"), new Complex128("0.35585+0.95449i"), new Complex128("0.43067+0.33572i"), new Complex128("0.39394+0.05754i"), new Complex128("0.44095+0.56715i"), new Complex128("0.02149+0.01673i"), new Complex128("0.2843+0.81159i"), new Complex128("0.96+0.85875i"), new Complex128("0.29013+0.25358i"), new Complex128("0.03828+0.06116i")}};
        A = new CMatrix(entries);
        exp = new Complex128("3.59504116227108-5.327292514154561i");
        act = ComplexDenseDeterminant.det(A);
        assertEquals(exp, act);

        // -------------- sub-case 5 --------------
        entries = new Complex128[][]{
                {new Complex128("1.0"), new Complex128("-53.0"), new Complex128("611.5"), new Complex128("234.6")},
                {new Complex128("-0.00351"), new Complex128("0.0"), new Complex128("24.56"), new Complex128("14.6")},
                {new Complex128("998.3"), new Complex128("2.5"), new Complex128("1.0"), new Complex128("0.00000025")}};
        A = new CMatrix(entries);
        assertThrows(LinearAlgebraException.class, ()->ComplexDenseDeterminant.det(A));

        // -------------- sub-case 6 --------------
        entries = new Complex128[][]{
                {new Complex128("1.0"), new Complex128("-53.0"), new Complex128("611.5")},
                {new Complex128("-0.00351"), new Complex128("0.0"), new Complex128("24.56")},
                {new Complex128("998.3"), new Complex128("2.5"), new Complex128("1.0")},
                {new Complex128("-92.5"), new Complex128("14.5"), new Complex128("7.1")}};
        A = new CMatrix(entries);
        assertThrows(LinearAlgebraException.class, ()->ComplexDenseDeterminant.det(A));
    }
}
