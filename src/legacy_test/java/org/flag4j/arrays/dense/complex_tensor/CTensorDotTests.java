package org.flag4j.arrays.dense.complex_tensor;

import org.flag4j.numbers.Complex128;
import org.flag4j.arrays.Shape;
import org.flag4j.arrays.dense.CTensor;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class CTensorDotTests {

    static int[] aAxes;
    static int[] bAxes;

    static Complex128[] aEntries;
    static Complex128[] bEntries;
    static Complex128[] expEntries;

    static Shape aShape;
    static Shape bShape;
    static Shape expShape;

    static CTensor A;
    static CTensor B;
    static CTensor exp;

    @BeforeEach
    void setup() {
        aEntries = new Complex128[]{
                new Complex128("1.4415+9.14i"), new Complex128("235.61-9.865i"), new Complex128("-0.00024+5.15i"),
                new Complex128("1.0"), new Complex128("-0.0-85.1i"), new Complex128("1.345+3.5i"),
                new Complex128("0.014+0.01i"), new Complex128("-140.0-0.0235i"), new Complex128("1.5+9.24i"),
                new Complex128("51.0"), new Complex128("6.1-265.55i"), new Complex128("-0.00014+4.14i")};
        bEntries = new Complex128[]{
                new Complex128("13.41+3.4i"), new Complex128("-99.23"), new Complex128("14.0+0.094i"), new Complex128("0.000245-9.0i"),
                new Complex128("0.0+1.25i"), new Complex128("95.14"), new Complex128("0.0+546.0i"), new Complex128("15.6+9234.6i"),
                new Complex128("1.45656"), new Complex128("0.009345"), new Complex128("156.21326-0.00351i"), new Complex128("125.6"),
                new Complex128("144.5+40.1i"), new Complex128("-1545.4+2.0i"), new Complex128("1.145-8.25i"), new Complex128("7.4"),
                new Complex128("-9.345"), new Complex128("0.0"), new Complex128("314.5"), new Complex128("0.0+4.155i"),
                new Complex128("-0.0"), new Complex128("145.5"), new Complex128("7.0"), new Complex128("1.4456+103.525i")
        };

        aShape = new Shape(3, 2, 1, 2);
        bShape = new Shape(4, 3, 2);

        A = new CTensor(aShape, aEntries);
        B = new CTensor(bShape, bEntries);
    }


    @Test
    void multiAxesTestCase() {
        // --------------------- sub-case 1 ---------------------
        aAxes = new int[]{0};
        bAxes = new int[]{1};
        expEntries = new Complex128[]{
                new Complex128("-15.296085 - 1062.0565i"), new Complex128("-766.2300449999999 - 27.88944950000007i"), new Complex128("-4756.0876776000005 + 2106.5110014i"),
                new Complex128("-84193.35660000001 + 14614.0086405i"), new Complex128("-874.30975 + 1194.74685i"), new Complex128("-2245.9741000000004 - 14751.813000000002i"),
                new Complex128("463.85175 + 2939.21i"), new Complex128("-992.3793000000001 - 12207.415723499998i"), new Complex128("3211.572100000001 + 781.6607800000002i"),
                new Complex128("-18495.939970475003 + 966.7998075i"), new Complex128("13355.125333199998 + 128647.97895i"), new Complex128("101180.45756902502 + 2175610.2447075i"),
                new Complex128("33995.051525 + 8015.379750000001i"), new Complex128("-364082.01100000006 + 15742.491i"), new Complex128("74456.345 - 3102.5425i"),
                new Complex128("310.412175 + 6767.984550000001i"), new Complex128("314.6193416 + 76.82700000000001i"), new Complex128("580.4678186299999 - 25775.58749755i"),
                new Complex128("-1859.9108026600004 - 41482.569078399996i"), new Complex128("-46792.03361317 - 33274.95621055i"), new Complex128("-263.45565 + 3225.6260760000005i"),
                new Complex128("-9.825504 - 7958.736480000001i"), new Complex128("42.62452 - 239.17499999999995i"), new Complex128("27480.52066 + 249.07742280000008i"),
                new Complex128("-1951.7627909999999 - 10.089175000000001i"), new Complex128("-99.48911960000001 + 1653.8795942425i"), new Complex128("-203.9257384564 + 1192.6886677314i"),
                new Complex128("14.274116000000001 + 9754.5837803925i"), new Complex128("-15.992566700000003 + 1156.3847924999998i"), new Complex128("-2581.4 + 1.8261i"),
                new Complex128("314.49902 + 28.979999999999997i"), new Complex128("-20798.593702384 + 6.706040499999999i")};
        expShape = new Shape(2, 1, 2, 4, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.tensorDot(B, aAxes, bAxes));

        // --------------------- sub-case 2 ---------------------
        aAxes = new int[]{0, 1};
        bAxes = new int[]{1, 2};
        expEntries = new Complex128[]{
                new Complex128("565.17173363 - 26837.64399755i"), new Complex128("-51548.12129077 - 31168.445209150002i"), new Complex128("-884.1352539999999 - 6763.989630000002i"),
                new Complex128("27944.37241 + 3188.2874228i"), new Complex128("3112.0829804000005 + 2435.5403742425i"), new Complex128("13369.3994492 + 138402.56273039253i"),
                new Complex128("31413.651525 + 8017.205850000001i"), new Complex128("53657.751297616 - 3095.8364595i")
        };
        expShape = new Shape(1, 2, 4);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.tensorDot(B, aAxes, bAxes));

        // --------------------- sub-case 3 ---------------------
        aAxes = new int[]{1};
        bAxes = new int[]{0};

        assertThrows(LinearAlgebraException.class, ()->A.tensorDot(B, aAxes, bAxes));

        // --------------------- sub-case 4 ---------------------
        aAxes = new int[]{0, 1};
        bAxes = new int[]{1};

        assertThrows(IllegalArgumentException.class, ()->A.tensorDot(B, aAxes, bAxes));

        // --------------------- sub-case 5 ---------------------
        aAxes = new int[]{0, 1};
        bAxes = new int[]{1, 0};

        assertThrows(LinearAlgebraException.class, ()->A.tensorDot(B, aAxes, bAxes));

        // --------------------- sub-case 5 ---------------------
        aAxes = new int[]{0, 1, 2, 3};
        bAxes = new int[]{1, 0, 3, 2};

        assertThrows(LinearAlgebraException.class, ()->A.tensorDot(B, aAxes, bAxes));
    }


    @Test
    void singleAxesTestCase() {
        // --------------------- sub-case 1 ---------------------
        expEntries = new Complex128[]{
                new Complex128("-15.296085 - 1062.0565i"), new Complex128("-766.2300449999999 - 27.88944950000007i"), new Complex128("-4756.0876776000005 + 2106.5110014i"),
                new Complex128("-84193.35660000001 + 14614.0086405i"), new Complex128("-874.30975 + 1194.74685i"), new Complex128("-2245.9741000000004 - 14751.813000000002i"),
                new Complex128("463.85175 + 2939.21i"), new Complex128("-992.3793000000001 - 12207.415723499998i"), new Complex128("3211.572100000001 + 781.6607800000002i"),
                new Complex128("-18495.939970475003 + 966.7998075i"), new Complex128("13355.125333199998 + 128647.97895i"), new Complex128("101180.45756902502 + 2175610.2447075i"),
                new Complex128("33995.051525 + 8015.379750000001i"), new Complex128("-364082.01100000006 + 15742.491i"), new Complex128("74456.345 - 3102.5425i"),
                new Complex128("310.412175 + 6767.984550000001i"), new Complex128("314.6193416 + 76.82700000000001i"), new Complex128("580.4678186299999 - 25775.58749755i"),
                new Complex128("-1859.9108026600004 - 41482.569078399996i"), new Complex128("-46792.03361317 - 33274.95621055i"), new Complex128("-263.45565 + 3225.6260760000005i"),
                new Complex128("-9.825504 - 7958.736480000001i"), new Complex128("42.62452 - 239.17499999999995i"), new Complex128("27480.52066 + 249.07742280000008i"),
                new Complex128("-1951.7627909999999 - 10.089175000000001i"), new Complex128("-99.48911960000001 + 1653.8795942425i"), new Complex128("-203.9257384564 + 1192.6886677314i"),
                new Complex128("14.274116000000001 + 9754.5837803925i"), new Complex128("-15.992566700000003 + 1156.3847924999998i"), new Complex128("-2581.4 + 1.8261i"),
                new Complex128("314.49902 + 28.979999999999997i"), new Complex128("-20798.593702384 + 6.706040499999999i")};
        expShape = new Shape(2, 1, 2, 4, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.tensorDot(B, 0, 1));

        // --------------------- sub-case 1 ---------------------
        assertThrows(LinearAlgebraException.class, ()->A.tensorDot(B, 1, 1));
    }


    @Test
    void noAxesTestCase() {
        // --------------------- sub-case 1 ---------------------
        A = A.reshape(2, 2, 3);
        expEntries = new Complex128[]{
                new Complex128("3281.284325 + 11.505539999999993i"), new Complex128("-231.79015415 - 2537.483616925i"), new Complex128("-4647.2793130824 + 1577.1883254424001i"),
                new Complex128("-84379.58496855001 + 14101.007711575001i"), new Complex128("30.1721928 - 624.6705250000001i"), new Complex128("-502.46010000000024 - 14195.074000000002i"),
                new Complex128("453.35006999999996 + 2910.5800000000004i"), new Complex128("33710.124203056 - 1421.9480735i"), new Complex128("17.034399999999998 - 1186.3187499999997i"),
                new Complex128("-737.1667 + 332.9691505i"), new Complex128("210.11911969999997 + 968.78843305i"), new Complex128("184.53199999999998 + 9673.4047405i"),
                new Complex128("-570.1440249999999 - 90.047i"), new Complex128("-1545.4 - 627.74i"), new Complex128("323.915 + 24.5i"),
                new Complex128("-360.39316800000006 - 12233.594275i"), new Complex128("-1971.394051 - 11.432300000000001i"), new Complex128("141.07498 + 2138.1012942425i"),
                new Complex128("24.973922399999964 + 1451.01502824i"), new Complex128("94.96409999999997 + 1289.9841803924999i"), new Complex128("-172.889375 + 1070.6316924999999i"),
                new Complex128("-1057.6556 - 15.5999i"), new Complex128("14.903 + 67.825i"), new Complex128("-21324.444150000003 + 165.28376400000005i"),
                new Complex128("789.0967 - 3543.726775i"), new Complex128("-7450.691825100001 + 338.91454024999996i"), new Complex128("8.8776775436 + 28105.933388891397i"),
                new Complex128("795.6394204999999 + 471482.10243525i"), new Complex128("5185.6983083000005 + 1652.03195i"), new Complex128("-78770.26000000001 - 1863.0700000000002i"),
                new Complex128("16039.49902 + 28.979999999999997i"), new Complex128("458.956297616 - 38419.6497095i")};
        expShape = new Shape(2, 2, 4, 2);
        exp = new CTensor(expShape, expEntries);

        assertEquals(exp, A.tensorDot(B));

        // --------------------- sub-case 1 ---------------------
        A = A.reshape(2, 2, 3);
        B = B.reshape(4, 2, 3);
        assertThrows(LinearAlgebraException.class, ()->A.tensorDot(B));
    }
}
