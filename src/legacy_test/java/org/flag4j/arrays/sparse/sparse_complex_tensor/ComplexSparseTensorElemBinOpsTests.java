package org.flag4j.arrays.sparse.sparse_complex_tensor;

import org.flag4j.arrays.Shape;
import org.flag4j.arrays.sparse.CooCTensor;
import org.flag4j.numbers.Complex128;
import org.flag4j.util.exceptions.LinearAlgebraException;
import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertThrows;

class ComplexSparseTensorElemBinOpsTests {

    CooCTensor A;
    Shape aShape;
    Complex128[] aEntries;
    int[][] aIndices;

    CooCTensor B;
    Shape bShape;
    Complex128[] bEntries;
    int[][] bIndices;

    CooCTensor exp;
    Shape expShape;
    Complex128[] expEntries;
    int[][] expIndices;

    @Test
    void elemMultTest() {
        // ------------------------ sub-case 1 ------------------------
        aShape = new Shape(3, 4, 5);
        aEntries = new Complex128[]{new Complex128(0.3554027346190748, 0.17814806338585254), new Complex128(0.4679008020105414, 0.552234592711959), new Complex128(0.8023911579050015, 0.9369773311398271), new Complex128(0.5361245107886304, 0.8426048494330756), new Complex128(0.5116663669327101, 0.14226079103124079), new Complex128(0.6031886469745519, 0.5507006336519896), new Complex128(0.30173187058465345, 0.44768758628894023), new Complex128(0.47089846293340487, 0.12372049483112735), new Complex128(0.8332812297009098, 0.14164829949927782)};
        aIndices = new int[][]{
                {0, 1, 2},
                {0, 1, 4},
                {0, 2, 0},
                {0, 2, 2},
                {1, 2, 1},
                {1, 2, 3},
                {2, 0, 2},
                {2, 0, 4},
                {2, 3, 3}};
        A = new CooCTensor(aShape, aEntries, aIndices);

        bShape = new Shape(3, 4, 5);
        bEntries = new Complex128[]{new Complex128(0.641688013673216, 0.6529926214338411), new Complex128(0.2861597200166691, 0.7749988114606634), new Complex128(0.768889671662357, 0.9929463956356764), new Complex128(0.7183956240028645, 0.8537468523253172), new Complex128(0.027404474628255016, 0.4765076195110751), new Complex128(0.8646619116065113, 0.25569175902000085), new Complex128(0.8117289028941926, 0.050361733504989625), new Complex128(0.38088270679667247, 0.9284192721773488), new Complex128(0.724481029520081, 0.7172167668050126), new Complex128(0.38627225756249206, 0.9663366975684748), new Complex128(0.5492034964244268, 0.25026921484309506), new Complex128(0.7637541926826775, 0.3901141826226544), new Complex128(0.3099992969453177, 0.08109621445801574), new Complex128(0.6784683457632178, 0.9760264123930158), new Complex128(0.9404512577097293, 0.849340176282691), new Complex128(0.24859753416430375, 0.8897819832223416)};
        bIndices = new int[][]{
                {0, 1, 2},
                {0, 2, 0},
                {0, 2, 4},
                {1, 0, 1},
                {1, 1, 1},
                {1, 1, 2},
                {1, 1, 4},
                {1, 2, 0},
                {1, 2, 3},
                {1, 3, 3},
                {2, 0, 3},
                {2, 1, 2},
                {2, 1, 3},
                {2, 2, 3},
                {2, 3, 0},
                {2, 3, 1}};
        B = new CooCTensor(bShape, bEntries, bIndices);

        expShape = new Shape(3, 4, 5);
        expEntries = new Complex128[]{new Complex128(0.11172830391805331, 0.34639084027746325), new Complex128(-0.49654428890900426, 0.8899773644438604), new Complex128(0.04202700400959636, 0.8315891731821327)};
        expIndices = new int[][]{
                {0, 1, 2},
                {0, 2, 0},
                {1, 2, 3}};
        exp = new CooCTensor(expShape, expEntries, expIndices);

        assertEquals(exp, A.elemMult(B));

        // ------------------------ sub-case 2 ------------------------
        aShape = new Shape(3, 4, 5);
        aEntries = new Complex128[]{new Complex128(0.8090733448617681, 0.4514905847217746), new Complex128(0.5139505427121954, 0.1480659883061629), new Complex128(0.5538830826585828, 0.39096787547155687), new Complex128(0.01148067313092116, 0.3414711394381831), new Complex128(0.8287843463670005, 0.09420330954836731), new Complex128(0.34229624969929295, 0.986859223440192), new Complex128(0.5049744445088055, 0.305164515526821), new Complex128(0.8313739736384473, 0.24290072013606512), new Complex128(0.5498472829959157, 0.8566546560473144)};
        aIndices = new int[][]{
                {0, 1, 0},
                {0, 1, 2},
                {0, 3, 4},
                {1, 0, 1},
                {1, 2, 1},
                {1, 3, 1},
                {1, 3, 2},
                {2, 2, 4},
                {2, 3, 0}};
        A = new CooCTensor(aShape, aEntries, aIndices);

        bShape = new Shape(3, 4, 5);
        bEntries = new Complex128[]{new Complex128(0.06894633766030189, 0.5338176684872393), new Complex128(0.5749573099425286, 0.6336906486388725), new Complex128(0.3153452897134157, 0.47285192243696095)};
        bIndices = new int[][]{
                {1, 2, 4},
                {2, 0, 1},
                {2, 2, 3}};
        B = new CooCTensor(bShape, bEntries, bIndices);

        expShape = new Shape(3, 4, 5);
        expEntries = new Complex128[]{};
        expIndices = new int[][]{
        };
        exp = new CooCTensor(expShape, expEntries, expIndices);

        // ------------------------ sub-case 3 ------------------------
        aShape = new Shape(3, 4, 1, 2);
        A = new CooCTensor(aShape);

        bShape = new Shape(3, 4, 1, 2, 1);
        B = new CooCTensor(bShape);

        assertThrows(LinearAlgebraException.class, ()->A.elemMult(B));

        // ------------------------ sub-case 4 ------------------------
        aShape = new Shape(3, 4, 1, 2);
        A = new CooCTensor(aShape);

        bShape = new Shape(3, 12, 1, 2);
        B = new CooCTensor(bShape);

        assertThrows(LinearAlgebraException.class, ()->A.elemMult(B));
    }


    @Test
    void addTests() {
        aShape = new Shape(3, 4, 5);
        aEntries = new Complex128[]{new Complex128(0.46213195128555784, 0.001797636584227047), new Complex128(0.46406207172042513, 0.5621122046310777), new Complex128(0.5725470068968052, 0.9700746820433621), new Complex128(0.9843102251516707, 0.2404706780804564), new Complex128(0.9860977087720486, 0.021269606309532563), new Complex128(0.7283075275461912, 0.10972824977304207), new Complex128(0.030709421345043286, 0.9359805400567995), new Complex128(0.8846542888774562, 0.08743530919856501), new Complex128(0.09825508993525567, 0.7679866718281502)};
        aIndices = new int[][]{
                {0, 1, 4},
                {0, 2, 0},
                {0, 2, 2},
                {0, 2, 3},
                {1, 0, 2},
                {1, 0, 3},
                {1, 3, 3},
                {1, 3, 4},
                {2, 0, 1}};
        A = new CooCTensor(aShape, aEntries, aIndices);

        bShape = new Shape(3, 4, 5);
        bEntries = new Complex128[]{new Complex128(0.4178929367564942, 0.2097901696665011), new Complex128(0.9428201725612915, 0.8603359879641461), new Complex128(0.5230084456156013, 0.044652376148752726)};
        bIndices = new int[][]{
                {1, 1, 4},
                {1, 3, 1},
                {1, 3, 4}};
        B = new CooCTensor(bShape, bEntries, bIndices);

        expShape = new Shape(3, 4, 5);
        expEntries = new Complex128[]{new Complex128(0.46213195128555784, 0.001797636584227047), new Complex128(0.46406207172042513, 0.5621122046310777), new Complex128(0.5725470068968052, 0.9700746820433621), new Complex128(0.9843102251516707, 0.2404706780804564), new Complex128(0.9860977087720486, 0.021269606309532563), new Complex128(0.7283075275461912, 0.10972824977304207), new Complex128(0.4178929367564942, 0.2097901696665011), new Complex128(0.9428201725612915, 0.8603359879641461), new Complex128(0.030709421345043286, 0.9359805400567995), new Complex128(1.4076627344930577, 0.13208768534731774), new Complex128(0.09825508993525567, 0.7679866718281502)};
        expIndices = new int[][]{
                {0, 1, 4},
                {0, 2, 0},
                {0, 2, 2},
                {0, 2, 3},
                {1, 0, 2},
                {1, 0, 3},
                {1, 1, 4},
                {1, 3, 1},
                {1, 3, 3},
                {1, 3, 4},
                {2, 0, 1}};
        exp = new CooCTensor(expShape, expEntries, expIndices);
        assertEquals(exp, A.add(B));

        // ------------------------ sub-case 2 ------------------------
        aShape = new Shape(3, 2, 1, 2, 3);
        aEntries = new Complex128[]{new Complex128(0.6160738463715426, 0.1968914050559546), new Complex128(0.18076095086893407, 0.6136186390756889), new Complex128(0.772159908547494, 0.35655250177408504), new Complex128(0.4591634888667765, 0.6382625477088578), new Complex128(0.7179540532210842, 0.9920449208502049)};
        aIndices = new int[][]{
                {0, 0, 0, 1, 2},
                {0, 1, 0, 0, 0},
                {1, 0, 0, 0, 2},
                {1, 0, 0, 1, 0},
                {1, 0, 0, 1, 2}};
        A = new CooCTensor(aShape, aEntries, aIndices);

        bShape = new Shape(3, 2, 1, 2, 3);
        bEntries = new Complex128[]{new Complex128(0.763978006457422, 0.8959301015405149)};
        bIndices = new int[][]{
                {2, 0, 0, 1, 2}};
        B = new CooCTensor(bShape, bEntries, bIndices);

        expShape = new Shape(3, 2, 1, 2, 3);
        expEntries = new Complex128[]{new Complex128(0.6160738463715426, 0.1968914050559546), new Complex128(0.18076095086893407, 0.6136186390756889), new Complex128(0.772159908547494, 0.35655250177408504), new Complex128(0.4591634888667765, 0.6382625477088578), new Complex128(0.7179540532210842, 0.9920449208502049), new Complex128(0.763978006457422, 0.8959301015405149)};
        expIndices = new int[][]{
                {0, 0, 0, 1, 2},
                {0, 1, 0, 0, 0},
                {1, 0, 0, 0, 2},
                {1, 0, 0, 1, 0},
                {1, 0, 0, 1, 2},
                {2, 0, 0, 1, 2}};
        exp = new CooCTensor(expShape, expEntries, expIndices);
        assertEquals(exp, A.add(B));

        // ------------------------ sub-case 3 ------------------------
        aShape = new Shape(3, 4, 1, 2);
        A = new CooCTensor(aShape);

        bShape = new Shape(3, 4, 1, 2, 1);
        B = new CooCTensor(bShape);

        assertThrows(LinearAlgebraException.class, ()->A.add(B));

        // ------------------------ sub-case 4 ------------------------
        aShape = new Shape(3, 4, 1, 2);
        A = new CooCTensor(aShape);

        bShape = new Shape(3, 12, 1, 2);
        B = new CooCTensor(bShape);

        assertThrows(LinearAlgebraException.class, ()->A.add(B));
    }


    @Test
    void subTests() {
        // ------------------------ sub-case 1 ------------------------
        aShape = new Shape(3, 4, 5);
        aEntries = new Complex128[]{new Complex128(0.17614198390401192, 0.9506050536197737), new Complex128(0.31444950275574934, 0.7698169544040158), new Complex128(0.6367546698696576, 0.41787425228492703), new Complex128(0.7815522736625252, 0.7719692133463149), new Complex128(0.2944302437797989, 0.9771280639457677), new Complex128(0.3918838885588891, 0.9684349409321966), new Complex128(0.7006607124763364, 0.693548623559356), new Complex128(0.6655396902912524, 0.5003215039870123), new Complex128(0.21520203030603913, 0.6556408883606212)};
        aIndices = new int[][]{
                {0, 0, 2},
                {0, 0, 4},
                {0, 3, 4},
                {1, 1, 2},
                {1, 1, 4},
                {1, 2, 1},
                {2, 1, 0},
                {2, 1, 3},
                {2, 3, 1}};
        A = new CooCTensor(aShape, aEntries, aIndices);

        bShape = new Shape(3, 4, 5);
        bEntries = new Complex128[]{new Complex128(0.6093967458567611, 0.7786077333918526), new Complex128(0.7447643624118477, 0.6813810317763505), new Complex128(0.12151981120621747, 0.8591037209429343)};
        bIndices = new int[][]{
                {1, 0, 2},
                {1, 2, 2},
                {2, 2, 3}};
        B = new CooCTensor(bShape, bEntries, bIndices);

        expShape = new Shape(3, 4, 5);
        expEntries = new Complex128[]{new Complex128(0.17614198390401192, 0.9506050536197737), new Complex128(0.31444950275574934, 0.7698169544040158), new Complex128(0.6367546698696576, 0.41787425228492703), new Complex128(-0.6093967458567611, -0.7786077333918526), new Complex128(0.7815522736625252, 0.7719692133463149), new Complex128(0.2944302437797989, 0.9771280639457677), new Complex128(0.3918838885588891, 0.9684349409321966), new Complex128(-0.7447643624118477, -0.6813810317763505), new Complex128(0.7006607124763364, 0.693548623559356), new Complex128(0.6655396902912524, 0.5003215039870123), new Complex128(-0.12151981120621747, -0.8591037209429343), new Complex128(0.21520203030603913, 0.6556408883606212)};
        expIndices = new int[][]{
                {0, 0, 2},
                {0, 0, 4},
                {0, 3, 4},
                {1, 0, 2},
                {1, 1, 2},
                {1, 1, 4},
                {1, 2, 1},
                {1, 2, 2},
                {2, 1, 0},
                {2, 1, 3},
                {2, 2, 3},
                {2, 3, 1}};
        exp = new CooCTensor(expShape, expEntries, expIndices);
        assertEquals(exp, A.sub(B));

        // ------------------------ sub-case 2 ------------------------
        aShape = new Shape(3, 4, 2, 1, 5);
        aEntries = new Complex128[]{new Complex128(0.03099867357042463, 0.16406141241826533),
                new Complex128(0.40025480480395215, 0.3573176589499272),
                new Complex128(0.5829756217789057, 0.4565973342942867),
                new Complex128(0.7547388953906143, 0.07578782068550338),
                new Complex128(0.7641536538894649, 0.47648315038565714),
                new Complex128(0.21477832989403312, 0.8465035759034767),
                new Complex128(0.08855744900763618, 0.5455018436675453),
                new Complex128(0.26992640122762657, 0.03869668362006762),
                new Complex128(0.20360217842311057, 0.7965727538938915),
                new Complex128(0.194673963657913, 0.0009715427968347568),
                new Complex128(0.37937000326996617, 0.4216812678687796),
                new Complex128(0.5731671837653347, 0.7676951841845797),
                new Complex128(0.08121898085530821, 0.1787417726037127),
                new Complex128(0.33182392659238336, 0.6266462602129336),
                new Complex128(0.5777169151653782, 0.8267276078397521),
                new Complex128(0.6983057129995509, 0.3050526193484504),
                new Complex128(0.9457803294130197, 0.4347251210252875),
                new Complex128(0.9151417933380515, 0.1816520611532081)};
        aIndices = new int[][]{
                {0, 0, 1, 0, 3},
                {0, 2, 0, 0, 1},
                {0, 3, 0, 0, 2},
                {0, 3, 1, 0, 0},
                {0, 3, 1, 0, 1},
                {1, 0, 0, 0, 1},
                {1, 1, 0, 0, 2},
                {1, 1, 1, 0, 4},
                {1, 2, 0, 0, 3},
                {1, 2, 1, 0, 4},
                {2, 0, 0, 0, 3},
                {2, 0, 1, 0, 2},
                {2, 0, 1, 0, 3},
                {2, 1, 0, 0, 4},
                {2, 1, 1, 0, 0},
                {2, 2, 0, 0, 4},
                {2, 3, 0, 0, 0},
                {2, 3, 1, 0, 0}};
        A = new CooCTensor(aShape, aEntries, aIndices);

        bShape = new Shape(3, 4, 2, 1, 5);
        bEntries = new Complex128[]{new Complex128(0.9786839486132947, 0.009654575491062634)};
        bIndices = new int[][]{
                {2, 2, 1, 0, 4}};
        B = new CooCTensor(bShape, bEntries, bIndices);

        expShape = new Shape(3, 4, 2, 1, 5);
        expEntries = new Complex128[]{new Complex128(0.03099867357042463, 0.16406141241826533), new Complex128(0.40025480480395215, 0.3573176589499272), new Complex128(0.5829756217789057, 0.4565973342942867), new Complex128(0.7547388953906143, 0.07578782068550338), new Complex128(0.7641536538894649, 0.47648315038565714), new Complex128(0.21477832989403312, 0.8465035759034767), new Complex128(0.08855744900763618, 0.5455018436675453), new Complex128(0.26992640122762657, 0.03869668362006762), new Complex128(0.20360217842311057, 0.7965727538938915), new Complex128(0.194673963657913, 0.0009715427968347568), new Complex128(0.37937000326996617, 0.4216812678687796), new Complex128(0.5731671837653347, 0.7676951841845797), new Complex128(0.08121898085530821, 0.1787417726037127), new Complex128(0.33182392659238336, 0.6266462602129336), new Complex128(0.5777169151653782, 0.8267276078397521), new Complex128(0.6983057129995509, 0.3050526193484504), new Complex128(-0.9786839486132947, -0.009654575491062634), new Complex128(0.9457803294130197, 0.4347251210252875), new Complex128(0.9151417933380515, 0.1816520611532081)};
        expIndices = new int[][]{
                {0, 0, 1, 0, 3},
                {0, 2, 0, 0, 1},
                {0, 3, 0, 0, 2},
                {0, 3, 1, 0, 0},
                {0, 3, 1, 0, 1},
                {1, 0, 0, 0, 1},
                {1, 1, 0, 0, 2},
                {1, 1, 1, 0, 4},
                {1, 2, 0, 0, 3},
                {1, 2, 1, 0, 4},
                {2, 0, 0, 0, 3},
                {2, 0, 1, 0, 2},
                {2, 0, 1, 0, 3},
                {2, 1, 0, 0, 4},
                {2, 1, 1, 0, 0},
                {2, 2, 0, 0, 4},
                {2, 2, 1, 0, 4},
                {2, 3, 0, 0, 0},
                {2, 3, 1, 0, 0}};
        exp = new CooCTensor(expShape, expEntries, expIndices);
        assertEquals(exp, A.sub(B));

        // ------------------------ sub-case 3 ------------------------
        aShape = new Shape(3, 4, 1, 2);
        A = new CooCTensor(aShape);

        bShape = new Shape(3, 4, 1, 2, 1);
        B = new CooCTensor(bShape);

        assertThrows(LinearAlgebraException.class, ()->A.sub(B));

        // ------------------------ sub-case 4 ------------------------
        aShape = new Shape(3, 4, 1, 2);
        A = new CooCTensor(aShape);

        bShape = new Shape(3, 12, 1, 2);
        B = new CooCTensor(bShape);

        assertThrows(LinearAlgebraException.class, ()->A.sub(B));
    }
}
