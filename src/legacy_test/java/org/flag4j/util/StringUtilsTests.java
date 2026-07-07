package org.flag4j.util;


import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;

class StringUtilsTests {

    int size;
    String pad;
    String str;
    String expStr;

    @Test
    void centerTestCase() {
        // --------------- sub-case 1 ---------------
        size = 13;
        str = "hello world";
        expStr = " hello world ";
        assertEquals(expStr, StringUtils.center(str, size));

        // --------------- sub-case 2 ---------------
        size = 5;
        str = "hello world";
        expStr = "hello world";
        assertEquals(expStr, StringUtils.center(str, size));

        // --------------- sub-case 3 ---------------
        size = 17;
        str = "hello world";
        expStr = "   hello world   ";
        assertEquals(expStr, StringUtils.center(str, size));

        // --------------- sub-case 4 ---------------
        size = 13;
        pad = "-";
        str = "hello world";
        expStr = "-hello world-";
        assertEquals(expStr, StringUtils.center(str, size, pad));

        // --------------- sub-case 5 ---------------
        size = 5;
        pad = "-";
        str = "hello world";
        expStr = "hello world";
        assertEquals(expStr, StringUtils.center(str, size, pad));

        // --------------- sub-case 6 ---------------
        size = 17;
        pad = "-";
        str = "hello world";
        expStr = "---hello world---";
        assertEquals(expStr, StringUtils.center(str, size, pad));
    }
}
