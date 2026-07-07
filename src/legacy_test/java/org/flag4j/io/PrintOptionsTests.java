package org.flag4j.io;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class PrintOptionsTests {

    @Test
    void printOptionTestCase() {
        assertEquals(2, PrintOptions.getPadding());
        assertEquals(10, PrintOptions.getMaxRows());
        assertEquals(10, PrintOptions.getMaxColumns());
        assertEquals(8, PrintOptions.getPrecision());
        assertTrue(PrintOptions.useCentering());

        PrintOptions.setPadding(14);
        PrintOptions.setMaxRows(15);
        PrintOptions.setMaxColumns(100);
        PrintOptions.setPrecision(134);
        PrintOptions.setCentering(false);

        assertEquals(14, PrintOptions.getPadding());
        assertEquals(15, PrintOptions.getMaxRows());
        assertEquals(100, PrintOptions.getMaxColumns());
        assertEquals(134, PrintOptions.getPrecision());
        assertFalse(PrintOptions.useCentering());

        assertThrows(IllegalArgumentException.class, () -> PrintOptions.setPadding(-1));
        assertThrows(IllegalArgumentException.class, () -> PrintOptions.setMaxRows(-1));
        assertThrows(IllegalArgumentException.class, () -> PrintOptions.setMaxColumns(-1));
        assertThrows(IllegalArgumentException.class, () -> PrintOptions.setPrecision(-1));

        PrintOptions.resetAll();
    }
}
