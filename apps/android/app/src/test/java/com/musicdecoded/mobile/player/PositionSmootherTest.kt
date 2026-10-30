package com.musicdecoded.mobile.player

import org.junit.Assert.assertEquals
import org.junit.Test

class PositionSmootherTest {
    private val frame = 16_666_667L

    @Test fun extrapolatesBetweenCoarseSteps() {
        val smoother = PositionSmoother()
        assertEquals(10.0, smoother.position(10.0, 0, 1f), 1e-9)
        assertEquals(10.0 + frame / 1e9, smoother.position(10.0, frame, 1f), 1e-9)
        assertEquals(10.1, smoother.position(10.0, 100_000_000, 1f), 1e-9)
    }

    @Test fun followsTheSpeed() {
        val smoother = PositionSmoother()
        smoother.position(10.0, 0, 0.5f)
        assertEquals(10.05, smoother.position(10.0, 100_000_000, 0.5f), 1e-9)
    }

    @Test fun stopsExtrapolatingWhenTheSourceStalls() {
        val smoother = PositionSmoother(maxLead = 0.5)
        smoother.position(10.0, 0, 1f)
        assertEquals(10.5, smoother.position(10.0, 3_000_000_000, 1f), 1e-9)
    }

    @Test fun neverStepsBackALittle() {
        val smoother = PositionSmoother()
        smoother.position(10.0, 0, 1f)
        assertEquals(10.25, smoother.position(10.0, 250_000_000, 1f), 1e-9)
        // The next step lands just behind the estimate: hold, then move on from the new step.
        assertEquals(10.25, smoother.position(10.2, 260_000_000, 1f), 1e-9)
        assertEquals(10.3, smoother.position(10.2, 360_000_000, 1f), 1e-9)
    }

    @Test fun bigJumpsPassThrough() {
        val smoother = PositionSmoother()
        smoother.position(10.0, 0, 1f)
        smoother.position(10.0, 200_000_000, 1f)
        assertEquals(3.0, smoother.position(3.0, 210_000_000, 1f), 1e-9)
        assertEquals(42.0, smoother.position(42.0, 220_000_000, 1f), 1e-9)
    }

    @Test fun resetForgetsTheLastValue() {
        val smoother = PositionSmoother()
        smoother.position(10.0, 0, 1f)
        smoother.position(10.0, 200_000_000, 1f)
        smoother.reset()
        assertEquals(10.1, smoother.position(10.1, 210_000_000, 1f), 1e-9)
    }
}
