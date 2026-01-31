"""
Tests for color math utilities (deltaE).
"""

import numpy as np

from src.engine_v7.core.utils import cie76_deltaE, cie2000_deltaE


def test_cie76_deltae_zero():
    lab = np.array([[50.0, 0.0, 0.0]], dtype=np.float32)
    de = cie76_deltaE(lab, lab)
    assert np.isclose(de[0], 0.0, atol=1e-6)


def test_cie76_deltae_simple():
    lab1 = np.array([[50.0, 0.0, 0.0]], dtype=np.float32)
    lab2 = np.array([[60.0, 0.0, 0.0]], dtype=np.float32)
    de = cie76_deltaE(lab1, lab2)
    assert np.isclose(de[0], 10.0, atol=1e-6)


def test_cie2000_known_value_1():
    # Sharma et al. CIEDE2000 example
    lab1 = np.array([[50.0, 2.6772, -79.7751]], dtype=np.float32)
    lab2 = np.array([[50.0, 0.0, -82.7485]], dtype=np.float32)
    de = cie2000_deltaE(lab1, lab2)
    assert np.isclose(de[0], 2.0425, atol=0.05)


def test_cie2000_known_value_2():
    # Sharma et al. CIEDE2000 example
    lab1 = np.array([[50.0, 3.1571, -77.2803]], dtype=np.float32)
    lab2 = np.array([[50.0, 0.0, -82.7485]], dtype=np.float32)
    de = cie2000_deltaE(lab1, lab2)
    assert np.isclose(de[0], 2.8615, atol=0.05)
