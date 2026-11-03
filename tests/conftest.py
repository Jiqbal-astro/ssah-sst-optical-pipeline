import os
import sys

import numpy as np
import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)          # allow running the tests without installing


def gauss(img, x, y, flux, sigma):
    yy, xx = np.mgrid[:img.shape[0], :img.shape[1]]
    img += flux / (2 * np.pi * sigma ** 2) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))


def trail(img, x0, y0, x1, y1, flux, sigma, n=None):
    n = n or int(np.hypot(x1 - x0, y1 - y0) * 4) + 1
    for t in (np.arange(n) + 0.5) / n:
        gauss_stamp(img, x0 + t * (x1 - x0), y0 + t * (y1 - y0), flux / n, sigma)


def gauss_stamp(img, x, y, flux, sigma):
    r = int(6 * sigma) + 2
    ny, nx = img.shape
    xa, xb, ya, yb = max(int(x) - r, 0), min(int(x) + r + 1, nx), max(int(y) - r, 0), min(int(y) + r + 1, ny)
    yy, xx = np.mgrid[ya:yb, xa:xb]
    img[ya:yb, xa:xb] += flux / (2 * np.pi * sigma ** 2) * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * sigma ** 2))


@pytest.fixture
def rng():
    return np.random.default_rng(12345)
