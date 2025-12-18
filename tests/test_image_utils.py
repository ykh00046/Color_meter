import numpy as np

from src.utils import image_utils


def test_bgr_rgb_conversion(sample_image):
    bgr = sample_image.copy()
    bgr[:, :, 0] = 255  # blue channel
    rgb = image_utils.bgr_to_rgb(bgr)
    bgr_back = image_utils.rgb_to_bgr(rgb)
    assert np.array_equal(bgr, bgr_back)


def test_resize_keep_aspect(sample_image):
    # Create a wide image (200H x 500W) to test width scaling
    big = np.zeros((200, 500, 3), dtype=np.uint8)
    resized = image_utils.resize_keep_aspect(big, max_width=300, max_height=300)
    # Width should be scaled to 300, height scaled proportionally to 120
    assert resized.shape[1] == 300  # width
    assert resized.shape[0] < 300  # height (should be 120)


def test_draw_circle(sample_image):
    out = image_utils.draw_circle(sample_image, center=(50, 50), radius=10)
    # 원을 그리면 픽셀 변화가 발생한다
    assert not np.array_equal(out, sample_image)


def test_validate_raises_on_wrong_type():
    try:
        image_utils.bgr_to_rgb("not-an-image")  # type: ignore[arg-type]
    except image_utils.ImageValidationError:
        pass
    else:
        raise AssertionError("ImageValidationError expected")
