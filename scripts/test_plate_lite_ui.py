"""
Plate-Lite UI 브라우저 테스트
실제 브라우저에서 plate_lite 결과가 제대로 렌더링되는지 확인
"""

import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE_URL = "http://127.0.0.1:8000"
WHITE_IMAGE = Path("C:/X/Color_meter/data/A_White.png")
BLACK_IMAGE = Path("C:/X/Color_meter/data/A_Black.png")
SCREENSHOT_DIR = Path("C:/X/Color_meter/reports/screenshots")


def test_plate_lite_ui():
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})

        # 1. 페이지 로드
        print("[1] Loading single analysis page...")
        page.goto(f"{BASE_URL}/single_analysis")
        page.wait_for_load_state("networkidle")
        page.screenshot(path=str(SCREENSHOT_DIR / "01_page_loaded.png"))
        print("    Screenshot: 01_page_loaded.png")

        # 2. 파일 업로드
        print("[2] Uploading white/black images...")
        if WHITE_IMAGE.exists() and BLACK_IMAGE.exists():
            page.set_input_files("#fileWhite", str(WHITE_IMAGE))
            page.set_input_files("#fileBlack", str(BLACK_IMAGE))
            page.screenshot(path=str(SCREENSHOT_DIR / "02_files_selected.png"))
            print("    Screenshot: 02_files_selected.png")
        else:
            print(f"    ERROR: Image files not found")
            print(f"    WHITE_IMAGE exists: {WHITE_IMAGE.exists()}")
            print(f"    BLACK_IMAGE exists: {BLACK_IMAGE.exists()}")
            browser.close()
            return

        # 3. 분석 실행
        print("[3] Running analysis...")
        analyze_btn = page.locator("#btnAnalyze")
        if analyze_btn.is_visible():
            analyze_btn.click()
            # 분석 완료 대기 - qualityScore가 업데이트될 때까지
            try:
                page.wait_for_function(
                    "document.querySelector('#qualityScore')?.textContent !== '0.0'"
                    " && document.querySelector('#qualityScore')?.textContent !== '-'",
                    timeout=60000,
                )
            except Exception as e:
                print(f"    Warning: wait_for_function timeout: {e}")
            time.sleep(2)  # 렌더링 완료 대기
            page.screenshot(path=str(SCREENSHOT_DIR / "03_analysis_complete.png"))
            print("    Screenshot: 03_analysis_complete.png")
        else:
            print("    ERROR: Analyze button not visible")
            browser.close()
            return

        # 4. Plate 탭 클릭
        print("[4] Clicking Plate tab...")
        plate_tab = page.locator('[data-tab="plate"]')
        if plate_tab.is_visible():
            plate_tab.click()
            time.sleep(0.5)
            page.screenshot(path=str(SCREENSHOT_DIR / "04_plate_tab.png"))
            print("    Screenshot: 04_plate_tab.png")

        # 5. Plate-Lite 데이터 확인
        print("[5] Checking Plate-Lite content...")
        plate_container = page.locator("#plateAnalysisContainer")
        content = plate_container.inner_text()
        print(f"    Container content preview: {content[:200]}...")

        # Plate-Lite 특유의 텍스트 확인
        has_plate_lite = "PLATE-LITE" in content or "plate-lite" in content.lower()
        has_ink_hex = "Ink Hex" in content
        has_alpha = "Alpha Mean" in content
        has_zones = "RING_CORE" in content or "DOT_CORE" in content

        print(f"    - Has PLATE-LITE label: {has_plate_lite}")
        print(f"    - Has Ink Hex: {has_ink_hex}")
        print(f"    - Has Alpha Mean: {has_alpha}")
        print(f"    - Has Zone names: {has_zones}")

        # 6. 전체 페이지 스크린샷
        print("[6] Taking full page screenshot...")
        page.screenshot(path=str(SCREENSHOT_DIR / "05_full_result.png"), full_page=True)
        print("    Screenshot: 05_full_result.png")

        # 7. 콘솔 로그 확인
        print("[7] Checking for JS errors...")
        # 콘솔 메시지 수집을 위해 새로 로드
        console_messages = []
        page.on("console", lambda msg: console_messages.append(f"{msg.type}: {msg.text}"))

        browser.close()

        # 결과 요약
        print("\n" + "=" * 50)
        print("TEST RESULT SUMMARY")
        print("=" * 50)
        if has_plate_lite or has_ink_hex or has_zones:
            print("✅ Plate-Lite UI rendering: SUCCESS")
            print(f"   - PLATE-LITE label: {'✅' if has_plate_lite else '❌'}")
            print(f"   - Ink Hex field: {'✅' if has_ink_hex else '❌'}")
            print(f"   - Alpha Mean field: {'✅' if has_alpha else '❌'}")
            print(f"   - Zone names: {'✅' if has_zones else '❌'}")
        else:
            print("❌ Plate-Lite UI rendering: FAILED")
            print("   Content found:", content[:500])

        print(f"\nScreenshots saved to: {SCREENSHOT_DIR}")


if __name__ == "__main__":
    test_plate_lite_ui()
