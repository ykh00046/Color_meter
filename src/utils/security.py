"""
Security Utilities

입력 검증 및 보안 유틸리티
"""

import re
from pathlib import Path


class SecurityError(Exception):
    """보안 검증 실패 예외"""

    pass


def validate_sku_identifier(sku: str) -> str:
    """
    SKU 식별자 검증 (경로 트래버설 차단)

    Args:
        sku: SKU 식별자

    Returns:
        검증된 SKU 문자열

    Raises:
        SecurityError: SKU 형식이 유효하지 않을 때

    Example:
        >>> validate_sku_identifier("SKU001")
        'SKU001'
        >>> validate_sku_identifier("../../etc/passwd")
        SecurityError: Invalid SKU format
    """
    # SKU 형식: 영숫자, 하이픈, 언더스코어만 허용 (최대 50자)
    # 예: SKU001, SKU_TEST, SKU-2024-01
    pattern = r"^[A-Za-z0-9_-]{1,50}$"

    if not isinstance(sku, str):
        raise SecurityError("SKU must be a string")

    if not sku:
        raise SecurityError("SKU cannot be empty")

    if not re.match(pattern, sku):
        raise SecurityError(
            f"Invalid SKU format: '{sku}'. "
            f"Only alphanumeric characters, hyphens, and underscores allowed (max 50 chars)"
        )

    # 추가 검증: .. 경로 트래버설 명시적 차단
    if ".." in sku or "/" in sku or "\\" in sku:
        raise SecurityError(f"Path traversal attempt detected in SKU: '{sku}'")

    return sku


def safe_sku_path(sku: str, config_dir: Path = Path("config/sku_db")) -> Path:
    """
    안전한 SKU 설정 파일 경로 생성

    Args:
        sku: SKU 식별자
        config_dir: SKU 설정 디렉토리

    Returns:
        검증된 절대 경로

    Raises:
        SecurityError: SKU 검증 실패 또는 경로 이탈 시

    Example:
        >>> safe_sku_path("SKU001")
        PosixPath('/absolute/path/config/sku_db/SKU001.json')
    """
    # 1. SKU 검증
    validated_sku = validate_sku_identifier(sku)

    # 2. 경로 생성
    config_dir = Path(config_dir).resolve()  # 절대 경로로 변환
    sku_path = (config_dir / f"{validated_sku}.json").resolve()

    # 3. 경로 이탈 검증 (config_dir 벗어나면 차단)
    try:
        sku_path.relative_to(config_dir)
    except ValueError:
        raise SecurityError(f"Path traversal detected: '{sku_path}' is outside '{config_dir}'")

    return sku_path


def validate_file_extension(filename: str, allowed_extensions: list = None) -> bool:
    """
    파일 확장자 검증

    Args:
        filename: 파일명
        allowed_extensions: 허용된 확장자 리스트 (예: ['.jpg', '.png'])

    Returns:
        유효 여부

    Example:
        >>> validate_file_extension("image.jpg", ['.jpg', '.png'])
        True
        >>> validate_file_extension("script.exe", ['.jpg', '.png'])
        False
    """
    if allowed_extensions is None:
        allowed_extensions = [".jpg", ".jpeg", ".png"]

    ext = Path(filename).suffix.lower()
    return ext in allowed_extensions


def validate_file_size(file_size: int, max_size_mb: int = 10) -> bool:
    """
    파일 크기 검증

    Args:
        file_size: 파일 크기 (bytes)
        max_size_mb: 최대 크기 (MB)

    Returns:
        유효 여부

    Example:
        >>> validate_file_size(5_000_000, max_size_mb=10)  # 5MB
        True
        >>> validate_file_size(15_000_000, max_size_mb=10)  # 15MB
        False
    """
    max_size_bytes = max_size_mb * 1024 * 1024
    return file_size <= max_size_bytes


def sanitize_filename(filename: str) -> str:
    """
    파일명 안전화 (Path traversal 방지)

    디렉토리 경로 제거, 위험 문자 치환, 빈 이름 처리

    Args:
        filename: 원본 파일명 (경로 포함 가능)

    Returns:
        안전한 파일명 (basename만, 영숫자/._- 만 허용)

    Example:
        >>> sanitize_filename("../../../etc/passwd")
        'etc_passwd'
        >>> sanitize_filename("image (1).jpg")
        'image__1_.jpg'
        >>> sanitize_filename("")
        'unnamed'
    """
    # 1. Extract basename only (remove directory path)
    safe_name = Path(filename).name

    # 2. Replace unsafe characters with underscore
    safe_name = re.sub(r"[^A-Za-z0-9._-]", "_", safe_name)

    # 3. Remove leading dots (hidden files / path tricks)
    safe_name = safe_name.lstrip(".")

    # 4. Ensure not empty
    if not safe_name:
        safe_name = "unnamed"

    # 5. Limit length (filesystem safety)
    if len(safe_name) > 200:
        ext = Path(safe_name).suffix
        safe_name = safe_name[: 200 - len(ext)] + ext

    return safe_name
