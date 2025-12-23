#!/bin/bash
# UTF-8 인코딩 설정 스크립트 (MSYS2/Git Bash용)

echo "Setting UTF-8 encoding for MSYS terminal..."

# 환경 변수 설정
export LANG=ko_KR.UTF-8
export LC_ALL=ko_KR.UTF-8
export PYTHONIOENCODING=utf-8

# 현재 설정 확인
echo "Current encoding settings:"
echo "  LANG: $LANG"
echo "  LC_ALL: $LC_ALL"
echo "  PYTHONIOENCODING: $PYTHONIOENCODING"

# 한글 테스트
echo ""
echo "한글 테스트: 정상 출력되어야 합니다"
python -c "print('Python 한글 출력: 정상')"

echo ""
echo "설정 완료! 이제 한글이 정상적으로 표시됩니다."
echo "이 설정을 영구적으로 적용하려면 ~/.bashrc에 다음을 추가하세요:"
echo ""
echo "  export LANG=ko_KR.UTF-8"
echo "  export LC_ALL=ko_KR.UTF-8"
echo "  export PYTHONIOENCODING=utf-8"
