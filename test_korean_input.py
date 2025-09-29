#!/usr/bin/env python3
"""한글 입력 자소 분리 현상 테스트 및 해결"""

import sys
import unicodedata
import locale

def normalize_korean_text(text):
    """한글 텍스트를 NFC로 정규화하여 자소 분리 문제 해결"""
    return unicodedata.normalize('NFC', text)

def analyze_text(text):
    """텍스트의 유니코드 구성 분석"""
    print(f"\n분석할 텍스트: '{text}'")
    print(f"길이: {len(text)}")
    print(f"바이트: {text.encode('utf-8')}")

    # 각 문자 분석
    print("\n문자별 분석:")
    for i, char in enumerate(text):
        code = ord(char)
        category = unicodedata.category(char)
        try:
            name = unicodedata.name(char)
        except ValueError:
            name = "UNKNOWN"
        print(f"  [{i}] '{char}' (U+{code:04X}) - {category} - {name}")

    # 정규화 상태 확인
    nfc = unicodedata.normalize('NFC', text)
    nfd = unicodedata.normalize('NFD', text)

    print(f"\nNFC (조합형): '{nfc}' (길이: {len(nfc)})")
    print(f"NFD (분해형): '{nfd}' (길이: {len(nfd)})")

    if text != nfc:
        print("⚠️ 경고: 텍스트가 자소 분리 상태입니다!")
        print(f"✅ 해결: NFC 정규화 적용 -> '{nfc}'")
        return nfc
    else:
        print("✅ 텍스트가 정상적인 조합형 상태입니다.")
        return text

def main():
    print("=" * 60)
    print("한글 명령 프롬프트 입력 자소 분리 테스트")
    print("=" * 60)

    # 시스템 정보
    print(f"\n시스템 정보:")
    print(f"  Python 버전: {sys.version}")
    print(f"  기본 인코딩: {sys.getdefaultencoding()}")
    print(f"  로케일: {locale.getlocale()}")
    print(f"  stdin 인코딩: {sys.stdin.encoding}")
    print(f"  stdout 인코딩: {sys.stdout.encoding}")

    # 문제 사례 테스트
    print("\n" + "=" * 60)
    print("문제 사례 테스트")
    print("=" * 60)

    # 사용자가 언급한 문제 텍스트
    problem_text = "test ㄹㅡㄹ 통해서"
    print(f"\n문제 텍스트: '{problem_text}'")

    # 분석
    fixed = analyze_text(problem_text)

    # 올바른 텍스트와 비교
    correct_text = "test 를 통해서"
    print(f"\n기대하는 텍스트: '{correct_text}'")

    # 자소가 분리된 경우의 처리
    jamo_separated = "ㄹㅡㄹ"  # 자소로 분리된 "를"
    print(f"\n자소 분리된 '를': '{jamo_separated}'")
    print(f"자소 개수: {len(jamo_separated)}")

    # 올바른 한글 조합
    correct_reul = "를"
    print(f"올바른 '를': '{correct_reul}'")
    print(f"문자 개수: {len(correct_reul)}")

    print("\n" + "=" * 60)
    print("대화형 입력 테스트")
    print("=" * 60)

    try:
        user_input = input("\n한글을 입력해보세요 (예: 테스트를 통해서): ")

        print(f"\n입력받은 텍스트: '{user_input}'")
        normalized = analyze_text(user_input)

        if user_input != normalized:
            print(f"\n✅ 정규화된 텍스트: '{normalized}'")
            print("자소 분리 문제가 해결되었습니다!")

    except KeyboardInterrupt:
        print("\n\n테스트를 종료합니다.")
    except Exception as e:
        print(f"\n오류 발생: {e}")

if __name__ == "__main__":
    main()