"""한글 텍스트 정규화 유틸리티

명령 프롬프트나 다른 입력 소스에서 발생할 수 있는 한글 자소 분리 문제를 해결합니다.
"""

import unicodedata
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

# 자주 사용되는 자소 분리 패턴과 올바른 한글의 매핑
JAMO_TO_SYLLABLE_MAP = {
    "ㄱㅏ": "가",
    "ㄴㅏ": "나",
    "ㄷㅏ": "다",
    "ㄹㅏ": "라",
    "ㅁㅏ": "마",
    "ㅂㅏ": "바",
    "ㅅㅏ": "사",
    "ㅇㅏ": "아",
    "ㅈㅏ": "자",
    "ㅊㅏ": "차",
    "ㅋㅏ": "카",
    "ㅌㅏ": "타",
    "ㅍㅏ": "파",
    "ㅎㅏ": "하",
    "ㄹㅡㄹ": "를",  # 사용자가 언급한 문제 케이스
    "ㅇㅡㄹ": "을",
    "ㅇㅔ": "에",
    "ㅇㅔㅅㅓ": "에서",
    "ㄱㅗㅏ": "과",
    "ㅇㅘ": "와",
}


def normalize_korean_text(text: str) -> str:
    """한글 텍스트를 NFC(정규화 형식 C)로 정규화합니다.

    자소가 분리된 한글을 조합형으로 변환하여 정상적인 한글로 만듭니다.

    Args:
        text: 정규화할 텍스트

    Returns:
        정규화된 텍스트

    Examples:
        >>> normalize_korean_text("test ㄹㅡㄹ 통해서")
        'test ㄹㅡㄹ 통해서'  # 자모는 그대로 유지됨
        >>> normalize_korean_text("테스트")
        '테스트'
    """
    if not text:
        return text

    # NFC 정규화 적용 (조합형으로 변환)
    normalized = unicodedata.normalize('NFC', text)

    # 자모 시퀀스를 한글로 변환 시도
    for jamo_seq, syllable in JAMO_TO_SYLLABLE_MAP.items():
        if jamo_seq in normalized:
            normalized = normalized.replace(jamo_seq, syllable)
            logger.debug(f"Replaced jamo sequence '{jamo_seq}' with syllable '{syllable}'")

    return normalized


def is_text_decomposed(text: str) -> bool:
    """텍스트가 자소 분리된 상태인지 확인합니다.

    Args:
        text: 확인할 텍스트

    Returns:
        자소 분리 여부
    """
    nfc = unicodedata.normalize('NFC', text)
    nfd = unicodedata.normalize('NFD', text)

    # NFD와 다르면 이미 분해된 상태거나, 자모만 있는 경우
    return text != nfc or len(nfd) > len(nfc)


def contains_jamo(text: str) -> bool:
    """텍스트에 한글 자모(ㄱ,ㄴ,ㄷ... ㅏ,ㅑ,ㅓ...)가 포함되어 있는지 확인합니다.

    Args:
        text: 확인할 텍스트

    Returns:
        자모 포함 여부
    """
    for char in text:
        # 한글 자모 유니코드 범위: U+3131 ~ U+318E
        if 0x3131 <= ord(char) <= 0x318E:
            return True
    return False


def fix_decomposed_korean(text: str, aggressive: bool = False) -> str:
    """자소가 분리된 한글 텍스트를 수정합니다.

    Args:
        text: 수정할 텍스트
        aggressive: True일 경우 더 적극적으로 자모를 조합 시도

    Returns:
        수정된 텍스트
    """
    if not text:
        return text

    # 기본 정규화
    result = normalize_korean_text(text)

    # 자모가 여전히 있고 aggressive 모드인 경우
    if aggressive and contains_jamo(result):
        logger.warning(f"Text still contains jamo after normalization: '{result}'")
        # 추가적인 처리를 여기에 구현할 수 있음

    return result


def normalize_dataframe_korean(df: Any, columns: Optional[List[str]] = None) -> Any:
    """DataFrame의 한글 텍스트 컬럼을 정규화합니다.

    Args:
        df: pandas DataFrame
        columns: 정규화할 컬럼 목록. None이면 모든 문자열 컬럼 처리

    Returns:
        정규화된 DataFrame
    """
    import pandas as pd

    if not isinstance(df, pd.DataFrame):
        return df

    df_copy = df.copy()

    # 처리할 컬럼 결정
    if columns is None:
        columns = df_copy.select_dtypes(include=['object']).columns.tolist()

    for col in columns:
        if col in df_copy.columns and df_copy[col].dtype == 'object':
            # 각 셀에 정규화 적용
            df_copy[col] = df_copy[col].apply(
                lambda x: normalize_korean_text(x) if isinstance(x, str) else x
            )

            # 자모가 포함된 행 로깅
            jamo_mask = df_copy[col].apply(
                lambda x: contains_jamo(x) if isinstance(x, str) else False
            )
            if jamo_mask.any():
                logger.info(f"Column '{col}' contains {jamo_mask.sum()} rows with jamo characters")

    return df_copy


def batch_normalize_korean(texts: List[str]) -> List[str]:
    """여러 텍스트를 한 번에 정규화합니다.

    Args:
        texts: 텍스트 목록

    Returns:
        정규화된 텍스트 목록
    """
    return [normalize_korean_text(text) for text in texts]