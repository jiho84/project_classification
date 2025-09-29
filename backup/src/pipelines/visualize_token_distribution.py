#!/usr/bin/env python3
"""
토큰 길이 분포 시각화 스크립트
MLflow에서 토큰 분석 결과를 가져와 시각화
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse
from pathlib import Path

def load_token_stats(json_path: str):
    """토큰 통계 JSON 파일 로드"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def plot_token_distribution(token_stats: dict, output_dir: str = "./"):
    """토큰 길이 분포 시각화"""
    # 출력 디렉토리 생성
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 스타일 설정
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Figure 생성
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'토큰 길이 분포 분석 - {token_stats["tokenizer_name"]}', fontsize=16)
    
    # 1. 히스토그램
    if 'distribution' in token_stats:
        ax = axes[0, 0]
        bins = token_stats['distribution']['bins']
        hist = token_stats['distribution']['histogram']
        
        # 막대 그래프로 표시
        ax.bar(bins[:-1], hist, width=np.diff(bins)[0]*0.8, alpha=0.7)
        ax.axvline(token_stats['mean'], color='red', linestyle='--', label=f"평균: {token_stats['mean']:.1f}")
        ax.axvline(token_stats['percentiles']['p95'], color='orange', linestyle='--', label=f"95%: {token_stats['percentiles']['p95']}")
        
        ax.set_xlabel('토큰 길이')
        ax.set_ylabel('샘플 수')
        ax.set_title('토큰 길이 히스토그램')
        ax.legend()
    
    # 2. 누적 분포 함수 (CDF)
    ax = axes[0, 1]
    percentiles = token_stats['percentiles']
    percentile_values = sorted([(int(k[1:]), v) for k, v in percentiles.items()])
    
    x_vals = [0] + [v for _, v in percentile_values] + [token_stats['max']]
    y_vals = [0] + [p/100 for p, _ in percentile_values] + [1.0]
    
    ax.plot(x_vals, y_vals, marker='o', markersize=8, linewidth=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlabel('토큰 길이')
    ax.set_ylabel('누적 확률')
    ax.set_title('누적 분포 함수 (CDF)')
    
    # 주요 백분위수 표시
    for p, v in percentile_values:
        ax.annotate(f'{p}%: {v}', xy=(v, p/100), xytext=(v+10, p/100),
                   arrowprops=dict(arrowstyle='->', alpha=0.5))
    
    # 3. 통계 요약
    ax = axes[1, 0]
    ax.axis('off')
    
    stats_text = f"""
    📊 토큰 길이 통계 요약
    
    토크나이저: {token_stats['tokenizer_name']}
    분석 샘플 수: {token_stats['sample_size']:,}
    
    기본 통계:
    • 평균: {token_stats['mean']:.1f} 토큰
    • 표준편차: {token_stats['std']:.1f}
    • 최소값: {token_stats['min']} 토큰
    • 최대값: {token_stats['max']} 토큰
    
    백분위수:
    • 50% (중앙값): {token_stats['percentiles']['p50']} 토큰
    • 75%: {token_stats['percentiles']['p75']} 토큰
    • 90%: {token_stats['percentiles']['p90']} 토큰
    • 95%: {token_stats['percentiles']['p95']} 토큰
    • 99%: {token_stats['percentiles']['p99']} 토큰
    """
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, 
            fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    # 4. 권장 max_length 계산
    ax = axes[1, 1]
    ax.axis('off')
    
    # 권장 max_length 계산 (95% 커버리지 기준)
    recommended_length = token_stats['percentiles']['p95']
    # 32의 배수로 올림 (효율적인 배치 처리를 위해)
    recommended_length = ((recommended_length + 31) // 32) * 32
    
    coverage_text = f"""
    🎯 권장 max_length 설정
    
    95% 커버리지 기준:
    • 실제 95% 백분위수: {token_stats['percentiles']['p95']} 토큰
    • 권장 max_length: {recommended_length} 토큰
      (32의 배수로 올림)
    
    커버리지 분석:
    • max_length=128: ~{sum(1 for v in hist[:3] for _ in range(v)) / token_stats['sample_size'] * 100:.1f}% 커버
    • max_length=256: ~{sum(1 for v in hist[:6] for _ in range(v)) / token_stats['sample_size'] * 100:.1f}% 커버
    • max_length=512: ~{sum(1 for v in hist[:11] for _ in range(v)) / token_stats['sample_size'] * 100:.1f}% 커버
    
    💡 추천:
    - 메모리 효율: max_length = 256
    - 성능 중심: max_length = {recommended_length}
    - 완전 커버: max_length = 512
    """
    
    ax.text(0.1, 0.9, coverage_text, transform=ax.transAxes,
            fontsize=12, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.3))
    
    # 레이아웃 조정 및 저장
    plt.tight_layout()
    output_file = output_path / "token_length_distribution.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✅ 시각화 저장 완료: {output_file}")
    
    # 추가로 간단한 요약 차트도 생성
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    
    # 박스플롯 스타일의 시각화
    positions = [1]
    box_data = [[
        token_stats['min'],
        token_stats['percentiles']['p25'] if 'p25' in token_stats['percentiles'] else token_stats['percentiles']['p50'],
        token_stats['percentiles']['p50'],
        token_stats['percentiles']['p75'],
        token_stats['max']
    ]]
    
    bp = ax2.boxplot(box_data, positions=positions, widths=0.6, patch_artist=True,
                     showmeans=True, meanline=True)
    
    # 스타일링
    for patch in bp['boxes']:
        patch.set_facecolor('lightblue')
        patch.set_alpha(0.7)
    
    # 백분위수 라벨 추가
    ax2.text(1, token_stats['percentiles']['p95'], f"95%: {token_stats['percentiles']['p95']}", 
            ha='center', va='bottom')
    ax2.text(1, token_stats['percentiles']['p99'], f"99%: {token_stats['percentiles']['p99']}", 
            ha='center', va='bottom')
    
    ax2.set_ylabel('토큰 길이')
    ax2.set_title(f'토큰 길이 분포 요약 - {token_stats["tokenizer_name"]}')
    ax2.set_xticklabels(['토큰 길이 분포'])
    ax2.grid(True, alpha=0.3)
    
    output_file2 = output_path / "token_length_summary.png"
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"✅ 요약 차트 저장 완료: {output_file2}")
    
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='토큰 길이 분포 시각화')
    parser.add_argument('json_path', help='토큰 통계 JSON 파일 경로')
    parser.add_argument('--output-dir', default='./', help='출력 디렉토리 (기본값: 현재 디렉토리)')
    
    args = parser.parse_args()
    
    # 토큰 통계 로드
    token_stats = load_token_stats(args.json_path)
    
    # 시각화 생성
    plot_token_distribution(token_stats, args.output_dir)

if __name__ == "__main__":
    main()