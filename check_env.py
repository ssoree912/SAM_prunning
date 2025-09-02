import sys
import os
import importlib.metadata

print("--- Python 실행 환경 진단 시작 ---")

# 1. 현재 실행되고 있는 파이썬의 위치
print(f"\n[파이썬 실행파일 경로]\n{sys.executable}\n")

# 2. 파이썬이 패키지를 찾기 위해 참조하는 모든 경로
print("[파이썬 경로 (sys.path)]")
for path in sys.path:
    print(path)

# 3. 주요 라이브러리의 실제 위치와 버전 확인
print("\n--- 라이브러리 위치 및 버전 ---")
libs = ['torch', 'transformers', 'regex']
for lib in libs:
    try:
        version = importlib.metadata.version(lib)
        # 패키지의 최상위 모듈을 찾아 파일 위치를 확인합니다.
        spec = importlib.util.find_spec(lib)
        path = spec.origin if spec else "경로를 찾을 수 없음"
        print(f"\n✅ {lib}")
        print(f"  - 버전: {version}")
        print(f"  - 위치: {path}")
    except importlib.metadata.PackageNotFoundError:
        print(f"\n❌ {lib}: 패키지를 찾을 수 없습니다.")
    except Exception as e:
        print(f"\n❌ {lib}: 정보를 가져오는 중 오류 발생 - {e}")

print("\n--- 진단 완료 ---")