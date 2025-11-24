#!/usr/bin/env python3
"""
IQS Streamlit 앱 실행 스크립트
"""
import subprocess
import sys
import os
from pathlib import Path

def main():
    """IQS Streamlit 앱 실행"""
    
    # 현재 디렉토리 확인
    current_dir = Path.cwd()
    app_path = current_dir / "streamlit" / "iqs_app.py"
    
    if not app_path.exists():
        print(f"❌ 앱 파일을 찾을 수 없습니다: {app_path}")
        print("프로젝트 루트 디렉토리에서 실행하세요.")
        return
    
    print("🚀 IQS Quality Data Analytics 시작...")
    print(f"📁 앱 경로: {app_path}")
    print("🌐 브라우저에서 http://localhost:8501 로 접속하세요")
    print("🛑 종료하려면 Ctrl+C를 누르세요")
    print("-" * 50)
    
    try:
        # Streamlit 앱 실행
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            str(app_path),
            "--server.port=8501",
            "--server.address=localhost",
            "--server.headless=false"
        ], check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ 앱 실행 실패: {e}")
    except KeyboardInterrupt:
        print("\n🛑 앱이 종료되었습니다.")

if __name__ == "__main__":
    main()