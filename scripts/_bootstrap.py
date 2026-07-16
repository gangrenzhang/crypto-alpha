"""让脚本无需安装即可导入 src 下的包。放在每个脚本最前面 import。"""
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))
