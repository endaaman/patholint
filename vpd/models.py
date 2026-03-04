import re
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel


class Report(BaseModel):
    フリガナ: Optional[str] = None
    依頼元コード: Optional[float] = None
    依頼医師名: Optional[str] = None
    受付日: Optional[str] = None
    年齢: Optional[float] = None
    性別: Optional[str] = None
    採取方法: Optional[str] = None
    採取日: Optional[str] = None
    施設名: Optional[str] = None
    氏名: Optional[str] = None
    生年月日: Optional[str] = None
    病理番号: str
    臨床診断: Optional[str] = None
    checker: Optional[str] = None
    診断医師: Optional[str] = None
    診断年月日: Optional[str] = None
    病理組織所見: str = ""
    病理組織診断: str = ""

    @staticmethod
    def load(path: str | Path) -> "Report":
        text = Path(path).read_text()
        # frontmatter解析
        parts = text.split("---", 2)
        if len(parts) < 3:
            raise ValueError(f"Invalid frontmatter in {path}")
        meta = yaml.safe_load(parts[1]) or {}
        body = parts[2].strip()

        # <所見>...</所見> と <診断>...</診断> をパース
        for tag, key in [("所見", "病理組織所見"), ("診断", "病理組織診断")]:
            m = re.search(rf"<{tag}>\n?(.*?)\n?</{tag}>", body, re.DOTALL)
            meta[key] = m.group(1).strip() if m else ""

        return Report(**meta)

    @staticmethod
    def load_dir(dirpath: str | Path) -> list["Report"]:
        dirpath = Path(dirpath)
        reports = []
        for p in sorted(dirpath.glob("*.md")):
            reports.append(Report.load(p))
        return reports
