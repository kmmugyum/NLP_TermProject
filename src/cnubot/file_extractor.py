"""첨부파일(PDF/HWP) → 텍스트 추출. 경량(fitz/olefile/표준 zlib), 무거운 엔진 없음.

PDF: PyMuPDF(fitz).
HWP 5.0: olefile로 OLE 열고 BodyText/SectionN(zlib raw-deflate) 압축해제 후
         레코드 파싱 → HWPTAG_PARA_TEXT(67) UTF-16LE 본문 추출.
         (제안된 PrvText는 '미리보기 요약'이라 본문이 아님 → 사용 안 함.)
"""
from __future__ import annotations

import io
import struct
import zlib

_HWPTAG_PARA_TEXT = 67  # HWPTAG_BEGIN(0x10) + 51


def extract_pdf(file_bytes: bytes) -> str:
    import fitz

    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            return "\n".join(p.get_text() for p in doc)
    except Exception as e:
        return f"[PDF_ERROR:{type(e).__name__}]"


def extract_hwp(file_bytes: bytes) -> str:
    import olefile

    try:
        f = io.BytesIO(file_bytes)
        if not olefile.isOleFile(f):
            return "[HWP_ERROR:not_ole]"
        ole = olefile.OleFileIO(f)
        header = ole.openstream("FileHeader").read()
        compressed = bool(header[36] & 0x01)  # FileHeader flags bit0 = 압축여부
        secs = sorted(
            (d for d in ole.listdir() if len(d) == 2 and d[0] == "BodyText"),
            key=lambda x: int("".join(filter(str.isdigit, x[1])) or 0),
        )
        out: list[str] = []
        for sec in secs:
            data = ole.openstream(sec).read()
            if compressed:
                data = zlib.decompress(data, -15)  # raw deflate
            i, n = 0, len(data)
            while i + 4 <= n:
                hdr = struct.unpack_from("<I", data, i)[0]
                i += 4
                tag = hdr & 0x3FF
                size = (hdr >> 20) & 0xFFF
                if size == 0xFFF:  # 확장 크기
                    size = struct.unpack_from("<I", data, i)[0]
                    i += 4
                if tag == _HWPTAG_PARA_TEXT:
                    txt = data[i:i + size].decode("utf-16le", errors="ignore")
                    txt = "".join(c for c in txt if ord(c) >= 32 or c in "\n\t")
                    if txt.strip():
                        out.append(txt)
                i += size
        return "\n".join(out)
    except Exception as e:
        return f"[HWP_ERROR:{type(e).__name__}]"


def extract(file_name: str, file_bytes: bytes) -> str:
    ext = file_name.rsplit(".", 1)[-1].lower()
    if ext == "pdf":
        return extract_pdf(file_bytes)
    if ext in ("hwp",):
        return extract_hwp(file_bytes)
    return f"[UNSUPPORTED:{ext}]"
