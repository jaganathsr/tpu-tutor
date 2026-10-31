"""Check every URL in research/sources.md and write research/url_check.txt.

Usage: python3 scripts/check_urls.py
"""
import concurrent.futures as cf
import datetime
import re
import urllib.request

SRC = "research/sources.md"
OUT = "research/url_check.txt"
UA = "Mozilla/5.0 (X11; Linux x86_64) tpu-course-link-check"


def check(url):
    for method in ("HEAD", "GET"):
        try:
            req = urllib.request.Request(url, method=method, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=25) as r:
                return url, r.status, r.geturl()
        except urllib.error.HTTPError as e:
            if method == "GET":
                return url, e.code, ""
        except Exception as e:  # network errors
            if method == "GET":
                return url, f"ERR {type(e).__name__}", ""
    return url, "?", ""


def main():
    text = open(SRC, encoding="utf-8").read()
    urls = sorted(set(u.rstrip(").,|") for u in re.findall(r"https?://[^\s)`|]+", text)))
    with cf.ThreadPoolExecutor(16) as ex:
        results = list(ex.map(check, urls))
    lines = [f"# URL check {datetime.date.today()} ({len(urls)} URLs)"]
    for url, status, final in sorted(results, key=lambda r: str(r[1])):
        moved = f" -> {final}" if final and final.rstrip("/") != url.rstrip("/") else ""
        lines.append(f"{status}\t{url}{moved}")
    open(OUT, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("\n".join(l for l in lines if not l.startswith("200")))


if __name__ == "__main__":
    main()
