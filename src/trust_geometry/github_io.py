"""Push files to a GitHub branch via the REST API (no git binary needed on the pod)."""
import os, base64, json, urllib.request, urllib.error

REPO = os.environ.get("TG_REPO", "ratnaditya-j/trust-geometry")
API = f"https://api.github.com/repos/{REPO}"


def _tok():
    return os.environ.get("GH_TOKEN") or os.environ["GITHUB_TOKEN"]


def _req(method, url, body=None):
    data = json.dumps(body).encode() if body is not None else None
    r = urllib.request.Request(url, data=data, method=method, headers={
        "Authorization": f"Bearer {_tok()}", "Accept": "application/vnd.github+json",
        "User-Agent": "tg-pod", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(r, timeout=40) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def ensure_branch(branch, base="main"):
    s, d = _req("GET", f"{API}/git/ref/heads/{base}")
    if s != 200:
        return False
    sha = d["object"]["sha"]
    s2, _ = _req("POST", f"{API}/git/refs", {"ref": f"refs/heads/{branch}", "sha": sha})
    return s2 in (200, 201, 422)  # 422 = already exists


def put_file(path, content_bytes, message, branch="r0-run"):
    # need current sha if file exists on branch
    s, d = _req("GET", f"{API}/contents/{path}?ref={branch}")
    sha = d.get("sha") if s == 200 else None
    body = {"message": message, "branch": branch,
            "content": base64.b64encode(content_bytes).decode()}
    if sha:
        body["sha"] = sha
    s2, d2 = _req("PUT", f"{API}/contents/{path}", body)
    return s2 in (200, 201), s2


def put_text(path, text, message, branch="r0-run"):
    return put_file(path, text.encode(), message, branch)
