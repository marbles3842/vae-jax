#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import urllib.request
import urllib.parse
from urllib.error import HTTPError, URLError

def load_env():
    env_path = os.path.expanduser("~/.env")
    if not os.path.exists(env_path):
        return
    with open(env_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("=", 1)
            if len(parts) == 2:
                key, val = parts
                val = val.strip().strip("'\"")
                if key not in os.environ:
                    os.environ[key] = val

def run_git(args):
    try:
        res = subprocess.run(args, capture_output=True, text=True, check=True)
        return res.stdout.strip()
    except subprocess.CalledProcessError as e:
        print(f"Git command failed: {' '.join(args)}", file=sys.stderr)
        print(f"Error output: {e.stderr}", file=sys.stderr)
        sys.exit(e.returncode)

def resolve_ssh_host(host):
    config_path = os.path.expanduser("~/.ssh/config")
    if not os.path.exists(config_path):
        return host
    
    current_host = None
    with open(config_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            host_match = re.match(r"^Host\s+(.+)$", line, re.IGNORECASE)
            if host_match:
                hosts = host_match.group(1).split()
                if host in hosts:
                    current_host = host
                else:
                    current_host = None
            elif current_host and re.match(r"^HostName\s+", line, re.IGNORECASE):
                hostname_match = re.match(r"^HostName\s+(.+)$", line, re.IGNORECASE)
                if hostname_match:
                    return hostname_match.group(1).strip()
    return host

def parse_git_remote(url):
    url_clean = url.strip()
    if url_clean.endswith(".git"):
        url_clean = url_clean[:-4]
    
    if url_clean.startswith("git@"):
        parts = url_clean[4:].split(":", 1)
        if len(parts) == 2:
            host, path = parts
        else:
            parts = url_clean[4:].split("/", 1)
            host = parts[0]
            path = parts[1] if len(parts) > 1 else ""
    elif url_clean.startswith("ssh://git@"):
        parts = url_clean[10:].split("/", 1)
        host = parts[0]
        path = parts[1] if len(parts) > 1 else ""
    elif url_clean.startswith("https://"):
        parts = url_clean[8:].split("/", 1)
        host = parts[0]
        path = parts[1] if len(parts) > 1 else ""
    elif url_clean.startswith("http://"):
        parts = url_clean[7:].split("/", 1)
        host = parts[0]
        path = parts[1] if len(parts) > 1 else ""
    else:
        host = ""
        path = url_clean
        
    path_parts = path.strip("/").split("/", 1)
    if len(path_parts) == 2:
        owner, repo = path_parts
    else:
        owner = ""
        repo = path_parts[0]
        
    return host, owner, repo

def make_request(url, headers=None, data=None, method="GET"):
    req_headers = {
        "User-Agent": "Antigravity-Agent"
    }
    if headers:
        req_headers.update(headers)
    
    req_data = None
    if data is not None:
        req_data = json.dumps(data).encode("utf-8")
        req_headers["Content-Type"] = "application/json"
    
    req = urllib.request.Request(url, headers=req_headers, data=req_data, method=method)
    try:
        with urllib.request.urlopen(req) as res:
            return json.loads(res.read().decode("utf-8"))
    except HTTPError as e:
        err_msg = e.read().decode("utf-8")
        try:
            err_json = json.loads(err_msg)
            print(f"API Error ({e.code}): {json.dumps(err_json, indent=2)}", file=sys.stderr)
        except Exception:
            print(f"API Error ({e.code}): {err_msg}", file=sys.stderr)
        sys.exit(1)
    except URLError as e:
        print(f"Failed to connect to API: {e.reason}", file=sys.stderr)
        sys.exit(1)

def main():
    load_env()
    
    parser = argparse.ArgumentParser(description="Create a PR/MR on GitHub/GitLab.")
    parser.add_argument("--title", help="PR/MR Title")
    parser.add_argument("--description", help="PR/MR Description")
    parser.add_argument("--base", help="Target/Base Branch")
    parser.add_argument("--draft", action="store_true", help="Create as a Draft (GitHub only)")
    parser.add_argument("--push", action="store_true", help="Push branch to remote before creating")
    parser.add_argument("--dry-run", action="store_true", help="Print details without making the API call")
    args = parser.parse_args()
    
    # 1. Get current branch
    head_branch = run_git(["git", "branch", "--show-current"])
    if not head_branch:
        # Fallback if in detached head
        head_branch = run_git(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    
    if head_branch in ("main", "master"):
        print(f"Warning: You are currently on the default branch '{head_branch}'. Creating a PR/MR from '{head_branch}' is usually not desired.", file=sys.stderr)
        
    # 2. Get remote origin URL
    remote_url = run_git(["git", "config", "--get", "remote.origin.url"])
    if not remote_url:
        print("Error: No remote origin URL found. Run 'git remote add origin <url>' first.", file=sys.stderr)
        sys.exit(1)
        
    host, owner, repo = parse_git_remote(remote_url)
    resolved_host = resolve_ssh_host(host)
    
    # Determine platform
    is_github = "github" in resolved_host.lower()
    is_gitlab = "gitlab" in resolved_host.lower()
    
    if not is_github and not is_gitlab:
        print(f"Warning: Unrecognized host '{resolved_host}'. Defaulting to GitHub API.", file=sys.stderr)
        is_github = True
        
    # Get credentials
    token = None
    if is_github:
        token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
        if not token:
            print("Error: GITHUB_TOKEN or GH_TOKEN is not set. Please set it in your environment or ~/.env", file=sys.stderr)
            sys.exit(1)
    else:
        token = os.environ.get("GITLAB_TOKEN") or os.environ.get("GL_TOKEN")
        if not token:
            print("Error: GITLAB_TOKEN is not set. Please set it in your environment or ~/.env", file=sys.stderr)
            sys.exit(1)
            
    # 3. Check if remote tracking branch exists
    remote_branch_exists = False
    try:
        subprocess.run(["git", "rev-parse", "--verify", f"origin/{head_branch}"], capture_output=True, check=True)
        remote_branch_exists = True
    except subprocess.CalledProcessError:
        pass
        
    if not remote_branch_exists:
        if args.push:
            print(f"Pushing current branch '{head_branch}' to remote origin...")
            run_git(["git", "push", "--set-upstream", "origin", head_branch])
        elif not args.dry_run:
            print(f"Error: Branch '{head_branch}' does not exist on remote. Push it first or run with --push.", file=sys.stderr)
            sys.exit(1)
            
    # 4. Fetch repo default branch
    headers = {}
    default_branch = args.base
    
    if is_github:
        headers["Authorization"] = f"Bearer {token}"
        headers["Accept"] = "application/vnd.github+json"
        
        if not default_branch:
            # Call GitHub API to get default branch
            api_url = f"https://api.github.com/repos/{owner}/{repo}"
            repo_info = make_request(api_url, headers=headers)
            default_branch = repo_info.get("default_branch", "main")
    else:
        headers["PRIVATE-TOKEN"] = token
        
        if not default_branch:
            # Call GitLab API to get default branch
            project_path_encoded = urllib.parse.quote(f"{owner}/{repo}", safe="")
            api_url = f"https://gitlab.com/api/v4/projects/{project_path_encoded}"
            project_info = make_request(api_url, headers=headers)
            default_branch = project_info.get("default_branch", "main")
            
    # 5. Generate default title & description
    title = args.title
    if not title:
        # Use latest commit message
        title = run_git(["git", "log", "-1", "--pretty=%B"]).split("\n")[0]
        
    description = args.description
    if not description:
        # Check if we can get commits diff
        try:
            commits = run_git(["git", "log", f"origin/{default_branch}..HEAD", "--oneline"])
            if commits:
                description = "### Commits on this branch:\n" + "\n".join(f"- {c}" for c in commits.split("\n"))
            else:
                description = "No new commits detected relative to " + default_branch
        except Exception:
            description = "Created automatically via git-create-mr skill"
            
    # 6. Execute or dry-run
    if args.dry_run:
        print("=== DRY RUN (No PR/MR will be created) ===")
        print(f"Platform:       {'GitHub' if is_github else 'GitLab'}")
        print(f"Repository:     {owner}/{repo}")
        print(f"Source Branch:  {head_branch}")
        print(f"Base Branch:    {default_branch}")
        print(f"Title:          {title}")
        print(f"Description:\n{description}")
        return
        
    print(f"Creating {'Pull Request' if is_github else 'Merge Request'} targeting '{default_branch}'...")
    
    if is_github:
        api_url = f"https://api.github.com/repos/{owner}/{repo}/pulls"
        data = {
            "title": title,
            "body": description,
            "head": head_branch,
            "base": default_branch,
            "draft": args.draft
        }
        res = make_request(api_url, headers=headers, data=data, method="POST")
        pr_url = res.get("html_url")
        print(f"Successfully created Pull Request: {pr_url}")
    else:
        project_path_encoded = urllib.parse.quote(f"{owner}/{repo}", safe="")
        api_url = f"https://gitlab.com/api/v4/projects/{project_path_encoded}/merge_requests"
        data = {
            "source_branch": head_branch,
            "target_branch": default_branch,
            "title": title,
            "description": description
        }
        res = make_request(api_url, headers=headers, data=data, method="POST")
        mr_url = res.get("web_url")
        print(f"Successfully created Merge Request: {mr_url}")

if __name__ == "__main__":
    main()
