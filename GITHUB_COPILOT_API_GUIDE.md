# GitHub Copilot API Guide

This guide explains how to get access to and use the GitHub Copilot API.

> **⚠️ Important Note:** Direct programmatic access to GitHub Copilot's code completion API may be limited or restricted. This guide covers available GitHub APIs related to Copilot subscription management and general AI model access through GitHub Models. Always refer to [GitHub's official documentation](https://docs.github.com/en/copilot) for the most current API capabilities and access methods.

## Table of Contents

- [What is GitHub Copilot API?](#what-is-github-copilot-api)
- [Prerequisites](#prerequisites)
- [Getting API Access](#getting-api-access)
- [Authentication](#authentication)
- [API Endpoints](#api-endpoints)
- [Usage Examples](#usage-examples)
- [Rate Limits and Pricing](#rate-limits-and-pricing)
- [Best Practices](#best-practices)
- [Resources](#resources)

---

## What is GitHub Copilot API?

GitHub Copilot is an AI-powered code completion tool that uses OpenAI's Codex model. While GitHub Copilot is primarily used through IDE extensions, GitHub provides APIs for:

1. **Subscription Management**: Check Copilot access and manage seats
2. **GitHub Models**: Access to AI models through GitHub's marketplace
3. **Integration Points**: Webhook and app integration capabilities

**Key Features:**
- Code completion and suggestions (primarily through IDE extensions)
- Code generation from natural language descriptions
- Context-aware programming assistance
- Support for multiple programming languages

**API Availability:**
- ✅ Copilot subscription and seat management APIs are available
- ✅ GitHub Models provides access to various AI models
- ⚠️ Direct code completion API access may be limited to IDE extensions

---

## Prerequisites

Before you can access the GitHub Copilot API, you need:

1. **GitHub Account**: An active GitHub account
2. **GitHub Copilot Subscription**: Either:
   - GitHub Copilot Individual subscription (~$10/month)
   - GitHub Copilot Business subscription (~$19/user/month)
   - GitHub Copilot Enterprise subscription (part of GitHub Enterprise Cloud)
3. **Organization Admin Access** (for Business/Enterprise): You need to be an organization owner or admin

---

## Getting API Access

### Step 1: Subscribe to GitHub Copilot

**For Individual Users:**
1. Go to [GitHub Copilot](https://github.com/features/copilot)
2. Click "Start free trial" or "Buy now"
3. Follow the subscription process
4. Complete payment setup

**For Organizations:**
1. Navigate to your organization settings
2. Go to "Copilot" under "Code, planning, and automation"
3. Enable GitHub Copilot for your organization
4. Assign seats to users

### Step 2: Access the API

GitHub Copilot API access is available through:

1. **GitHub Models** (Recommended):
   - Visit [GitHub Models](https://github.com/marketplace/models)
   - Browse available models including Copilot models
   - Generate API keys from your GitHub settings

2. **Personal Access Token (Classic Method)**:
   - Go to GitHub Settings → Developer settings → Personal access tokens
   - Generate new token (classic)
   - Select required scopes:
     - `read:user`
     - `user:email`
     - `copilot` (if available)
   - Copy and securely store your token

3. **GitHub Apps** (For Production):
   - Create a GitHub App in your organization settings
   - Configure permissions for Copilot access
   - Install the app in your organization
   - Use the app's credentials for API access

### Step 3: Verify Access

Test your access with a simple API call:

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     https://api.github.com/user/copilot_seat_details
```

---

## Authentication

GitHub Copilot API uses token-based authentication:

### Personal Access Token

```bash
curl -H "Authorization: Bearer ghp_xxxxxxxxxxxx" \
     -H "Accept: application/vnd.github+json" \
     https://api.github.com/copilot/...
```

### GitHub App Authentication

```python
import time
import requests
from jwt import encode

# Generate JWT for GitHub App
def generate_jwt(app_id, private_key):
    payload = {
        'iat': int(time.time()),
        'exp': int(time.time()) + 600,  # 10 minutes
        'iss': app_id
    }
    return encode(payload, private_key, algorithm='RS256')

# Use JWT to get installation token
headers = {
    'Authorization': f'Bearer {jwt_token}',
    'Accept': 'application/vnd.github+json'
}
```

---

## API Endpoints

### 1. Check Copilot Seat Status

**Endpoint:** `GET /user/copilot_seat_details`

**Description:** Check if the authenticated user has access to Copilot

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Accept: application/vnd.github+json" \
     https://api.github.com/user/copilot_seat_details
```

**Response:**
```json
{
  "seat": {
    "created_at": "2024-01-01T00:00:00Z",
    "last_activity_at": "2024-12-01T00:00:00Z",
    "organization": {
      "login": "your-org",
      "id": 12345
    }
  }
}
```

### 2. Code Completion (via GitHub Models)

**Endpoint:** Available through GitHub Models marketplace

**Example using Python:**

```python
import requests
import json

def get_code_completion(prompt, token):
    """
    Get code completion from GitHub Copilot via GitHub Models
    
    NOTE: This is a conceptual example. The actual endpoint URL and
    request format should be obtained from GitHub's official documentation
    as the API is continuously evolving.
    
    See: https://github.com/marketplace/models for current endpoints
    """
    
    # PLACEHOLDER: Replace with actual endpoint from GitHub documentation
    url = "https://api.github.com/models/completions"  # Verify current endpoint
    
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        "Accept": "application/json",
        "X-GitHub-Api-Version": "2022-11-28"
    }
    
    data = {
        "model": "copilot",  # Verify available model names
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.5
    }
    
    response = requests.post(url, headers=headers, json=data)
    return response.json()
```

### 3. Organization Copilot Settings

**Endpoint:** `GET /orgs/{org}/copilot/billing`

**Description:** Get Copilot billing information for an organization

```bash
curl -H "Authorization: Bearer YOUR_TOKEN" \
     -H "Accept: application/vnd.github+json" \
     https://api.github.com/orgs/YOUR_ORG/copilot/billing
```

---

## Usage Examples

### Example 1: Simple Code Completion in Python

```python
import os
import requests

class GitHubCopilotClient:
    def __init__(self, token):
        self.token = token
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
    
    def check_access(self):
        """Check if user has Copilot access"""
        response = requests.get(
            f"{self.base_url}/user/copilot_seat_details",
            headers=self.headers
        )
        return response.json()
    
    def get_completion(self, code_context, language="python"):
        """
        Get code completion suggestion
        
        NOTE: This is a placeholder method. The actual GitHub Copilot API
        for code completions may not be publicly available or may require
        different authentication/endpoints.
        
        For production use:
        1. Check GitHub's official documentation for available endpoints
        2. Consider using GitHub Copilot through its official IDE extensions
        3. Explore GitHub Models marketplace for available AI models
        
        Returns:
            NotImplementedError: This method requires implementation based on
                                current GitHub API documentation
        """
        raise NotImplementedError(
            "Code completion endpoint not yet available. "
            "Please check GitHub documentation for current API capabilities: "
            "https://docs.github.com/en/copilot"
        )

# Usage example (check_access only)
token = os.environ.get("GITHUB_TOKEN")
client = GitHubCopilotClient(token)

# Check if you have Copilot access
access_info = client.check_access()
print(f"Copilot access: {access_info}")
```

### Example 2: Using with Environment Variables

```python
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

GITHUB_TOKEN = os.getenv("GITHUB_COPILOT_TOKEN")

# Use token for API calls
headers = {
    "Authorization": f"Bearer {GITHUB_TOKEN}",
    "Accept": "application/vnd.github+json"
}
```

### Example 3: Integration with This Health Agent Project

To integrate GitHub Copilot API into this Health Agent project:

```python
# Add to config/settings.py
GITHUB_COPILOT_TOKEN = os.getenv("GITHUB_COPILOT_TOKEN", "")
GITHUB_COPILOT_ENABLED = bool(GITHUB_COPILOT_TOKEN)

# Add to .streamlit/secrets.toml.example
GITHUB_COPILOT_TOKEN = "your_github_copilot_token_here"
```

---

## Rate Limits and Pricing

### GitHub Copilot Subscription Pricing

- **Individual**: $10/month or $100/year
- **Business**: $19/user/month
- **Enterprise**: Included with GitHub Enterprise Cloud
- **Free for**: Verified students, teachers, and maintainers of popular open-source projects

### API Rate Limits

GitHub API rate limits apply:
- **Authenticated requests**: 5,000 requests per hour
- **Unauthenticated requests**: 60 requests per hour

### Copilot-Specific Limits

- Code completion requests may have additional throttling
- Large organizations may have higher limits
- Check your organization's quota in settings

---

## Best Practices

### Security

1. **Never commit tokens to version control**
   ```bash
   # Add to .gitignore
   .streamlit/secrets.toml
   .env
   *.key
   ```

2. **Use environment variables**
   ```python
   import os
   token = os.environ.get("GITHUB_COPILOT_TOKEN")
   ```

3. **Rotate tokens regularly**
   - Regenerate tokens every 90 days
   - Immediately revoke compromised tokens

4. **Use minimal scopes**
   - Only request necessary permissions
   - Prefer fine-grained tokens over classic tokens

### Performance

1. **Cache responses** when appropriate
2. **Implement retry logic** with exponential backoff
3. **Monitor rate limits** and adjust usage accordingly
4. **Use batch requests** when possible

### Code Quality

1. **Review AI-generated code** before using in production
2. **Add proper error handling** for API calls
3. **Log API usage** for debugging and monitoring
4. **Test thoroughly** before deployment

---

## Resources

### Official Documentation

- [GitHub Copilot Documentation](https://docs.github.com/en/copilot)
- [GitHub REST API Documentation](https://docs.github.com/en/rest)
- [GitHub Copilot for Business](https://docs.github.com/en/copilot/copilot-business)
- [GitHub Models Marketplace](https://github.com/marketplace/models)

### Getting Started

- [GitHub Copilot Quick Start](https://docs.github.com/en/copilot/quickstart)
- [Setting up GitHub Copilot](https://docs.github.com/en/copilot/setting-up-github-copilot)
- [GitHub Copilot in the CLI](https://docs.github.com/en/copilot/github-copilot-in-the-cli)

### API References

- [GitHub REST API Reference](https://docs.github.com/en/rest)
- [GitHub GraphQL API](https://docs.github.com/en/graphql)
- [Authentication Guide](https://docs.github.com/en/authentication)

### Community & Support

- [GitHub Community Forum](https://github.community/)
- [GitHub Support](https://support.github.com/)
- [Stack Overflow - GitHub Copilot](https://stackoverflow.com/questions/tagged/github-copilot)

### Alternative AI Code Assistants

If GitHub Copilot doesn't meet your needs, consider:

- **Amazon CodeWhisperer**: Free tier available, AWS integration
- **Tabnine**: Privacy-focused, self-hosted options
- **Codeium**: Free for individuals, similar features
- **OpenAI Codex API**: Direct access to underlying model

---

## Troubleshooting

### Common Issues

**Issue**: "Resource not accessible by personal access token"
- **Solution**: Ensure your token has the correct scopes. For Copilot, you may need `copilot` scope if available.

**Issue**: "Copilot subscription required"
- **Solution**: Verify you have an active Copilot subscription at https://github.com/settings/copilot

**Issue**: Rate limit exceeded
- **Solution**: Implement rate limiting in your code, or upgrade to GitHub Enterprise for higher limits.

**Issue**: Authentication failed
- **Solution**: Check that your token hasn't expired and is correctly formatted in the Authorization header.

---

## Example: Complete Integration

Here's a complete example of integrating GitHub Copilot API checking into this project:

```python
# File: utils/github_copilot.py

import os
import requests
import streamlit as st
from typing import Optional, Dict

class GitHubCopilotHelper:
    """Helper class for GitHub Copilot API interactions"""
    
    def __init__(self, token: Optional[str] = None):
        self.token = token or os.environ.get("GITHUB_COPILOT_TOKEN")
        self.base_url = "https://api.github.com"
        
    def get_headers(self) -> Dict[str, str]:
        """Get API request headers"""
        return {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28"
        }
    
    def check_copilot_access(self) -> bool:
        """Check if user has GitHub Copilot access"""
        if not self.token:
            return False
            
        try:
            response = requests.get(
                f"{self.base_url}/user/copilot_seat_details",
                headers=self.get_headers(),
                timeout=10
            )
            return response.status_code == 200
        except Exception as e:
            st.warning(f"Could not verify Copilot access: {e}")
            return False
    
    def get_copilot_info(self) -> Optional[Dict]:
        """Get Copilot subscription information"""
        if not self.token:
            return None
            
        try:
            response = requests.get(
                f"{self.base_url}/user/copilot_seat_details",
                headers=self.get_headers(),
                timeout=10
            )
            
            if response.status_code == 200:
                return response.json()
            return None
        except Exception as e:
            st.error(f"Error fetching Copilot info: {e}")
            return None

# Usage in Streamlit app
def display_copilot_status():
    """Display GitHub Copilot status in sidebar"""
    if "GITHUB_COPILOT_TOKEN" in st.secrets:
        helper = GitHubCopilotHelper(st.secrets["GITHUB_COPILOT_TOKEN"])
        
        with st.sidebar:
            st.markdown("### GitHub Copilot")
            
            if helper.check_copilot_access():
                st.success("✅ Copilot Active")
                info = helper.get_copilot_info()
                if info and "seat" in info:
                    st.caption(f"Org: {info['seat'].get('organization', {}).get('login', 'N/A')}")
            else:
                st.warning("⚠️ No Copilot Access")
                st.caption("See GITHUB_COPILOT_API_GUIDE.md")
```

---

## Next Steps

1. **Get a subscription**: Visit [GitHub Copilot](https://github.com/features/copilot)
2. **Generate a token**: Go to GitHub Settings → Developer settings
3. **Test the API**: Use the examples in this guide
4. **Integrate**: Add Copilot features to your projects
5. **Stay updated**: Follow [GitHub Blog](https://github.blog/) for new features

---

## Contributing

Found an error or have improvements? Please contribute:

1. Fork the repository
2. Update this guide
3. Submit a pull request

---

**Last Updated:** February 2026

**Note:** GitHub Copilot API endpoints and features are continuously evolving. Always refer to the [official GitHub documentation](https://docs.github.com/en/copilot) for the most current information.
