# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 0.1.x   | :white_check_mark: |

## Reporting a Vulnerability

If you discover a security vulnerability in FE Engine, please report it by:

1. **Do not** open a public issue
2. Email the maintainer directly at: [Contact via GitHub @krank56]
3. Include:
   - Description of the vulnerability
   - Steps to reproduce
   - Potential impact
   - Suggested fix (if any)

## Response Timeline

- Initial response: Within 48 hours
- Status update: Within 7 days
- Fix timeline: Depends on severity
  - Critical: Within 7 days
  - High: Within 14 days
  - Medium: Within 30 days
  - Low: Next release cycle

## Security Best Practices

When using FE Engine:

1. **Input Validation**: Always validate structural model inputs
2. **Dependencies**: Keep dependencies up to date
3. **Audit Trail**: Enable audit logging for production use
4. **Data Security**: Do not include sensitive data in model names or comments
5. **Resource Limits**: Set appropriate limits for model size to prevent resource exhaustion

## Known Security Considerations

- FE Engine performs numerical computations and does not handle untrusted external input by default
- GPU features are experimental and should be tested thoroughly before production use
- Large models may consume significant memory resources

## Updates

Security updates will be released as patch versions and documented in CHANGELOG.md with a `[SECURITY]` tag.
