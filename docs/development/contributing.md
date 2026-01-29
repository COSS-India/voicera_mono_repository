# Contributing to VoiceERA

Guidelines for contributing to VoiceERA.

## Code of Conduct

- Be respectful and inclusive
- No harassment or discrimination
- Provide constructive feedback
- Help others learn and improve

## Getting Started

1. **Fork the repository**
   ```bash
   # Go to https://github.com/voicera/voicera_mono_repository
   # Click "Fork"
   ```

2. **Clone your fork**
   ```bash
   git clone https://github.com/your-username/voicera_mono_repository.git
   cd voicera_mono_repository
   ```

3. **Set up development environment**
   ```bash
   # See Local Development Setup
   ```

4. **Create a branch**
   ```bash
   git checkout -b feature/your-feature
   ```

## Development Process

### 1. Before You Start

- Check [Issues](https://github.com/voicera/voicera_mono_repository/issues) for existing work
- Create an issue for new features
- Discuss major changes in issues first
- Follow the [Roadmap](roadmap.md) for planned features

### 2. Code Style

#### Python

```python
# Follow PEP 8
# Use type hints
def calculate_statistics(
    call_logs: list[CallLog],
    filter_date: datetime
) -> dict[str, float]:
    """Calculate statistics from call logs.
    
    Args:
        call_logs: List of call logs to analyze
        filter_date: Date filter for logs
        
    Returns:
        Dictionary with statistics
    """
    total_calls = len(call_logs)
    return {"total": total_calls}

# Use meaningful names
agent_name = "Customer Support Bot"  # Good
x = "Customer Support Bot"  # Bad

# Add docstrings to all functions
```

Format with:
```bash
pip install black isort
black .
isort .
```

#### JavaScript/TypeScript

```typescript
// Use TypeScript for type safety
interface Agent {
  id: string;
  name: string;
  config: AgentConfig;
}

// Use const and let, not var
const agentName = "Support Bot";

// Use arrow functions
const processCall = async (callId: string): Promise<CallResult> => {
  // ...
};

// Add JSDoc comments
/**
 * Get agent details
 * @param agentId - The agent ID
 * @returns Agent details
 */
export const getAgent = async (agentId: string): Promise<Agent> => {
  // ...
};
```

Format with:
```bash
npm run format
npm run lint
```

### 3. Commit Messages

Follow conventional commits:

```
type(scope): subject

body

footer
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

Examples:

```
feat(agent): Add agent cloning functionality
fix(api): Return correct status code for 404 responses
docs(readme): Update installation instructions
test(backend): Add tests for auth service
chore(deps): Update dependencies
```

### 4. Testing

Write tests for all code:

```bash
# Backend
cd voicera_backend
pytest

# Frontend
cd voicera_frontend
npm test
```

#### Test Coverage

- Unit tests for functions/components
- Integration tests for APIs
- E2E tests for critical flows

Example test:

```python
# tests/test_agent_service.py
def test_create_agent():
    """Test agent creation"""
    service = AgentService()
    agent = service.create_agent(
        name="Test Agent",
        llm_provider="openai"
    )
    
    assert agent.name == "Test Agent"
    assert agent.llm_provider == "openai"
```

### 5. Documentation

Update docs for all changes:

```bash
# Create feature docs
touch docs/features/your-feature.md
```

Docs should include:
- What it does
- How to use it
- Code examples
- Configuration options
- Troubleshooting

### 6. Pull Request

#### Create PR

```bash
git push origin feature/your-feature
# Go to https://github.com/voicera/voicera_mono_repository/pull/new/feature/your-feature
```

#### PR Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation

## Changes
- Change 1
- Change 2

## Testing
- [ ] Tested locally
- [ ] Added/updated tests
- [ ] Tests pass

## Checklist
- [ ] Code follows style guide
- [ ] Documentation updated
- [ ] No new warnings
- [ ] Ready for review
```

#### PR Review Guidelines

- All PRs require at least 1 approval
- Tests must pass
- Code coverage should not decrease
- Documentation should be updated

### 7. Code Review

When reviewing:

```
✅ Approve if:
- Code is correct and follows style guide
- Tests are adequate
- Documentation is updated
- No major issues

⚠️ Request changes if:
- Logic issues
- Missing tests
- Poor naming or style
- Security concerns

💬 Comment with suggestions
```

## Architecture Decisions

### ADR (Architecture Decision Records)

Document major decisions:

```markdown
# ADR 001: Use Pipecat for Voice Processing

## Status
ACCEPTED

## Context
Need real-time voice processing for agent calls

## Decision
Use Pipecat framework for voice bot pipeline

## Consequences
- Benefits: Real-time, modular, extensible
- Drawbacks: Learning curve for team
- Mitigations: Documentation, examples, training
```

Save as: `docs/adr/001-use-pipecat.md`

## Code Structure

### Backend Structure

```
voicera_backend/
├── app/
│   ├── main.py              # App initialization
│   ├── config.py            # Configuration
│   ├── models/              # Data models
│   ├── routers/             # API endpoints
│   ├── services/            # Business logic
│   ├── schemas/             # Pydantic schemas
│   └── utils/               # Utilities
├── tests/                   # Test files
├── requirements.txt
└── README.md
```

Guidelines:
- Models: Data structures only
- Schemas: API request/response
- Services: Business logic
- Routers: API endpoints
- Utils: Shared helpers

### Frontend Structure

```
voicera_frontend/
├── app/                     # Next.js app
├── components/              # React components
├── hooks/                   # Custom hooks
├── lib/                     # Utilities
├── public/                  # Static files
├── styles/                  # CSS modules
└── __tests__/              # Tests
```

Guidelines:
- Components: UI components only
- Hooks: Reusable logic
- Lib: API, utilities, helpers
- Use TypeScript for type safety

### Voice Server Structure

```
voice_2_voice_server/
├── api/
│   ├── server.py            # FastAPI server
│   ├── bot.py               # Bot logic
│   └── services.py          # Service integrations
├── config/                  # Configuration
├── services/                # Service providers
│   ├── ai4bharat/
│   ├── bhashini/
│   └── kenpath_llm/
├── storage/                 # Data storage
└── main.py                  # Entry point
```

## Security

### Security Checklist

Before submitting PR:

- [ ] No hardcoded secrets
- [ ] Input validation added
- [ ] SQL injection prevention (if applicable)
- [ ] XSS prevention (for frontend)
- [ ] CSRF tokens used
- [ ] Rate limiting considered
- [ ] Error messages don't leak info
- [ ] Dependencies scanned for vulnerabilities

### Reporting Security Issues

Don't open public issues for security vulnerabilities!

Email: security@voicera.ai

Include:
- Description of vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## Performance Guidelines

### Backend

```python
# Use async/await
async def get_agent(agent_id: str) -> Agent:
    return await Agent.find_by_id(agent_id)

# Use indexes for database queries
# Add caching for frequently accessed data
@cache(expire=300)  # 5 minute cache
async def get_agent_config(agent_id: str):
    return await Agent.find_by_id(agent_id)

# Avoid N+1 queries
agents = await Agent.find_many().populate("campaigns")
```

### Frontend

```typescript
// Use React.memo for expensive components
const ExpensiveComponent = React.memo(({ data }) => {
  return <div>{data}</div>;
});

// Use useMemo for expensive calculations
const memoizedValue = useMemo(() => {
  return expensiveCalculation(a, b);
}, [a, b]);

// Use useCallback to avoid unnecessary rerenders
const handleClick = useCallback(() => {
  // ...
}, [dependencies]);

// Code split large components
const HeavyComponent = dynamic(() => import('./Heavy'));
```

## Documentation Standards

### Code Comments

```python
# Good: Explains WHY, not WHAT
# We cache the agent config because it's accessed frequently
# and rarely changes, reducing database load by ~40%
agent_config = cache.get_or_fetch(...)

# Bad: Explains what the code does (obvious from code)
# Set agent_config variable
agent_config = cache.get_or_fetch(...)
```

### README Files

Each service should have:
- Quick description
- Prerequisites
- Installation steps
- Configuration options
- Running instructions
- Troubleshooting

### API Documentation

Use docstrings:

```python
@router.post("/agents", response_model=Agent)
async def create_agent(
    agent: AgentCreate,
    current_user: User = Depends(get_current_user)
) -> Agent:
    """Create a new voice agent.
    
    Args:
        agent: Agent creation data
        current_user: Current authenticated user
        
    Returns:
        Created agent
        
    Raises:
        HTTPException: If agent creation fails
        
    Example:
        POST /agents
        {
            "name": "Support Bot",
            "llm_provider": "openai"
        }
    """
    return await AgentService.create(agent, current_user)
```

## Running Tests

```bash
# Backend
cd voicera_backend
pytest                          # Run all tests
pytest tests/test_auth.py       # Run single file
pytest -v                       # Verbose output
pytest --cov=app               # With coverage
pytest -k "test_create"        # Run tests matching pattern

# Frontend
cd voicera_frontend
npm test                        # Run tests
npm test -- --watch            # Watch mode
npm test -- --coverage         # With coverage
```

## CI/CD

### GitHub Actions

Tests run automatically on:
- Push to main branch
- All pull requests
- Scheduled nightly builds

Check status in PR: "All checks passed" ✅

## Release Process

### Versioning

Follow semantic versioning: `MAJOR.MINOR.PATCH`

- MAJOR: Breaking changes
- MINOR: New features
- PATCH: Bug fixes

Example: `v1.2.3`

### Creating Release

```bash
# Update version in setup.py/package.json
# Update CHANGELOG.md
git tag -a v1.2.3 -m "Release v1.2.3"
git push origin v1.2.3
# GitHub creates release automatically
```

## Getting Help

- **Questions**: Open GitHub discussion
- **Bugs**: Report on GitHub issues
- **Security**: Email security@voicera.ai
- **Chat**: Join our Slack/Discord community
- **Docs**: Check documentation

## Useful Resources

- [Local Development Setup](../development/local-setup.md)
- [Architecture](../architecture/overview.md)
- [API Documentation](../api/rest-api.md)
- [Style Guide](style-guide.md)
- [Roadmap](roadmap.md)

## Thank You!

Thank you for contributing to VoiceERA! Your efforts help make this project better for everyone.

---

For questions, reach out on GitHub or community channels. Happy coding! 🚀
